"""Split the prepped comment exports per platform and fold their features into
an EXTENDED community-vibe schema — adapted to the HeteroGraph output layout.

The ``Data_Preparation_Pipeline_colab`` notebook (Section 5: "Hybrid
Heterogeneous Graph Pipeline") writes its results to ``Output/Prepared Comments/``::

    comments_llm.jsonl            — raw text per comment   {comment_id, text, platform}
    comments_ml.parquet           — numeric feature matrix  (emoji/punctuation stats)
    HeteroGraph/
        nodes_author.parquet      {author_id, platform}
        nodes_comment.parquet     {comment_id, author_id, media_id, platform,
                                   timestamp, sentiment_neg/neu/pos, topic_vec_dim}
        nodes_media.parquet       {media_id, platform, parent_media_id, is_short, format_type}
        edges_posted.parquet      author --POSTED--> comment
        edges_belongs_to.parquet  comment --BELONGS_TO--> media
        edges_replies_to.parquet  comment --REPLIES_TO--> comment   (src replies to dst)
        edges_derived_from.parquet  media --DERIVED_FROM--> media   (Short → LongForm, YouTube)
        edges_similar_to.parquet  comment --SIMILAR_TO--> comment   (cosine, TikTok)

This REPLACES the legacy homogeneous ``comments_gml.parquet``: per-comment
media/author/timestamp now live in ``nodes_comment``, and the reply structure
lives in ``edges_replies_to``. Every node/edge table already carries a
``platform`` column, so the data is multi-platform at the source
(today: youtube, instagram, tiktok — YouTube being the largest).

This module:

  1. ``separate_by_platform`` — fan the LLM jsonl, the ML parquet, and every
     HeteroGraph node/edge table out into
     ``by_platform/<platform>/{comments_llm.jsonl, comments_ml.parquet, HeteroGraph/*.parquet}``
     so each platform can be processed independently.

  2. ``PrepedCommentsLoader.load_for_pipeline`` — rebuild a raw-schema comment
     DataFrame (text + ids + reply ref) from ``nodes_comment`` + ``edges_replies_to``
     + the LLM text, named so the base ``SchemaNormalizer.normalize_comments``
     accepts it — no recompute needed.

  3. ``aggregate_community_features`` — turn the *precomputed* ML features into
     per-``media_id`` deterministic signals that EXTEND the LLM community vibe
     (emoji intensity/diversity, reply depth, interrogative/exclamatory mix,
     link/mention/hashtag pressure, author concentration). ``media_id`` /
     ``author_id`` are joined from ``nodes_comment`` (not assumed on the ML
     table), and reply depth comes from ``edges_replies_to``.

The ``DERIVED_FROM`` (Short→Long) and ``SIMILAR_TO`` edges are split per platform
too but are not folded into the per-media vibe; they are consumed by the
graph-analytics / PyG downstreams.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("parse_comments")

# Raw IG-compatible column names (so the base SchemaNormalizer accepts them).
COL_COMMENT_ID = "comment_id"
COL_MEDIA_ID = "media_id"
COL_TEXT = "text"
COL_AUTHOR = "author_id"
COL_REPLY = "reply_to_comment_id"
COL_TIMESTAMP = "timestamp"
COL_PLATFORM = "platform"

# HeteroGraph edge/node column names.
COL_SRC_COMMENT = "src_comment_id"
COL_DST_COMMENT = "dst_comment_id"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrepedConfig:
    input_dir: str = "Output/Prepared Comments"
    output_dir: str = "Output/Prepared Comments/by_platform"
    hetero_subdir: str = "HeteroGraph"
    llm_name: str = "comments_llm.jsonl"
    ml_name: str = "comments_ml.parquet"
    # HeteroGraph tables (split per platform alongside the flat exports).
    node_names: tuple = ("nodes_author.parquet", "nodes_comment.parquet", "nodes_media.parquet")
    edge_names: tuple = (
        "edges_posted.parquet",
        "edges_belongs_to.parquet",
        "edges_replies_to.parquet",
        "edges_derived_from.parquet",
        "edges_similar_to.parquet",
    )
    nodes_comment_name: str = "nodes_comment.parquet"
    edges_replies_name: str = "edges_replies_to.parquet"


# --------------------------------------------------------------------------- #
# Extended community-vibe schema (deterministic, from prepped features)
# --------------------------------------------------------------------------- #
@dataclass
class ExtendedCommunityVibeSchema:
    """The per-media columns produced by ``aggregate_community_features``.

    These EXTEND the LLM community-vibe fields (sentiment_polarization_index,
    topical_adherence_score, dominant_community_emotion) with exact, cheap
    signals derived from the prepped ML/HeteroGraph comment features.
    """

    columns: List[str] = field(
        default_factory=lambda: [
            "comment_volume",          # number of comments on the media
            "unique_authors",          # distinct commenters
            "author_concentration",    # 1 - unique_authors/comment_volume (0=all distinct)
            "reply_ratio",             # share of comments that are replies (REPLIES_TO edges)
            "mean_word_count",         # avg comment length (words)
            "mean_emoji_per_word",     # emoji intensity
            "mean_emoji_entropy",      # emoji diversity (Shannon)
            "mean_emoji_variety",      # unique/total emoji ratio
            "interrogative_ratio",     # share of comments with a '?'
            "exclamatory_ratio",       # share of comments with a '!'
            "link_ratio",              # share of comments containing a URL
            "mention_ratio",           # share with @mention
            "hashtag_ratio",           # share with #hashtag
        ]
    )


# --------------------------------------------------------------------------- #
# 1) Separation per platform
# --------------------------------------------------------------------------- #
def separate_by_platform(config: Optional[PrepedConfig] = None) -> Dict[str, Dict[str, str]]:
    """Fan the combined exports out into one folder per platform.

    Returns a manifest ``{platform: {kind: path, "<kind>_rows": n}}`` and writes
    a ``manifest.json`` alongside the splits. Sources missing on disk (e.g. the
    ML parquet while Drive is still syncing) are skipped with a warning.
    """
    cfg = config or PrepedConfig()
    in_dir, out_dir = Path(cfg.input_dir), Path(cfg.output_dir)
    hetero_in = in_dir / cfg.hetero_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, str]] = {}

    # --- LLM jsonl: stream so we never hold 600MB+ in memory ---------------
    llm_path = in_dir / cfg.llm_name
    if llm_path.exists():
        handles: Dict[str, Any] = {}
        counts: Dict[str, int] = {}
        try:
            with llm_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        plat = json.loads(line).get(COL_PLATFORM, "unknown")
                    except json.JSONDecodeError:
                        plat = "unknown"
                    if plat not in handles:
                        pdir = out_dir / plat
                        pdir.mkdir(parents=True, exist_ok=True)
                        handles[plat] = (pdir / cfg.llm_name).open("w", encoding="utf-8")
                        counts[plat] = 0
                    handles[plat].write(line + "\n")
                    counts[plat] += 1
        finally:
            for h in handles.values():
                h.close()
        for plat, n in counts.items():
            manifest.setdefault(plat, {})["llm"] = str(out_dir / plat / cfg.llm_name)
            manifest[plat]["llm_rows"] = str(n)
        LOGGER.info("LLM split: %s", {p: c for p, c in counts.items()})
    else:
        LOGGER.warning("Missing %s", llm_path)

    # --- ML parquet: groupby platform, write per group ---------------------
    # The ML table may not carry a 'platform' column; backfill it from
    # nodes_comment (comment_id -> platform) when absent.
    ml_path = in_dir / cfg.ml_name
    if ml_path.exists():
        ml = pd.read_parquet(ml_path)
        if COL_PLATFORM not in ml.columns:
            nc_path = hetero_in / cfg.nodes_comment_name
            if nc_path.exists():
                plat_map = pd.read_parquet(nc_path, columns=[COL_COMMENT_ID, COL_PLATFORM])
                ml = ml.merge(plat_map, on=COL_COMMENT_ID, how="left")
                ml[COL_PLATFORM] = ml[COL_PLATFORM].fillna("unknown")
            else:
                LOGGER.warning("%s has no '%s' and %s is missing; writing as 'unknown'.",
                               cfg.ml_name, COL_PLATFORM, nc_path)
                ml[COL_PLATFORM] = "unknown"
        for plat, sub in ml.groupby(COL_PLATFORM):
            pdir = out_dir / str(plat)
            pdir.mkdir(parents=True, exist_ok=True)
            dest = pdir / cfg.ml_name
            sub.drop(columns=[COL_PLATFORM]).to_parquet(dest, compression="snappy", index=False)
            manifest.setdefault(str(plat), {})["ml"] = str(dest)
            manifest[str(plat)]["ml_rows"] = str(len(sub))
        LOGGER.info("ML split: %s", ml[COL_PLATFORM].value_counts().to_dict())
    else:
        LOGGER.warning("Missing %s (Drive still syncing?) — ML split skipped.", ml_path)

    # --- HeteroGraph node/edge tables: groupby their 'platform' column -----
    for name in (*cfg.node_names, *cfg.edge_names):
        ppath = hetero_in / name
        if not ppath.exists():
            LOGGER.warning("Missing %s", ppath)
            continue
        df = pd.read_parquet(ppath)
        if COL_PLATFORM not in df.columns:
            LOGGER.warning("%s has no '%s' column; writing as 'unknown'.", name, COL_PLATFORM)
            df[COL_PLATFORM] = "unknown"
        for plat, sub in df.groupby(COL_PLATFORM):
            hdir = out_dir / str(plat) / cfg.hetero_subdir
            hdir.mkdir(parents=True, exist_ok=True)
            dest = hdir / name
            sub.drop(columns=[COL_PLATFORM]).to_parquet(dest, compression="snappy", index=False)
            manifest.setdefault(str(plat), {})[name] = str(dest)
            manifest[str(plat)][f"{name}_rows"] = str(len(sub))
        LOGGER.info("%s split: %s", name, df[COL_PLATFORM].value_counts().to_dict())

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest → %s", out_dir / "manifest.json")
    return manifest


# --------------------------------------------------------------------------- #
# 2 + 3) Per-platform loading & extended-vibe aggregation
# --------------------------------------------------------------------------- #
class PrepedCommentsLoader:
    """Read a platform's split and expose pipeline-ready / aggregate views.

    Reads from the ``by_platform/<platform>/`` splits produced by
    ``separate_by_platform`` (run that first).
    """

    def __init__(self, config: Optional[PrepedConfig] = None) -> None:
        self.cfg = config or PrepedConfig()

    def _pdir(self, platform: str) -> Path:
        return Path(self.cfg.output_dir) / platform

    def _hdir(self, platform: str) -> Path:
        return self._pdir(platform) / self.cfg.hetero_subdir

    def _read_llm(self, platform: str) -> pd.DataFrame:
        path = self._pdir(platform) / self.cfg.llm_name
        rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    def _reply_parent_map(self, platform: str) -> pd.DataFrame:
        """``edges_replies_to`` as a comment_id → reply_to_comment_id frame.

        REPLIES_TO is directed ``src --replies to--> dst``, so for each reply the
        parent is the dst. Platforms without the edge (e.g. TikTok) yield an
        empty frame.
        """
        path = self._hdir(platform) / self.cfg.edges_replies_name
        if not path.exists():
            return pd.DataFrame(columns=[COL_COMMENT_ID, COL_REPLY])
        edges = pd.read_parquet(path, columns=[COL_SRC_COMMENT, COL_DST_COMMENT])
        return edges.rename(
            columns={COL_SRC_COMMENT: COL_COMMENT_ID, COL_DST_COMMENT: COL_REPLY}
        ).drop_duplicates(subset=[COL_COMMENT_ID])

    def load_for_pipeline(
        self, platform: str, media_ids: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Raw-schema comments (text + ids + reply ref) for the vibe pipelines.

        Built from ``nodes_comment`` (ids + timestamp) + ``edges_replies_to``
        (reply ref) + the LLM text. Output columns are named so the base
        ``SchemaNormalizer.normalize_comments(df, platform)`` accepts them.
        """
        nc = pd.read_parquet(
            self._hdir(platform) / self.cfg.nodes_comment_name,
            columns=[COL_COMMENT_ID, COL_AUTHOR, COL_MEDIA_ID, COL_TIMESTAMP],
        )
        if media_ids is not None:
            keep = set(map(str, media_ids))
            nc = nc[nc[COL_MEDIA_ID].astype(str).isin(keep)]

        replies = self._reply_parent_map(platform)
        merged = nc.merge(replies, on=COL_COMMENT_ID, how="left")

        llm = self._read_llm(platform)
        merged = merged.merge(llm[[COL_COMMENT_ID, COL_TEXT]], on=COL_COMMENT_ID, how="left")

        out = merged[[COL_COMMENT_ID, COL_MEDIA_ID, COL_TEXT, COL_AUTHOR,
                      COL_REPLY, COL_TIMESTAMP]].copy()
        out[COL_PLATFORM] = platform
        LOGGER.info("[%s] pipeline comments: %d rows across %d media.",
                    platform, len(out), out[COL_MEDIA_ID].nunique())
        return out

    def aggregate_community_features(
        self, platform: str, media_ids: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Per-``media_id`` extended community-vibe features (deterministic)."""
        ml_path = self._pdir(platform) / self.cfg.ml_name
        if not ml_path.exists():
            raise FileNotFoundError(
                f"{ml_path} not found. The ML feature matrix is required for "
                f"aggregate_community_features; ensure comments_ml.parquet has "
                f"finished downloading and separate_by_platform has run."
            )
        ml = pd.read_parquet(ml_path)

        # media_id / author_id are authoritative on nodes_comment; join them in
        # rather than assuming the ML table still carries them.
        nc = pd.read_parquet(
            self._hdir(platform) / self.cfg.nodes_comment_name,
            columns=[COL_COMMENT_ID, COL_MEDIA_ID, COL_AUTHOR],
        )
        ml = ml.merge(nc, on=COL_COMMENT_ID, how="left", suffixes=("", "_nc"))
        for col in (COL_MEDIA_ID, COL_AUTHOR):
            if f"{col}_nc" in ml.columns:
                ml[col] = ml[col].where(ml[col].notna(), ml[f"{col}_nc"])
                ml = ml.drop(columns=[f"{col}_nc"])

        if media_ids is not None:
            keep = set(map(str, media_ids))
            ml = ml[ml[COL_MEDIA_ID].astype(str).isin(keep)]

        # Which comments are replies (src of a REPLIES_TO edge)?
        reply_src = set(self._reply_parent_map(platform)[COL_COMMENT_ID].astype(str))

        # Boolean helpers from the ML feature columns.
        ml = ml.assign(
            _has_q=(ml.get("question_count", 0) > 0).astype(int),
            _has_excl=(ml.get("exclamation_count", 0) > 0).astype(int),
            _has_mention=(ml.get("mention_count", 0) > 0).astype(int),
            _has_hashtag=(ml.get("hashtag_count", 0) > 0).astype(int),
            _is_reply=ml[COL_COMMENT_ID].astype(str).isin(reply_src).astype(int),
        )
        g = ml.groupby(COL_MEDIA_ID)
        agg = pd.DataFrame({
            "comment_volume": g.size(),
            "unique_authors": g[COL_AUTHOR].nunique(),
            "mean_word_count": g["word_count"].mean(),
            "mean_emoji_per_word": g["emoji_per_word_ratio"].mean(),
            "mean_emoji_entropy": g["emoji_entropy"].mean(),
            "mean_emoji_variety": g["emoji_variety_ratio"].mean(),
            "interrogative_ratio": g["_has_q"].mean(),
            "exclamatory_ratio": g["_has_excl"].mean(),
            "link_ratio": g["has_links"].mean(),
            "mention_ratio": g["_has_mention"].mean(),
            "hashtag_ratio": g["_has_hashtag"].mean(),
            "reply_ratio": g["_is_reply"].mean(),
        })
        agg["author_concentration"] = 1.0 - (agg["unique_authors"] / agg["comment_volume"]).clip(0, 1)

        # Order to the documented schema + downcast.
        cols = ExtendedCommunityVibeSchema().columns
        agg = agg.reset_index()[[COL_MEDIA_ID, *cols]]
        float_cols = agg.select_dtypes("float").columns
        agg[float_cols] = agg[float_cols].apply(pd.to_numeric, downcast="float")
        LOGGER.info("[%s] extended community features for %d media.", platform, len(agg))
        return agg


# --------------------------------------------------------------------------- #
# Main — separate, then preview the extended vibe for a few media.
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    manifest = separate_by_platform()
    LOGGER.info("Manifest:\n%s", json.dumps(manifest, indent=2))

    loader = PrepedCommentsLoader()
    for platform in manifest:
        try:
            ext = loader.aggregate_community_features(platform)
        except FileNotFoundError as exc:
            LOGGER.warning("[%s] skipping aggregate: %s", platform, exc)
            continue
        top = ext.sort_values("comment_volume", ascending=False).head(5)
        pd.set_option("display.max_columns", None, "display.width", 240)
        LOGGER.info("[%s] extended community-vibe (top-5 by volume):\n%s",
                    platform, top.to_string(index=False))


if __name__ == "__main__":
    main()