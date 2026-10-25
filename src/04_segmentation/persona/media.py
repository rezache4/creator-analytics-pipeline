"""Multimodal media context builder.

Ports the notebook's "Multimodal Media Context" cell into ``MediaContextBuilder``.
It bridges each commenter to the media they engage with most and emits the
multipart payload (post-metadata text + image ``fileData`` parts + transcript
text) attached to that user's Stage 1 / Stage 2 request.

Transcripts are read from the LOCAL mirror at ``media_local_prefix`` when present
(no GCS round-trip in the build loop), falling back to GCS download otherwise.
Image ``fileData`` URIs always stay ``gs://`` - Vertex fetches those server-side.
For non-Instagram platforms this builder is inert (``build_user_media_parts``
returns ``[]``).
"""
from __future__ import annotations

import glob
import os

from .config import PipelineConfig


class MediaContextBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.enabled = config.attach_media
        self.user_top_posts: dict = {}
        self.media_images: dict = {}
        self.media_transcripts: dict = {}
        self.media_transcripts_local: dict = {}
        self._mediaid_to_shortcode: dict = {}
        self._meta_by_shortcode: dict = {}
        self._transcript_cache: dict = {}

    # ------------------------------ helpers ------------------------------
    @staticmethod
    def _norm_id(x) -> str:
        s = str(x)
        return s[:-2] if s.endswith(".0") else s

    @staticmethod
    def _txt(v) -> str:
        return v.strip() if isinstance(v, str) else ""

    def _as_list_str(self, v) -> str:
        import numpy as np

        if isinstance(v, (list, tuple, np.ndarray)):
            return ", ".join(str(x) for x in v if str(x) not in ("nan", "None", ""))
        return self._txt(v)

    # ------------------------------ build --------------------------------
    def build(self, ig_comments, media_index, user_features) -> None:
        """Populate maps and manifests. No-op (text-only) when media is disabled."""
        cfg = self.config
        if not self.enabled:
            print(f"Media context disabled for platform={cfg.platform}; requests will be text-only.")
            return

        # 1) media index maps
        mi = media_index[media_index["media_id"].notna()].copy()
        mi["media_id"] = mi["media_id"].map(self._norm_id)
        self._mediaid_to_shortcode = dict(zip(mi["media_id"], mi["shortcode"]))
        self._meta_by_shortcode = {r["shortcode"]: r for r in mi.to_dict("records")}

        # 2) per-user top engaged posts (by comment volume)
        tp = ig_comments[["author_id", "media_id"]].dropna().copy()
        tp["author_id"] = tp["author_id"].astype(str)
        tp["media_id"] = tp["media_id"].map(self._norm_id)
        counts = (
            tp.groupby(["author_id", "media_id"]).size()
            .reset_index(name="n")
            .sort_values(["author_id", "n"], ascending=[True, False])
        )
        self.user_top_posts = (
            counts.groupby("author_id")["media_id"]
            .apply(lambda s: list(s.head(cfg.max_media_posts_per_user)))
            .to_dict()
        )
        print(f"user_top_posts computed for {len(self.user_top_posts):,} users.")

        # 3) GCS media manifest (one list pass)
        self.media_images, self.media_transcripts = self._build_media_manifest()

        # 3b) LOCAL transcript manifest (mirror of the GCS tree)
        self._build_local_transcript_manifest()

        # 4) demo
        demo = next(
            (a for a in user_features["author_id"].astype(str) if self.build_user_media_parts(a)),
            None,
        )
        print(
            "builders ready. Example user media parts:",
            len(self.build_user_media_parts(demo)) if demo else 0,
        )

    def _build_media_manifest(self):
        from google.cloud import storage

        cfg = self.config
        c = storage.Client(project=cfg.gcp_project_id)
        images, transcripts = {}, {}
        for blob in c.list_blobs(cfg.gcs_bucket, prefix=cfg.media_gcs_prefix):
            name = blob.name
            low = name.lower()
            segs = name.split("/")  # multimodal_dataset_fixed/<form>/<shortcode>/...
            if len(segs) < 3:
                continue
            sc = segs[2]
            uri = f"gs://{cfg.gcs_bucket}/{name}"
            if low.endswith((".jpg", ".jpeg", ".png")):
                images.setdefault(sc, []).append(uri)
            elif low.endswith(".txt") and "transcri" in low:
                transcripts.setdefault(sc, []).append(uri)
        for d in (images, transcripts):
            for k in d:
                d[k].sort()
        print(
            f"manifest: {len(images):,} posts with images, "
            f"{len(transcripts):,} with transcripts."
        )
        return images, transcripts

    def _build_local_transcript_manifest(self) -> None:
        cfg = self.config
        prefix = cfg.media_local_prefix
        local: dict = {}
        if os.path.isdir(prefix):
            base = os.path.basename(prefix)
            for p in glob.iglob(os.path.join(prefix, "*", "*", "**", "*.txt"), recursive=True):
                if "transcri" not in os.path.basename(p).lower():
                    continue
                segs = p.replace(chr(92), "/").split("/")  # <prefix>/<form>/<shortcode>/...
                try:
                    sc = segs[segs.index(base) + 2]
                except (ValueError, IndexError):
                    continue
                local.setdefault(sc, []).append(p)
            for k in local:
                local[k].sort()
            print(f"local transcripts: {len(local):,} posts under {prefix}/")
        else:
            print(f"local media folder {prefix!r} not found - using GCS for transcripts.")
        self.media_transcripts_local = local

    # ------------------------- transcript fetch --------------------------
    def fetch_transcript(self, sc: str) -> str:
        if sc in self._transcript_cache:
            return self._transcript_cache[sc]
        chunks = []
        local_paths = self.media_transcripts_local.get(sc)
        if local_paths:  # fast path: local disk, no network
            for lp in local_paths[:8]:  # carousel_video has one .txt per slide
                try:
                    with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                        chunks.append(f.read())
                except Exception:  # noqa: BLE001
                    pass
        else:  # fallback: download from GCS
            from google.cloud import storage

            c = storage.Client(project=self.config.gcp_project_id)
            for uri in self.media_transcripts.get(sc, [])[:8]:
                try:
                    chunks.append(storage.Blob.from_string(uri, client=c).download_as_text())
                except Exception:  # noqa: BLE001
                    pass
        txt = " ".join(x.strip() for x in chunks if x.strip())
        self._transcript_cache[sc] = txt
        return txt

    def _pick_images(self, sc: str, k: int):
        uris = self.media_images.get(sc, [])
        if len(uris) <= k:
            return uris
        step = len(uris) / k
        return [uris[int(i * step)] for i in range(k)]

    def _format_post_meta_text(self, sc: str) -> str:
        row = self._meta_by_shortcode.get(sc, {})
        cf = self._txt(row.get("content_form"))
        bits = [f"POST [{cf or 'post'}] shortcode={sc}"]
        cap = self._txt(row.get("caption"))
        if cap:
            bits.append(f"caption: {cap[:300]}")
        tg = self._as_list_str(row.get("tagged_usernames"))
        if tg:
            bits.append(f"tagged: {tg}")
        co = self._as_list_str(row.get("coauthors"))
        if co:
            bits.append(f"coauthors: {co}")
        song = self._txt(row.get("song_title"))
        artist = self._txt(row.get("artist"))
        atype = self._txt(row.get("audio_type"))
        if song or artist:
            bits.append(f"music: {song} - {artist} ({atype})")
        loc = self._txt(row.get("location_name"))
        if loc:
            bits.append(f"location: {loc}")
        return " | ".join(bits)

    def build_user_media_parts(self, author_id) -> list:
        """Multipart payload for a user's most-engaged posts."""
        if not self.enabled:
            return []
        cfg = self.config
        parts = []
        for media_id in self.user_top_posts.get(str(author_id), []):
            sc = self._mediaid_to_shortcode.get(self._norm_id(media_id))
            if not sc or sc not in self.media_images:
                continue
            parts.append({"text": self._format_post_meta_text(sc)})
            for uri in self._pick_images(sc, cfg.max_images_per_post):
                parts.append({"fileData": {"fileUri": uri, "mimeType": "image/jpeg"}})
            if cfg.include_transcript:
                t = self.fetch_transcript(sc)
                if t:
                    parts.append({"text": "transcript: " + t[: cfg.max_transcript_chars]})
        return parts
