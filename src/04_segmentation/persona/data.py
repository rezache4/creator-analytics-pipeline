"""Data loading: prepared comments, reply edges, and (IG) the media index.

Ports the notebook's "Data Loading" cell into a ``DataLoader`` that returns the
``ig_comments`` feature frame and, for Instagram, the ``media_index`` frame.
"""
from __future__ import annotations

from .config import PipelineConfig


class DataLoader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def load(self):
        """Return (ig_comments, media_index). media_index is None off-IG."""
        import pandas as pd

        cfg = self.config
        print(f"Loading prepared comments for platform = {cfg.platform!r}")

        # 1) comment feature matrix, predicate-pushdown filter to PLATFORM
        ig_comments = pd.read_parquet(
            cfg.comments_ml_path, filters=[("platform", "==", cfg.platform)]
        )
        ig_comments["author_id"] = ig_comments["author_id"].astype(str)
        ig_comments["media_id"] = ig_comments["media_id"].astype(str)
        print(
            f"  comments_ml[{cfg.platform}]: {len(ig_comments):,} comments | "
            f"{ig_comments['author_id'].nunique():,} authors | "
            f"{ig_comments['media_id'].nunique():,} media"
        )

        # 2) reply structure
        try:
            replies = pd.read_parquet(
                cfg.edges_replies_path,
                filters=[("platform", "==", cfg.platform)],
                columns=["src_comment_id", "dst_comment_id"],
            )
            replies = (
                replies.rename(
                    columns={
                        "src_comment_id": "comment_id",
                        "dst_comment_id": "reply_to_comment_id",
                    }
                )
                .drop_duplicates("comment_id")
            )
            ig_comments = ig_comments.merge(replies, on="comment_id", how="left")
            print(
                f"  reply edges: {len(replies):,} | replies flagged on "
                f"{ig_comments['reply_to_comment_id'].notna().sum():,} comments"
            )
        except Exception as e:  # noqa: BLE001
            print("  edges_replies_to unavailable -> no reply features:", str(e)[:120])
            ig_comments["reply_to_comment_id"] = pd.NA

        # 3) IG-only multimodal context
        if cfg.attach_media:
            media_index = pd.read_parquet(cfg.media_index_path)
            media_index = media_index[media_index["media_id"].notna()].copy()
            media_index["media_id"] = media_index["media_id"].astype(str)
            print(
                f"  media_index: {len(media_index):,} posts | "
                f"{media_index['shortcode'].nunique():,} shortcodes"
            )
        else:
            media_index = None
            print(f"  media context OFF for platform={cfg.platform} -> comments-only run")

        # 4) validation
        n_before = len(ig_comments)
        ig_comments = ig_comments[
            ig_comments["author_id"].notna() & (ig_comments["author_id"] != "")
        ].copy()
        print(
            f"\nComments ready: {len(ig_comments):,}/{n_before:,} | "
            f"unique authors: {ig_comments['author_id'].nunique():,}"
        )
        return ig_comments, media_index
