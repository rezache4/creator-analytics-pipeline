"""Vertex AI client wrapper and batch-job infrastructure.

Ports the notebook's "Vertex AI Client" + "Batch Infrastructure" cells into a
``BatchClient`` class: write JSONL -> upload -> submit -> poll -> retrieve+parse,
plus the submitted-job record/lookup helpers that make submit/retrieve survive a
fresh process (the close-the-laptop property).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import time

from .config import PipelineConfig


def strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


class BatchClient:
    def __init__(self, config: PipelineConfig):
        self.config = config
        from google import genai
        from google.genai.types import CreateBatchJobConfig, JobState

        self._CreateBatchJobConfig = CreateBatchJobConfig
        self._JobState = JobState
        self.client = genai.Client(
            vertexai=True,
            project=config.gcp_project_id,
            location=config.gcp_location,
        )
        print("Vertex client ready:", config.gcp_project_id, config.gcp_location)

    # ------------------------- connectivity test -------------------------
    def connectivity_test(self) -> None:
        for stage, model in [
            ("Stage 1", self.config.model_stage1_exploratory),
            ("Stage 2", self.config.model_stage2_classify),
        ]:
            try:
                r = self.client.models.generate_content(
                    model=model, contents="Reply with the single word: OK"
                )
                print(f"OK {stage} ({model}): {r.text.strip()[:40]}")
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"FAIL {stage} ({model}): {e}")

    # ----------------------------- I/O helpers ---------------------------
    def write_jsonl(self, lines: list, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[prep] wrote {len(lines):,} requests -> {path}")
        return path

    def upload_to_gcs(self, local_path: str, blob_name: str) -> str:
        from google.cloud import storage

        c = storage.Client(project=self.config.gcp_project_id)
        c.bucket(self.config.gcs_bucket).blob(blob_name).upload_from_filename(local_path)
        uri = f"gs://{self.config.gcs_bucket}/{blob_name}"
        print(f"[upload] {local_path} -> {uri}")
        return uri

    # ---------------------------- submit / poll --------------------------
    def submit_batch_job(self, input_uri: str, output_uri: str, model: str):
        """Non-blocking submit. dest is a PREFIX; Vertex writes a unique subfolder."""
        job = self.client.batches.create(
            model=model,
            src=input_uri,
            config=self._CreateBatchJobConfig(dest=output_uri),
        )
        print(f"[submit] {model} -> {job.name}  ({job.state})")
        return job

    def poll_until_complete(self, job_name: str):
        JobState = self._JobState
        terminal = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }
        while True:
            job = self.client.batches.get(name=job_name)
            print(f"[poll {datetime.datetime.now():%H:%M:%S}] state = {job.state}")
            if job.state in terminal:
                return job
            time.sleep(self.config.poll_interval_seconds)

    # ---------------------------- retrieve -------------------------------
    def retrieve_response_texts(self, job) -> list:
        """Download every output shard, return model text per row (None on failure)."""
        from google.cloud import storage

        bucket_name = self.config.gcs_bucket
        out_loc = job.dest.gcs_uri
        prefix = out_loc.replace(f"gs://{bucket_name}/", "")
        c = storage.Client(project=self.config.gcp_project_id)
        bk = c.bucket(bucket_name)
        blobs = [b for b in bk.list_blobs(prefix=prefix) if b.name.endswith(".jsonl")]
        texts = []
        for blob in blobs:
            for line in blob.download_as_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                resp = rec.get("response")
                if not resp:
                    texts.append(None)
                    continue
                try:
                    parts = resp["candidates"][0]["content"]["parts"]
                    texts.append("".join(p.get("text", "") for p in parts))
                except (KeyError, IndexError):
                    texts.append(None)
        print(f"[retrieve] {len(texts):,} response rows from {len(blobs)} shard(s)")
        return texts

    # ------------------------- job record/lookup -------------------------
    def record_batch_job(self, tag: str, job, dest: str) -> dict:
        rec = {
            "name": job.name,
            "dest": dest,
            "state": str(job.state),
            "submitted_at": datetime.datetime.now().isoformat(),
        }
        path = f"{self.config.local_dir}/{tag}_job.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(f"[record] {tag} job -> {path}")
        return rec

    def get_recorded_job(self, tag: str):
        p = f"{self.config.local_dir}/{tag}_job.json"
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"No submitted '{tag}' job recorded at {p} - submit it first."
            )
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        job = self.client.batches.get(name=rec["name"])
        print(f"[{tag}] {rec['name']} -> {job.state}")
        return job

    @property
    def JobState(self):
        return self._JobState
