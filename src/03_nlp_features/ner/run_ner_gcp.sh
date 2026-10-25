#!/usr/bin/env bash
# run_ner_gcp.sh
#
# The ONLY part of this pipeline that needs your laptop:
#   - uploading the input parquet files to GCS
#   - submitting the Cloud Build job (returns in seconds)
#
# After this script exits you can close your laptop.
# Cloud Build builds the container on GCP, then Cloud Run executes the NER job.
# Both bill only for the seconds they are actually running and stop automatically.
#
# Prerequisites (one-time)
# ------------------------
#   gcloud auth login
#   gcloud auth configure-docker us-central1-docker.pkg.dev
#   gcloud config set project <GCP_PROJECT_ID>
#
# Usage (from the repo root — the folder containing )
# ---------------------------------------------------------
#   bash run_ner_gcp.sh
#
# Skip re-uploading files on subsequent runs:
#   bash run_ner_gcp.sh --skip-upload
#
# Monitor progress after closing your laptop:
#   https://console.cloud.google.com/cloud-build/builds?project=<GCP_PROJECT_ID>
#   https://console.cloud.google.com/run/jobs?project=<GCP_PROJECT_ID>

set -euo pipefail

PROJECT="<GCP_PROJECT_ID>"
BUCKET="socialmediaanalyticsproject"
REGION="us-central1"

SKIP_UPLOAD=false
for arg in "$@"; do
  [[ $arg == "--skip-upload" ]] && SKIP_UPLOAD=true
done

# ── 1. Upload input data ──────────────────────────────────────────────────────

if [ "$SKIP_UPLOAD" = false ]; then
  echo "=== Uploading input files to gs://${BUCKET}/input/ ==="
  gsutil -m cp \
    Data_Cleaned/yt_videos_with_local_transcripts.parquet \
    Data_Cleaned/yt_comments_1_cleaned.parquet \
    Data_Cleaned/yt_comments_2_cleaned.parquet \
    Data_Cleaned/yt_comments_3_cleaned.parquet \
    Data_Cleaned/yt_comments_4_cleaned.parquet \
    "gs://${BUCKET}/input/"
  echo "Upload complete."
else
  echo "=== Skipping upload (--skip-upload) ==="
fi

# ── 2. Enable required APIs (idempotent) ──────────────────────────────────────

echo "=== Enabling GCP APIs ==="
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJECT}" --quiet

# ── 3. Ensure Artifact Registry repo exists ───────────────────────────────────

gcloud artifacts repositories describe creator-analytics \
  --location="${REGION}" --project="${PROJECT}" --quiet 2>/dev/null \
  || gcloud artifacts repositories create creator-analytics \
       --repository-format=docker \
       --location="${REGION}" \
       --project="${PROJECT}" \
       --quiet

# ── 4. Grant Cloud Build SA permission to deploy Cloud Run Jobs ───────────────
# Cloud Build needs run.admin to create/execute Cloud Run Jobs, and
# iam.serviceAccountUser to attach the job's service account.

CLOUDBUILD_SA="$(gcloud projects describe "${PROJECT}" \
  --format='value(projectNumber)')@cloudbuild.gserviceaccount.com"

gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/run.admin" --quiet

gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/iam.serviceAccountUser" --quiet

# ── 5. Submit Cloud Build — async, returns immediately ────────────────────────

echo "=== Submitting Cloud Build (async) ==="
gcloud builds submit . \
  --config  "cloudbuild-ner.yaml" \
  --project "${PROJECT}" \
  --async

echo ""
echo "=================================================="
echo " All done. You can now close your laptop."
echo "=================================================="
echo ""
echo " Track the build:"
echo "   https://console.cloud.google.com/cloud-build/builds?project=${PROJECT}"
echo ""
echo " Track the NER job:"
echo "   https://console.cloud.google.com/run/jobs?project=${PROJECT}"
echo ""
echo " Download results when the job finishes:"
echo "   gsutil -m cp -r gs://${BUCKET}/output/ner/ Data_Cleaned/"
echo ""
