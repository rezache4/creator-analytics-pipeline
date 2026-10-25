#!/usr/bin/env bash
# run_thumbnail_batch.sh
#
# Two-step workflow:
#
#   Step 1 — build the container image (run once, or after code changes):
#     bash run_thumbnail_batch.sh
#     # Wait ~60–90 min for Cloud Build to finish, then run step 2.
#
#   Step 2 — submit the GCP Batch job (spot VMs):
#     bash run_thumbnail_batch.sh --submit-batch
#     bash run_thumbnail_batch.sh --submit-batch --tasks 100
#
# Flags (can be combined):
#   --skip-upload    skip gsutil cp of thumbnails + metadata (already in GCS)
#   --submit-batch   submit the Batch job instead of kicking off Cloud Build
#   --tasks N        number of parallel tasks (default 50)
#
# Monitor:
#   https://console.cloud.google.com/cloud-build/builds?project=<GCP_PROJECT_ID>
#   gcloud batch jobs list --location=us-central1 --project=<GCP_PROJECT_ID>
#
# After all tasks finish, merge the output shards:
#   python thumbnail_cv/merge_thumbnail_features.py

set -euo pipefail

PROJECT="<GCP_PROJECT_ID>"
BUCKET="socialmediaanalyticsproject"
REGION="us-central1"
SA="<email>"
REPO="creator-analytics"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/thumbnail-job:latest"

TASKS=50
PARALLELISM=8
SKIP_UPLOAD=false
SUBMIT_BATCH=false
CLOUD_RUN=false
THUMB_LOCAL_DIR="thumbnail_cv/thumbnails"
META_LOCAL="clean_data/yt_videos_with_local_transcripts.parquet"

# Parse flags
i=1
while [[ $i -le $# ]]; do
  arg="${!i}"
  case "$arg" in
    --skip-upload)   SKIP_UPLOAD=true ;;
    --submit-batch)  SUBMIT_BATCH=true ;;
    --cloud-run)     CLOUD_RUN=true ;;
    --tasks)         i=$((i + 1)); TASKS="${!i}" ;;
    --tasks=*)       TASKS="${arg#*=}" ;;
    --parallelism)   i=$((i + 1)); PARALLELISM="${!i}" ;;
    --parallelism=*) PARALLELISM="${arg#*=}" ;;
  esac
  i=$((i + 1))
done

# ── Submit Cloud Run Job ──────────────────────────────────────────────────────
if [ "$CLOUD_RUN" = true ]; then
  JOB_NAME="thumbnail-features"
  echo "=== Submitting Cloud Run Job: ${JOB_NAME} (${TASKS} tasks, ${PARALLELISM} parallel) ==="

  gcloud services enable run.googleapis.com --project "${PROJECT}" --quiet

  # Create or update the job definition
  JOB_ARGS=(
    --image="${IMAGE}"
    --tasks="${TASKS}"
    --parallelism="${PARALLELISM}"
    --task-timeout=3600
    --cpu=4
    --memory=6Gi
    --set-env-vars="GCS_BUCKET=${BUCKET},BATCH_TASK_COUNT=${TASKS}"
    --service-account="${SA}"
    --region="${REGION}"
    --project="${PROJECT}"
  )

  if gcloud run jobs describe "${JOB_NAME}" --region="${REGION}" --project="${PROJECT}" --quiet &>/dev/null; then
    echo "Job exists — updating definition..."
    gcloud run jobs update "${JOB_NAME}" "${JOB_ARGS[@]}"
  else
    echo "Creating job..."
    gcloud run jobs create "${JOB_NAME}" "${JOB_ARGS[@]}"
  fi

  # Execute asynchronously — no need to keep terminal open
  gcloud run jobs execute "${JOB_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT}"

  echo ""
  echo "====================================================================="
  echo " Cloud Run Job '${JOB_NAME}' executing (${TASKS} tasks, ${PARALLELISM} concurrent)."
  echo " You can close your terminal — the job runs fully on GCP."
  echo "====================================================================="
  echo ""
  echo " Monitor:"
  echo "   https://console.cloud.google.com/run/jobs?project=${PROJECT}"
  echo "   gcloud run jobs executions list --job=${JOB_NAME} --region=${REGION} --project=${PROJECT}"
  echo ""
  echo " After all tasks complete, merge shards locally:"
  echo "   python thumbnail_cv/merge_thumbnail_features.py"
  echo ""
  exit 0
fi

# ── Submit GCP Batch job ──────────────────────────────────────────────────────
if [ "$SUBMIT_BATCH" = true ]; then
  JOB_ID="thumbnail-features-$(date +%Y%m%d-%H%M%S 2>/dev/null || python -c 'import datetime; print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))')"
  echo "=== Submitting GCP Batch job: ${JOB_ID} (${TASKS} tasks, SPOT VMs) ==="

  gcloud services enable batch.googleapis.com --project "${PROJECT}" --quiet

  # Ensure SA has GCS and Artifact Registry access
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${SA}" \
    --role="roles/storage.objectAdmin" --quiet 2>/dev/null || true
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${SA}" \
    --role="roles/artifactregistry.reader" --quiet 2>/dev/null || true

  TMP_CONFIG=$(mktemp /tmp/batch-thumbnail-XXXXXX.json)
  trap 'rm -f "${TMP_CONFIG}"' EXIT

  cat > "${TMP_CONFIG}" <<EOF
{
  "taskGroups": [{
    "taskSpec": {
      "runnables": [{
        "container": {
          "imageUri": "${IMAGE}",
          "entrypoint": "python",
          "commands": ["thumbnail_features.py"]
        }
      }],
      "computeResource": { "cpuMilli": 4000, "memoryMib": 6144 },
      "maxRetryCount": 2,
      "maxRunDuration": "3600s",
      "environment": {
        "variables": { "GCS_BUCKET": "${BUCKET}" }
      }
    },
    "taskCount": ${TASKS},
    "parallelism": ${PARALLELISM}
  }],
  "allocationPolicy": {
    "instances": [{
      "policy": {
        "provisioningModel": "SPOT",
        "machineType": "n2-standard-4"
      }
    }],
    "serviceAccount": { "email": "${SA}" }
  },
  "logsPolicy": { "destination": "CLOUD_LOGGING" }
}
EOF

  gcloud batch jobs submit "${JOB_ID}" \
    --location="${REGION}" \
    --project="${PROJECT}" \
    --config="${TMP_CONFIG}"

  echo ""
  echo "====================================================================="
  echo " Batch job '${JOB_ID}' submitted."
  echo " ${TASKS} tasks, ${PARALLELISM} running concurrently (SPOT VMs). Cost estimate: ~\$2–4 total."
  echo "====================================================================="
  echo ""
  echo " Monitor:"
  echo "   gcloud batch jobs describe ${JOB_ID} \\"
  echo "     --location=${REGION} --project=${PROJECT}"
  echo "   https://console.cloud.google.com/batch/jobs?project=${PROJECT}"
  echo ""
  echo " View task logs:"
  echo "   gcloud logging read 'resource.type=batch.googleapis.com/Job' \\"
  echo "     --project=${PROJECT} --limit=50 --format=json"
  echo ""
  echo " After all tasks complete, merge shards locally:"
  echo "   python thumbnail_cv/merge_thumbnail_features.py"
  echo ""
  exit 0
fi

# ── Build mode (default) ──────────────────────────────────────────────────────

if [ "$SKIP_UPLOAD" = false ]; then
  # Upload thumbnails
  if [ ! -d "${THUMB_LOCAL_DIR}" ]; then
    echo "ERROR: '${THUMB_LOCAL_DIR}' not found."
    echo "       Use --skip-upload if thumbnails are already in gs://${BUCKET}/thumbnails/"
    exit 1
  fi
  THUMB_COUNT=$(find "${THUMB_LOCAL_DIR}" -name "*.jpg" | wc -l | tr -d '[:space:]')
  echo "=== Uploading ${THUMB_COUNT} thumbnails to gs://${BUCKET}/thumbnails/ ==="
  gsutil -m cp "${THUMB_LOCAL_DIR}"/*.jpg "gs://${BUCKET}/thumbnails/"

  # Upload metadata parquet (videoId → title mapping used by tasks)
  if [ -f "${META_LOCAL}" ]; then
    echo "=== Uploading metadata parquet ==="
    gsutil cp "${META_LOCAL}" "gs://${BUCKET}/metadata/yt_videos_metadata.parquet"
  else
    echo "WARNING: '${META_LOCAL}' not found — clip_title_align will be NaN for all rows."
  fi
fi

# Enable required APIs
echo "=== Enabling GCP APIs ==="
gcloud services enable \
  batch.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJECT}" --quiet

# Ensure Artifact Registry repo exists
gcloud artifacts repositories describe "${REPO}" \
  --location="${REGION}" --project="${PROJECT}" --quiet 2>/dev/null \
  || gcloud artifacts repositories create "${REPO}" \
       --repository-format=docker \
       --location="${REGION}" \
       --project="${PROJECT}" \
       --quiet

# Grant Cloud Build SA permission to push images
CLOUDBUILD_SA="$(gcloud projects describe "${PROJECT}" \
  --format='value(projectNumber)')@cloudbuild.gserviceaccount.com"
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/artifactregistry.writer" --quiet

# Submit Cloud Build — returns immediately, build runs on GCP
echo "=== Submitting Cloud Build (async) ==="
gcloud builds submit thumbnail_cv \
  --config "thumbnail_cv/cloudbuild-thumbnail.yaml" \
  --project "${PROJECT}" \
  --async

echo ""
echo "====================================================================="
echo " Cloud Build submitted. First build takes ~60–90 min."
echo "====================================================================="
echo ""
echo " Track build progress:"
echo "   https://console.cloud.google.com/cloud-build/builds?project=${PROJECT}"
echo ""
echo " Once the build shows green, submit the Batch job:"
echo "   bash thumbnail_cv/run_thumbnail_batch.sh --submit-batch --skip-upload --tasks ${TASKS}"
echo ""
