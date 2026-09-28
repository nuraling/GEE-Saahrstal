#!/usr/bin/env bash
# One-time creation of the cloud resources this engine writes to.
# Reuses Solain's GCP project and service accounts; creates NEW, separate
# bucket / dataset / topics so stockpile data never mixes with Solain's.
#
#   PROJECT=level-sol-480011-r1 REGION=europe-west1 ./infra/setup_gcp.sh
set -euo pipefail
PROJECT="${PROJECT:-level-sol-480011-r1}"
REGION="${REGION:-europe-west1}"
BUCKET="${BUCKET:-stockpile-intelligence-engine}"
DATASET="${DATASET:-stockpile_engine}"
JOBS_TOPIC="${JOBS_TOPIC:-stockpile-processing-queue}"
STATUS_TOPIC="${STATUS_TOPIC:-stockpile-processing-status}"
# the account Solain writes GCS/BigQuery with (GCP_CREDENTIALS_GCS)
SA="${SA:-geospatial-engine-226@${PROJECT}.iam.gserviceaccount.com}"

echo "== bucket gs://${BUCKET} (outputs land under stockpile/<project>/<date>/)"
gcloud storage buckets describe "gs://${BUCKET}" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${SA}" --role roles/storage.objectAdmin >/dev/null

echo "== BigQuery dataset ${DATASET} (tables are created by the engine on first write)"
bq --project_id "$PROJECT" show "$DATASET" >/dev/null 2>&1 || \
  bq --project_id "$PROJECT" --location EU mk --dataset "${PROJECT}:${DATASET}"

echo "== Pub/Sub topics"
for t in "$JOBS_TOPIC" "$STATUS_TOPIC"; do
  gcloud pubsub topics describe "$t" --project "$PROJECT" >/dev/null 2>&1 || \
    gcloud pubsub topics create "$t" --project "$PROJECT"
done

cat <<MSG

Done. To switch the sinks on for the Cloud Run service:
  gcloud run services update stockpile-intelligence-engine --region ${REGION} \\
    --update-env-vars STOCKPILE_ENABLE_GCS=1,STOCKPILE_ENABLE_BQ=1,STOCKPILE_ENABLE_PUBSUB=1

Push subscription (jobs → service), after the first deploy:
  gcloud pubsub subscriptions create stockpile-jobs-push --topic ${JOBS_TOPIC} \\
    --push-endpoint "\$(gcloud run services describe stockpile-intelligence-engine --region ${REGION} --format 'value(status.url)')/" \\
    --ack-deadline 600 --project ${PROJECT}
MSG
