#!/usr/bin/env bash
# tests → build → deploy. Credentials: mount the same secrets as Solain
# (GCP_CREDENTIALS_GEE / GCP_CREDENTIALS_GCS as file paths).
set -euo pipefail
cd "$(dirname "$0")"
echo "== 1/3 tests"; python3 -m pytest -q
echo "== 2/3 build"; gcloud builds submit . --config cloudbuild.yaml --region europe-west1
echo "== 3/3 deploy"
gcloud run deploy stockpile-intelligence-engine --region europe-west1 \
  --image europe-west1-docker.pkg.dev/level-sol-480011-r1/cloud-run-source-deploy/stockpile-intelligence-engine:latest \
  --memory 16Gi --cpu 4 --timeout 3600 --no-allow-unauthenticated \
  --set-env-vars STOCKPILE_ENABLE_GCS=0,STOCKPILE_ENABLE_BQ=0,STOCKPILE_ENABLE_PUBSUB=0
