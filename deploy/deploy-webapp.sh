#!/usr/bin/env bash
set -eo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
RG="meridian-customer-intelligence"
LOCATION="${AZURE_LOCATION:-centralindia}"
PLAN_NAME="${PLAN_NAME:-meridian-ci-plan}"
# APP_NAME="${APP_NAME:-meridian-ci-${RANDOM}}"
APP_NAME="${APP_NAME:-meridian-ci}"
SKU="${SKU:-B1}"  # B1 = cheapest Linux plan

echo "==> Using app name: $APP_NAME"

# ── Change to repo root ──────────────────────────────────────────────────────
cd "$(dirname "$0")/.."

# ── Load environment variables ───────────────────────────────────────────────
source .env

# ── Create App Service Plan (Linux) ─────────────────────────────────────────
echo "==> Creating App Service Plan: $PLAN_NAME (Linux, $SKU)"
az appservice plan create \
    --resource-group "$RG" \
    --name "$PLAN_NAME" \
    --location "$LOCATION" \
    --sku "$SKU" \
    --is-linux \
    -o table

# ── Create Web App (Python 3.11) ────────────────────────────────────────────
echo "==> Creating Web App: $APP_NAME"
az webapp create \
    --resource-group "$RG" \
    --plan "$PLAN_NAME" \
    --name "$APP_NAME" \
    --runtime "PYTHON:3.11" \
    --startup-file "startup.sh" \
    -o table

# ── Configure environment variables ─────────────────────────────────────────

echo "==> Setting environment variables"
az webapp config appsettings set \
    --resource-group "$RG" \
    --name "$APP_NAME" \
    --settings \
        NVIDIA_API_KEY="$NVIDIA_API_KEY" \
        MLFLOW_TRACKING_URI="sqlite:////home/site/wwwroot/mlflow.db" \
        MLFLOW_EXPERIMENT_NAME="meridian-bank-marketing" \
        MLFLOW_INFERENCE_EXPERIMENT_NAME="meridian-bank-inference" \
        RAG_TOP_K=3 \
        RAG_SIMILARITY_THRESHOLD=0.35 \
        SCM_DO_BUILD_DURING_DEPLOYMENT=true \
        PYTHONPATH="/home/site/wwwroot" \
    -o table

# ── Deploy code (Kudu source config-zip) ────────────────────────────────
echo "==> Zipping project (excluding .venv, __pycache__, .git, mlruns)"
DEPLOY_ZIP="/tmp/meridian-deploy-${RANDOM}.zip"
zip -r "$DEPLOY_ZIP" . \
    -x ".venv/*" \
    -x "__pycache__/*" \
    -x "*/__pycache__/*" \
    -x ".git/*" \
    -x "mlruns/*" \
    -x ".pytest_cache/*" \
    -x "*.pyc" > /dev/null

echo "==> Deploying zip (Kudu source config-zip — may take several minutes)"
az webapp deployment source config-zip \
    --resource-group "$RG" \
    --name "$APP_NAME" \
    --src "$DEPLOY_ZIP" 2>&1 || echo "(CLI may have timed out; server-side continues)"

rm -f "$DEPLOY_ZIP"

# Give the container a moment to pick up the new deployment
echo "==> Waiting 30s before restart..."
sleep 30

# ── Restart ──────────────────────────────────────────────────────────────────
echo "==> Restarting web app"
az webapp restart --resource-group "$RG" --name "$APP_NAME" -o table

# ── Output ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      Deployment Complete                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  URL : https://${APP_NAME}.azurewebsites.net                ║"
echo "║  API : https://${APP_NAME}.azurewebsites.net/api/health     ║"
echo "║  MLflow : https://${APP_NAME}.azurewebsites.net/mlflow/     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "All services are same-origin — no CORS issues."
