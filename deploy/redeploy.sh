#!/usr/bin/env bash
set -eo pipefail

DEPLOY_TARGET=${1:-all}

if [[ ! "$DEPLOY_TARGET" =~ ^(all|fastapi|mlflow|nginx)$ ]]; then
    echo "Usage: $0 [all|fastapi|mlflow|nginx]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "==> [Redeploy] Starting redeployment of updated container images"

# ── Build & Push Images ──────────────────────────────────────────────────────
export BUILD_TARGET="$DEPLOY_TARGET"
source "${SCRIPT_DIR}/build.sh"

# ── Resolve ACA environment FQDN ────────────────────────────────────────────
ACA_ENV_FQDN=$(az containerapp env show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ENV_NAME" \
    --query "properties.defaultDomain" -o tsv | tr -d '\r')
echo "==> ACA env FQDN suffix: $ACA_ENV_FQDN"

# ── Update Nginx UI Container App ───────────────────────────────────────────
if [[ "$DEPLOY_TARGET" == "all" || "$DEPLOY_TARGET" == "nginx" ]]; then
    echo "==> Updating Nginx UI container app with new image"
    az containerapp update \
        --resource-group "$RESOURCE_GROUP" \
        --name "$NGINX_APP" \
        --image "${ACR_SERVER}/nginx-ui:${IMAGE_TAG}" \
        --set-env-vars \
            FASTAPI_HOST="${FASTAPI_APP}.internal.${ACA_ENV_FQDN}" \
            MLFLOW_HOST="${MLFLOW_APP}.internal.${ACA_ENV_FQDN}" \
        -o table
fi

# ── Update MLflow UI Container App ───────────────────────────────────────────
if [[ "$DEPLOY_TARGET" == "all" || "$DEPLOY_TARGET" == "mlflow" ]]; then
    echo "==> Updating MLflow UI container app with new image"
    az containerapp update \
        --resource-group "$RESOURCE_GROUP" \
        --name "$MLFLOW_APP" \
        --image "${ACR_SERVER}/mlflow:${IMAGE_TAG}" \
        --allow-insecure true \
        --set-env-vars \
            MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="*" \
        -o table
fi

# ── Update FastAPI App Container App ─────────────────────────────────────────
if [[ "$DEPLOY_TARGET" == "all" || "$DEPLOY_TARGET" == "fastapi" ]]; then
    echo "==> Updating FastAPI app container app with new image"
    az containerapp update \
        --resource-group "$RESOURCE_GROUP" \
        --name "$FASTAPI_APP" \
        --image "${ACR_SERVER}/fastapi:${IMAGE_TAG}" \
        --allow-insecure true \
        -o table
fi

# ── Verify Health ──────────────────────────────────────────────────────────
echo "==> Waiting 20s for new revisions to stabilize..."
sleep 20

# Get the nginx public FQDN
UI_FQDN=$(az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$NGINX_APP" \
    --query properties.configuration.ingress.fqdn \
    -o tsv | tr -d '\r')

echo ""
echo "==> Testing /api/health endpoint..."
HTTP_CODE=$(curl -s -o /tmp/health_response.json -w "%{http_code}" "https://${UI_FQDN}/api/health" 2>/dev/null || echo "000")
echo "    HTTP Status: $HTTP_CODE"
if [ "$HTTP_CODE" = "200" ]; then
    echo "    Response:"
    cat /tmp/health_response.json | head -5
    echo ""
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     Redeployment Complete                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  UI + Gateway : https://${UI_FQDN}                          ║"
echo "║  API Health   : https://${UI_FQDN}/api/health               ║"
echo "║  MLflow UI    : https://${UI_FQDN}/mlflow/                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
