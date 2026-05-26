#!/usr/bin/env bash

# ── Load Environment Variables ───────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    echo "==> Loading environment variables from .env"
    source "${REPO_ROOT}/.env"
fi

# ── Configuration Defaults ───────────────────────────────────────────────────
export RESOURCE_GROUP="${RESOURCE_GROUP:-meridian-customer-intelligence}"
export LOCATION="${AZURE_LOCATION:-centralindia}"
export ACR_NAME="${ACR_NAME:-meridianciacr}"
export ENV_NAME="${ACA_ENV:-meridian-ci-env}"
export IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"

export FASTAPI_APP="fastapi-app"
export MLFLOW_APP="mlflow-ui"
export NGINX_APP="nginx-ui"

# ── Resolve ACR Server Address ───────────────────────────────────────────────
if command -v az &> /dev/null; then
    if ACR_SERVER_VAL=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv 2>/dev/null | tr -d '\r'); then
        export ACR_SERVER="$ACR_SERVER_VAL"
    else
        export ACR_SERVER="${ACR_NAME}.azurecr.io"
    fi
else
    export ACR_SERVER="${ACR_NAME}.azurecr.io"
fi
