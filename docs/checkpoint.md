# Deployment Checkpoint - 2026-05-26

## Infrastructure Overview

| Service | Azure Resource | Status |
|---|---|---|
| FastAPI app | `fastapi-app` (Container App) | ✅ Healthy (Revision `fastapi-app--0000012`, `allowInsecure: true`) |
| MLflow UI | `mlflow-ui` (Container App) | ✅ Healthy (Revision `mlflow-ui--0000021`, `allowInsecure: true`) |
| Nginx reverse proxy | `nginx-ui` (Container App) | ✅ Healthy (Revision `nginx-ui--0000026`) |
| Container Registry | `meridianciacr` (ACR) | ✅ Active |
| ACA Environment | `meridian-ci-env` | ✅ Active |
| Resource Group | `meridian-customer-intelligence` | ✅ Central India |

### Public URLs
```
https://nginx-ui.blackmushroom-f84087ba.centralindia.azurecontainerapps.io
```
- `/`         → React UI (static, served by nginx) ✅ 200 OK
- `/api/health` → FastAPI health check endpoint (proxied) ✅ 200 OK (returns valid JSON)
- `/mlflow/`  → MLflow tracking UI (proxied) ✅ 200 OK

---

## Major Root-Cause Discovery & Fixes

### 1. Host Header Propagation (Nginx to ACA Ingress)
- **Problem**: Azure's internal load balancer requires the `Host` header to match the internal FQDN of the destination container app (`fastapi-app.internal...`). By default, Nginx passed the client's host header, causing a 404.
- **Fix**: Added `proxy_set_header Host ${FASTAPI_HOST};` in `nginx.conf.template` so the internal Envoy router successfully routes requests.

### 2. Nginx Rewrite Bug
- **Problem**: The frontend expects `/api/health`, but Nginx had `rewrite ^/api/(.*) /$1 break;`. It was stripping the `/api/` prefix. The `app.py` mounts the FastAPI application at `/api/` (`app.mount("/api", fastapi_app)`). Thus, the backend received `/health` instead of `/api/health` and returned a 404.
- **Fix**: Removed the `rewrite` directive for the `/api/` block.

### 3. MLflow Azure Quickstart Image Bug
- **Problem**: The latest `mlflow-ui` revision was accidentally deployed using the default Azure Quickstart image (`mcr.microsoft.com/k8se/quickstart:latest`) listening on port 80 instead of port 5000, causing health probes to fail.
- **Fix**: Built the correct MLflow image from `docker/Dockerfile.mlflow`, pushed it, and updated the Container App via `deploy/redeploy-fix.sh` to route traffic to the healthy MLflow server.

### 4. Azure CA Internal DNS and Envoy SNI 404 Bug
- **Problem**: When proxying to the internal `fastapi-app` and `mlflow-ui` FQDNs, Azure returned a hard `404 - This Container App is stopped or does not exist`. This occurred because Nginx was configured with a hardcoded Azure public DNS resolver (`resolver 168.63.129.16;`). This resolver bypassed the internal OS CoreDNS, resolving `.internal.` FQDNs to the External Load Balancer instead of the Internal Load Balancer.
- **Fix**: Removed the `resolver` directive completely. This forces Nginx to resolve the IPs once at startup using the native OS resolver (`/etc/resolv.conf`), which correctly maps the `.internal.` domains to the Azure Internal Load Balancer IP. We also ensured `proxy_http_version 1.1;` is used for Envoy compatibility.

### 5. FastAPI ML Model Cold-Start 504 Timeout
- **Problem**: After fixing the DNS routing, the `/predict` endpoint returned a `504 Gateway Time-out`. This happened because the FastAPI container was cold-starting and taking longer than Nginx's default 60-second read timeout to load the Machine Learning model into memory.
- **Fix**: Added `proxy_read_timeout 300s;` and `proxy_send_timeout 300s;` to the Nginx configuration to allow the backend ample time to process cold-start requests.

### 6. Redirect Loops & Internal HTTPS Redirection
- **Problem**: Internal container app ingresses in Azure Container Apps automatically redirect standard HTTP traffic on port 80 to HTTPS by default. Because Nginx proxied internally over `http://`, the Azure internal router returned a `301 Moved Permanently` pointing to the internal domain (`https://fastapi-app.internal...`). This leaked the internal FQDN to the client's browser and caused 404 errors.
- **Fix**:
  1. Updated the internal ingresses of `fastapi-app` and `mlflow-ui` to allow insecure connections (`--allow-insecure true` / `allowInsecure: true`) so they accept port 80 HTTP traffic inside the virtual cluster network without redirecting.
  2. Configured `absolute_redirect off;` in the Nginx `server` block and kept `proxy_redirect` directives to rewrite any relative backend redirections (like Starlette slash corrections) back to relative paths, which safely resolve against the public HTTPS gateway.

---

## Current Status (End-to-End Success)
- ✅ `nginx-ui` successfully routes `/api/` traffic to `fastapi-app` internally via standard HTTP.
- ✅ `nginx-ui` successfully routes `/mlflow/` traffic to `mlflow-ui` internally.
- ✅ End-to-end health checks return a perfect `200 OK` with JSON payloads.
- ✅ The UI is fully functional for all real-time model predictions and LangGraph aggregate complaint insights.
