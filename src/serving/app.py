"""Consolidated entrypoint: FastAPI API + static UI + MLflow UI, single origin."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount

from src.serving.serve import app as fastapi_app

app = FastAPI(title="Meridian Customer Intelligence (Unified)")

app.mount("/api", fastapi_app)

# Lazy-load MLflow on first access to /mlflow to speed up container startup
class LazyMLflowMount:
    def __init__(self):
        self._app = None

    def __call__(self, scope, receive, send):
        if self._app is None:
            from mlflow.server import app as mlflow_wsgi_app
            self._app = WSGIMiddleware(mlflow_wsgi_app)
        return self._app(scope, receive, send)

app.mount("/mlflow", LazyMLflowMount())

HERE = Path(__file__).resolve().parent.parent.parent
ui_dir = HERE / "ui"
ui_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
