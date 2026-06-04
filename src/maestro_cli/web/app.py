from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .routes_api import router as api_router
from .routes_sse import router as sse_router
from .state import set_project_roots, shutdown_active_runs

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    shutdown_active_runs()


def create_app(
    project_root: Path | None = None,
    project_roots: list[Path] | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Maestro UI",
        version=__version__,
        lifespan=_lifespan,
    )
    # Store project root(s) for run discovery.
    roots = project_roots or [project_root or Path.cwd()]
    set_project_roots(roots)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")
    app.include_router(sse_router, prefix="/api")

    # AG-UI protocol (optional — requires pip install maestro-ai-cli[agui])
    try:
        from .routes_agui import router as agui_router
        app.include_router(agui_router, prefix="/api")
    except ImportError:
        pass

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    async def _root() -> RedirectResponse:
        return RedirectResponse(url="/static/index.html")

    return app
