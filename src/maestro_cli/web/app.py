from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .routes_api import router as api_router
from .routes_sse import router as sse_router
from .state import set_project_roots, shutdown_active_runs

_STATIC_DIR = Path(__file__).parent / "static"

# Same-machine origins (any port) — the bundled dashboard is served same-origin,
# this only matters for a separate local frontend (e.g. an AG-UI client on
# localhost:3000). A malicious public website you happen to visit has a
# different origin and will NOT match, so it cannot drive the dashboard via
# your browser.
_LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


def _cors_kwargs() -> dict[str, Any]:
    """Build CORS settings for the dashboard (local-first secure default).

    The default confines cross-origin access to same-machine origins. Set
    ``MAESTRO_UI_ALLOW_ORIGINS`` (comma-separated origins, or ``*`` to restore
    the permissive wildcard) when intentionally exposing the dashboard behind a
    trusted proxy or to a known external frontend.
    """
    explicit = os.environ.get("MAESTRO_UI_ALLOW_ORIGINS", "").strip()
    if explicit:
        origins = [o.strip() for o in explicit.split(",") if o.strip()]
        if "*" in origins:
            return {"allow_origins": ["*"]}
        if origins:
            return {"allow_origins": origins}
    return {"allow_origin_regex": _LOCAL_ORIGIN_REGEX}


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
        allow_methods=["*"],
        allow_headers=["*"],
        **_cors_kwargs(),
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
