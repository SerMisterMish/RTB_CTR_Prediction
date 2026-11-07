"""
Main FastAPI application entry point.
Configures lifespan management, middlewares, exception handling, and API routers.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as api_v1_router
from src.config import settings
from src.services.model_runner import get_model_runner

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler: loads model artifacts once upon application startup
    and gracefully cleans up on shutdown.
    """
    logger.info("Initializing application and loading ML artifacts...")
    runner = get_model_runner()
    try:
        runner.load_artifacts()
        logger.info("Application startup complete. Ready to serve traffic.")
    except Exception as exc:
        logger.error(f"Failed to load artifacts on startup: {exc}", exc_info=True)
        # Continue startup so /health can report not ready
    yield
    logger.info("Shutting down CTR Inference Service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-Time Bidding CTR Inference Microservice powered by FastAPI and XGBoost.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable) -> Response:
    """
    Measures request round-trip processing time and sets X-Process-Time-Ms header.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0

    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

    if request.url.path not in ("/health",):
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} "
            f"({process_time_ms:.2f} ms)"
        )
    return response


# Mount API routers
# 1. Versioned API under /api/v1
app.include_router(api_v1_router, prefix="/api/v1")

# 2. Root-level endpoints for Docker HEALTHCHECK and monitoring tools
app.include_router(api_v1_router, tags=["Root"])


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,
    )
