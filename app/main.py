from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.api.v1 import health, parser, scoring, admin, candidate, recruiter, employer, interview
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME, 
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS is handled by the server (Nginx) proxy
# app.add_middleware(
#     CORSMiddleware,
#     ...
# )

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)},
    )

# --- Include Routers ---
app.include_router(health.router, tags=["Health"])
app.include_router(parser.router, prefix="/v1")
app.include_router(scoring.router, prefix="/v1")
app.include_router(admin.router, prefix="/v1")
app.include_router(candidate.router, prefix="/v1")
app.include_router(recruiter.router, prefix="/v1")
app.include_router(employer.router, prefix="/v1")
app.include_router(interview.router, prefix="/v1")

@app.on_event("startup")
async def startup_event():
    print(f"Starting {settings.PROJECT_NAME}...")

@app.on_event("shutdown")
async def shutdown_event():
    print(f"Shutting down {settings.PROJECT_NAME}...")
