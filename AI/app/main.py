"""API application entry point"""

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Insurance Policy AI Assistant",
    version="1.0"
)

# Register API routes
app.include_router(
    router,
    prefix="/api"
)


@app.get("/")
def health_check():

    return {
        "status": "running",
        "service": "Policy AI Assistant"
    }
