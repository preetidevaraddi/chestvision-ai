from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine
from app.core.config import settings
from app.routers import auth, admin, doctor

Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.APP_NAME, description="AI-Assisted Multi-Disease Chest X-Ray Analysis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/files/reports", StaticFiles(directory=settings.REPORT_DIR), name="reports")

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(doctor.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": settings.APP_NAME}
