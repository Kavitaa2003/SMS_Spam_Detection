import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.database import Base, engine
from app import models  # noqa: F401 — needed so Base.metadata knows about the tables
from app.routers import auth_routes, predict_routes

# Creates spam_detector.db and all tables on first run if they don't exist yet.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Spam SMS Detector API",
    description="Ensemble ML spam classifier with JWT-authenticated history tracking.",
    version="1.0.0",
)

app.include_router(auth_routes.router)
app.include_router(predict_routes.router)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Serves index.html, login.html, style.css, main.js etc. directly.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/login")
def serve_login():
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))


@app.get("/register")
def serve_register():
    return FileResponse(os.path.join(STATIC_DIR, "register.html"))


@app.get("/history")
def serve_history():
    return FileResponse(os.path.join(STATIC_DIR, "history.html"))
