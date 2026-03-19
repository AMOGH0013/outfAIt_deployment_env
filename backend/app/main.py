from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.database import engine, Base, SessionLocal, ensure_sqlite_schema
from app.models import user, scan, wardrobe, outfit, feedback, assistant
from app.models import body_profile
from app.dependencies import get_password_hash, ensure_password_hashed
from app.api import outfits
from app.api import scan as scan_api
from app.api import wardrobe as wardrobe_api
from app.api import feedback as feedback_api
from app.api import body_profile as body_profile_api
from app.api import style_profile as style_profile_api
from app.api import auth as auth_api



app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")


@app.get("/")
def serve_index():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))

@app.get("/login")
def serve_login():
    return FileResponse(str(_FRONTEND_DIR / "login.html"))

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_schema()
    db = SessionLocal()
    try:
        existing = db.query(user.User).first()
        if existing and ensure_password_hashed(existing, raw_password_fallback="dev"):
            db.add(existing)

        dev_user = (
            db.query(user.User)
            .filter(user.User.email == "dev@example.com")
            .first()
        )
        if not dev_user:
            dev_user = user.User(
                email="dev@example.com",
                password_hash=get_password_hash("dev"),
            )
            db.add(dev_user)
        elif ensure_password_hashed(dev_user, raw_password_fallback="dev"):
            db.add(dev_user)

        db.commit()
    finally:
        db.close()

    # Pre-load CLIP backend so the first scan request doesn't pay cold-start cost.
    try:
        from app.services.clip_service import preload_clip_models

        loaded, message = preload_clip_models()
        if loaded:
            print(f"CLIP model preloaded: {message}")
        else:
            print(f"CLIP preload skipped: {message}")
    except Exception as e:
        print(f"CLIP preload skipped: {e}")

app.include_router(outfits.router)
app.include_router(scan_api.router)
app.include_router(wardrobe_api.router)
app.include_router(feedback_api.router)
app.include_router(body_profile_api.router)
app.include_router(style_profile_api.router)
app.include_router(auth_api.router)
