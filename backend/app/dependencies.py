from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.security import decode_access_token
from app.models import Admin, Doctor

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/admin/login")

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_claims(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_access_token(token)
    if payload is None or "sub" not in payload or "role" not in payload:
        raise credentials_exception
    return payload


def require_admin(claims: dict = Depends(get_current_claims), db: Session = Depends(get_db)) -> Admin:
    """
    Enforces ADMIN role at the API layer - not just hidden buttons in the UI.
    A Doctor token is rejected here even if they hit the URL directly.
    """
    if claims.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    admin = db.query(Admin).filter(Admin.admin_id == claims["sub"]).first()
    if not admin:
        raise credentials_exception
    return admin


def require_doctor(claims: dict = Depends(get_current_claims), db: Session = Depends(get_db)) -> Doctor:
    """
    Enforces DOCTOR role at the API layer. Doctor routes are all read-only
    by construction - no write endpoints exist under /api/doctor/*.
    """
    if claims.get("role") != "doctor":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor access required")
    doctor = db.query(Doctor).filter(Doctor.doctor_id == claims["sub"]).first()
    if not doctor:
        raise credentials_exception
    return doctor
