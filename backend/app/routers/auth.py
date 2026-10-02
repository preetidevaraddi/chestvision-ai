from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Admin, Doctor
from app.schemas import AdminRegister, DoctorRegister, LoginRequest, TokenResponse
from app.core.security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ---------- Admin ----------

@router.post("/admin/register", status_code=status.HTTP_201_CREATED)
def register_admin(payload: AdminRegister, db: Session = Depends(get_db)):
    if db.query(Admin).filter(Admin.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    admin = Admin(name=payload.name, email=payload.email, password_hash=hash_password(payload.password))
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return {"message": "Admin registered successfully", "admin_id": admin.admin_id}


@router.post("/admin/login", response_model=TokenResponse)
def login_admin(payload: LoginRequest, db: Session = Depends(get_db)):
    admin = db.query(Admin).filter(Admin.email == payload.email).first()
    if not admin:
        raise HTTPException(status_code=401, detail="No account found with that email address.")
    if not verify_password(payload.password, admin.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password. Please try again.")
    token = create_access_token({"sub": admin.admin_id, "role": "admin"})
    return TokenResponse(access_token=token, role="admin", name=admin.name, email=admin.email)


# ---------- Doctor ----------

@router.post("/doctor/register", status_code=status.HTTP_201_CREATED)
def register_doctor(payload: DoctorRegister, db: Session = Depends(get_db)):
    if db.query(Doctor).filter(Doctor.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    doctor = Doctor(name=payload.name, email=payload.email, password_hash=hash_password(payload.password))
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return {"message": "Doctor registered successfully", "doctor_id": doctor.doctor_id}


@router.post("/doctor/login", response_model=TokenResponse)
def login_doctor(payload: LoginRequest, db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.email == payload.email).first()
    if not doctor:
        raise HTTPException(status_code=401, detail="No account found with that email address.")
    if not verify_password(payload.password, doctor.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password. Please try again.")
    token = create_access_token({"sub": doctor.doctor_id, "role": "doctor"})
    return TokenResponse(access_token=token, role="doctor", name=doctor.name, email=doctor.email)
