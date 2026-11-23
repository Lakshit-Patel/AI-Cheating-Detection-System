from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .auth import authenticate_user, create_access_token, create_refresh_token, revoke_refresh_token, rotate_refresh_token
from .config import ACCESS_TOKEN_EXPIRE_MINUTES
from .database import get_db
from .schemas import RefreshRequest, LogoutRequest, TokenResponse, UserRegister
from .rate_limit import limiter

router = APIRouter(tags=["auth"])


def _tokens_for_user(db: Session, user, request: Request):
    access = create_access_token(user)
    refresh = create_refresh_token(
        db,
        user,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
def register(request: Request, user: UserRegister, db: Session = Depends(get_db)):
    email = user.email.strip().lower()
    from .models import User
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="Email already exists")
    from .auth import hash_password
    account = User(name=user.name.strip(), email=email, role="student", password_hash=hash_password(user.password), is_active=1)
    db.add(account)
    db.commit()
    db.refresh(account)
    return {"message": "User registered successfully", "user_id": account.id}


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password", headers={"WWW-Authenticate": "Bearer"})
    return _tokens_for_user(db, user, request)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("20/minute")
def refresh(request: Request, body: RefreshRequest, db: Session = Depends(get_db)):
    result = rotate_refresh_token(
        db,
        body.refresh_token,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked refresh token")
    user, new_refresh = result
    return {
        "access_token": create_access_token(user),
        "refresh_token": new_refresh,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@router.post("/logout")
@limiter.limit("20/minute")
def logout(request: Request, body: LogoutRequest, db: Session = Depends(get_db)):
    if not revoke_refresh_token(db, body.refresh_token):
        return {"message": "Logged out"}
    return {"message": "Logged out"}
