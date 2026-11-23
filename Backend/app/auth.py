import bcrypt
import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from jose import JWTError, jwt
from sqlalchemy.orm import Session

from .config import ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS, SECRET_KEY
from .models import RefreshToken, User

ALGORITHM = "HS256"
ACCESS_TOKEN_TYPE = "access"
MAX_BCRYPT_BYTES = 72


def _password_bytes(password: str) -> bytes:
    raw = password.encode("utf-8")
    if len(raw) > MAX_BCRYPT_BYTES:
        raise ValueError("Password is too long for bcrypt; use 72 UTF-8 bytes or fewer")
    return raw


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(_password_bytes(password), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def normalize_email(email: str) -> str:
    return email.strip().lower()


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = db.query(User).filter(User.email == normalize_email(email)).first()
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def create_access_token(user: User) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user.id),
        "role": user.role,
        "type": ACCESS_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "jti": str(uuid4()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != ACCESS_TOKEN_TYPE or not payload.get("sub"):
            return None
        return payload
    except JWTError:
        return None


def _hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_refresh_token(
    db: Session,
    user: User,
    family_id: str | None = None,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> str:
    raw = secrets.token_urlsafe(48)
    token = RefreshToken(
        user_id=user.id,
        token_hash=_hash_refresh_token(raw),
        family_id=family_id or str(uuid4()),
        expires_at=datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        user_agent=user_agent,
        ip_address=ip_address,
    )
    db.add(token)
    db.commit()
    return raw


def ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def rotate_refresh_token(
    db: Session,
    raw_token: str,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> tuple[User, str] | None:
    token_hash = _hash_refresh_token(raw_token)
    stored = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    now = datetime.now(UTC)
    if stored is None:
        return None

    if stored.revoked_at is not None:
        active_family_tokens = db.query(RefreshToken).filter(
            RefreshToken.family_id == stored.family_id,
            RefreshToken.revoked_at.is_(None),
        ).all()
        for family_token in active_family_tokens:
            family_token.revoked_at = now
        db.commit()
        return None

    expires_at = ensure_utc(stored.expires_at)
    if expires_at is not None and expires_at <= now:
        stored.revoked_at = now
        db.commit()
        return None

    user = db.query(User).filter(User.id == stored.user_id, User.is_active == 1).first()
    if user is None:
        stored.revoked_at = now
        db.commit()
        return None

    stored.revoked_at = now
    new_raw = secrets.token_urlsafe(48)
    new_stored = RefreshToken(
        user_id=user.id,
        token_hash=_hash_refresh_token(new_raw),
        family_id=stored.family_id,
        expires_at=now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        user_agent=user_agent,
        ip_address=ip_address,
    )
    stored.replaced_by_hash = new_stored.token_hash
    db.add(new_stored)
    db.commit()
    return user, new_raw


def revoke_refresh_token(db: Session, raw_token: str) -> bool:
    stored = db.query(RefreshToken).filter(
        RefreshToken.token_hash == _hash_refresh_token(raw_token)
    ).first()
    if stored is None or stored.revoked_at is not None:
        return False
    stored.revoked_at = datetime.now(UTC)
    db.commit()
    return True
