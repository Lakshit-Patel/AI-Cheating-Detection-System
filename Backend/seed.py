import argparse
from app.auth import hash_password, normalize_email
from app.database import SessionLocal
from app.models import User

def seed_admin(email: str, password: str, name: str) -> None:
    db = SessionLocal()
    try:
        email = normalize_email(email)
        user = db.query(User).filter(User.email == email).first()
        if user:
            if user.role != "admin":
                raise RuntimeError(f"User {email} already exists and is not an admin")
            print(f"Admin already exists: {email}")
            return
        db.add(User(name=name.strip(), email=email, role="admin", password_hash=hash_password(password), is_active=1))
        db.commit()
        print(f"Admin created: {email}")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--name", default="Administrator")
    a = p.parse_args()
    seed_admin(a.email, a.password, a.name)
