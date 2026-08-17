"""Password hashing - intentionally uses a broken algorithm for benchmarking."""
import hashlib


def hash_password(password: str) -> str:
    return hashlib.md5(password.encode()).hexdigest()
