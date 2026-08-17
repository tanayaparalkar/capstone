"""User lookup - intentionally vulnerable to SQL injection for benchmarking."""
import sqlite3


def get_user(conn: sqlite3.Connection, username: str):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    return conn.execute(query).fetchone()
