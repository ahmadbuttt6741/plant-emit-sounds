"""
Database and Authentication Manager for PhytoEcho AI.
Uses SQLite for persistent user storage and Werkzeug for cryptographic password hashing.
"""

import os
import shutil
import sqlite3
import secrets
import tempfile
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

# Handle Vercel / AWS Lambda read-only filesystem by storing DB in temporary writable dir
if os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
    DB_PATH = os.path.join(tempfile.gettempdir(), 'phytoecho_users.db')
    source_db = os.path.join(os.path.dirname(__file__), 'users.db')
    if not os.path.exists(DB_PATH) and os.path.exists(source_db):
        try:
            shutil.copyfile(source_db, DB_PATH)
        except Exception:
            pass
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')


def get_db_connection():
    # If on Vercel/serverless and DB doesn't exist yet in temp dir, copy initial seeded DB
    if (os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME')) and not os.path.exists(DB_PATH):
        source_db = os.path.join(os.path.dirname(__file__), 'users.db')
        if os.path.exists(source_db):
            try:
                shutil.copyfile(source_db, DB_PATH)
            except Exception:
                pass
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite database and seeds default demonstration accounts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT DEFAULT 'Botanical Researcher',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            reset_token TEXT,
            reset_token_expiry TIMESTAMP
        )
    """)
    
    # Ensure 'botany' primary user/admin account exists with password 'nanstack'
    nanstack_hash = generate_password_hash('nanstack')
    
    cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", ('botany', 'botany@nanstack.tech'))
    botany_user = cursor.fetchone()
    if not botany_user:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'botany',
            'botany@nanstack.tech',
            nanstack_hash,
            'Botany Administrator',
            'Lead Bio-Acoustic Scientist'
        ))
    else:
        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (nanstack_hash, botany_user['id']))

    # Also update/seed 'admin' user with password 'nanstack'
    cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", ('admin@nanstack.tech', 'admin'))
    admin_user = cursor.fetchone()
    if not admin_user:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'admin',
            'admin@nanstack.tech',
            nanstack_hash,
            'Nanstack Administrator',
            'Lead Bio-Acoustic Engineer'
        ))
    else:
        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (nanstack_hash, admin_user['id']))
        
    # Also update/seed 'researcher' user with password 'nanstack'
    cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", ('researcher@nanstack.tech', 'researcher'))
    researcher_user = cursor.fetchone()
    if not researcher_user:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'researcher',
            'researcher@nanstack.tech',
            nanstack_hash,
            'Dr. Elena Vance',
            'Plant Physiologist & Acoustic Researcher'
        ))
    else:
        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (nanstack_hash, researcher_user['id']))

    conn.commit()
    conn.close()


def get_user_by_identifier(identifier):
    """Finds user by email or username (case-insensitive)."""
    if not identifier:
        return None
    ident = identifier.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM users 
        WHERE LOWER(email) = ? OR LOWER(username) = ?
    """, (ident, ident))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def get_user_by_id(user_id):
    """Retrieves user by primary key id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def verify_user(identifier, password):
    """Verifies credentials. Returns user dict on success, None on failure."""
    user = get_user_by_identifier(identifier)
    if not user:
        return None
    if check_password_hash(user['password_hash'], password):
        # Update last_login
        conn = get_db_connection()
        conn.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user['id'],))
        conn.commit()
        conn.close()
        return user
    return None


def generate_reset_token(email):
    """Generates a secure 6-digit verification code or token for password reset."""
    user = get_user_by_identifier(email)
    if not user:
        return None, "No account registered with this email address."
        
    # Generate 6-digit code for clean demo usability
    reset_code = f"{secrets.randbelow(900000) + 100000}"
    expiry = datetime.utcnow() + timedelta(minutes=30)
    
    conn = get_db_connection()
    conn.execute("""
        UPDATE users 
        SET reset_token = ?, reset_token_expiry = ? 
        WHERE id = ?
    """, (reset_code, expiry.strftime('%Y-%m-%d %H:%M:%S'), user['id']))
    conn.commit()
    conn.close()
    
    return reset_code, None


def reset_password_with_token(email, token, new_password):
    """Validates reset token and sets new password."""
    user = get_user_by_identifier(email)
    if not user:
        return False, "User not found."
        
    if not user.get('reset_token') or user.get('reset_token') != token.strip():
        return False, "Invalid reset code. Please check and try again."
        
    # Check expiry
    expiry_str = user.get('reset_token_expiry')
    if expiry_str:
        try:
            expiry_dt = datetime.strptime(expiry_str, '%Y-%m-%d %H:%M:%S')
            if datetime.utcnow() > expiry_dt:
                return False, "Reset code has expired. Please request a new one."
        except Exception:
            pass

    if len(new_password) < 6:
        return False, "Password must be at least 6 characters long."
        
    new_hash = generate_password_hash(new_password)
    conn = get_db_connection()
    conn.execute("""
        UPDATE users 
        SET password_hash = ?, reset_token = NULL, reset_token_expiry = NULL 
        WHERE id = ?
    """, (new_hash, user['id']))
    conn.commit()
    conn.close()
    return True, "Password has been successfully updated. You can now sign in."


# Auto-initialize database on import
init_db()
