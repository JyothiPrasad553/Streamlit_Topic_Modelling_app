import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection settings (read from .env)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "sentiment_app")
DB_USER = os.getenv("DB_USER", "sentiment_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")
DB_PORT = os.getenv("DB_PORT", "5432")

# ------------------------- DATABASE CONNECTION -------------------------
def get_connection():
    """Establish connection to PostgreSQL database."""
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    return conn

# ------------------------- CREATE TABLES -------------------------
def ensure_users_table():
    """Create users table if it does not exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            full_name TEXT,
            dob TEXT,
            address TEXT,
            pincode TEXT,
            gender TEXT,
            email TEXT UNIQUE,
            password TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def ensure_uploads_table():
    """Create uploads table to store processed sentiment results."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            id SERIAL PRIMARY KEY,
            email TEXT REFERENCES users(email) ON DELETE CASCADE,
            filename TEXT,
            total_records INT,
            positive INT,
            negative INT,
            neutral INT,
            accuracy FLOAT,
            process_time FLOAT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()

# ------------------------- USER OPERATIONS -------------------------
def insert_user(first_name, last_name, full_name, dob, address, pincode, gender, email, hashed_password):
    """Insert new user record into database."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO users (first_name, last_name, full_name, dob, address, pincode, gender, email, password)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (first_name, last_name, full_name, dob, address, pincode, gender, email, hashed_password)
        )
        conn.commit()
        return True, "✅ User created successfully."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"
    finally:
        conn.close()


def get_user(email) -> Optional[Dict]:
    """Retrieve user record by email."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    conn.close()
    return row


def update_user(email, full_name, dob, address, pincode, gender) -> bool:
    """Update user details."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE users
            SET full_name = %s, dob = %s, address = %s, pincode = %s, gender = %s
            WHERE email = %s
            """,
            (full_name, dob, address, pincode, gender, email)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

# ------------------------- UPLOAD RECORDS -------------------------
def insert_upload_record(email, filename, total_records, positive, negative, neutral, accuracy, process_time):
    """Insert analysis summary for uploaded CSV/text data."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO uploads (email, filename, total_records, positive, negative, neutral, accuracy, process_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (email, filename, total_records, positive, negative, neutral, accuracy, process_time)
    )
    conn.commit()
    conn.close()

def get_user_uploads(email):
    """Fetch all sentiment analysis uploads for a user."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM uploads WHERE email = %s ORDER BY uploaded_at DESC", (email,))
    rows = cur.fetchall()
    conn.close()
    return rows
