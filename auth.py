# auth.py
import bcrypt
from database import insert_user, get_user, get_connection

def create_user(first_name, last_name, full_name, dob,
                address, pincode, gender, email, password):
    if not all([first_name, last_name, full_name, dob, address, pincode, gender, email, password]):
        return False, "All fields are required."
    if len(pincode) != 6 or not pincode.isdigit():
        return False, "Pincode must be exactly 6 digits."
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    ok, msg = insert_user(first_name, last_name, full_name, dob, address, pincode, gender, email, hashed)
    return ok, msg

def authenticate_user(email, password):
    user = get_user(email)
    if not user:
        return False, "User not found."
    hashed = user["password"]
    try:
        valid = bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False, "Invalid password format."
    if valid:
        return True, user["email"]
    return False, "Incorrect password."

def update_password(email, new_password):
    user = get_user(email)
    if not user:
        return False
    hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE users SET password = ? WHERE email = ?", (hashed, email))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()
