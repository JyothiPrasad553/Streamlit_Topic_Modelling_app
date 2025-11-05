# init_db.py
from database import ensure_users_table, ensure_uploads_table

ensure_users_table()
ensure_uploads_table()

print("✅ PostgreSQL tables created successfully!")