import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import get_engine
from sqlalchemy import text

def setup_database():
    """Setup database tables from SQL file"""
    try:
        print("🔄 Connecting to database...")
        engine = get_engine()
        
        # Use relative path
        sql_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sql', 'create_tables.sql')

        print(f"📁 Reading SQL file: {sql_file_path}")
        
        with open(sql_file_path, 'r') as f:
            sql_commands = f.read()
        
        print("⚙️ Executing SQL commands...")
        with engine.connect() as conn:
            conn.execute(text(sql_commands))
            conn.commit()
            
        print("✅ Database tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

if __name__ == "__main__":
    setup_database()