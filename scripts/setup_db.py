import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import get_engine
from sqlalchemy  import text


def setup_database():
    engine = get_engine()
    with open('sql/create_tables.sql', 'r') as f :
        sql_commands = f.read()
    try:
        with engine.connect() as conn:
            conn.execute(text(sql_commands))
            conn.commit()
            print("✅ Database tables created successfully!")    
    except Exception as e :
        print(f"❌ Error creating tables: {e}")
if __name__ =="__main__":
    setup_database()        