\
import psycopg2
import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def initialize_database():
    """Connects to the PostgreSQL database and executes the schema script."""
    conn = None
    cur = None
    try:
        print(f"Connecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT} as user '{DB_USER}'...")
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True # Use autocommit for DDL statements
        cur = conn.cursor()
        print("Connection successful.")

        schema_file_path = os.path.join(project_root, 'sql', 'pricepirate_schema.sql')
        print(f"Reading schema from {schema_file_path}...")

        if not os.path.exists(schema_file_path):
            print(f"Error: Schema file not found at {schema_file_path}", file=sys.stderr)
            return

        with open(schema_file_path, 'r') as f:
            sql_script = f.read()

        print("Executing schema script...")
        # Execute the script. Handle potential errors if needed.
        # Note: Splitting by semicolon might be too naive if the SQL contains semicolons within strings or comments.
        # A more robust approach might be needed for complex scripts, but this works for many basic DDL dumps.
        # For pg_dump output, it's generally safe as statements are separated by semicolons at the end of lines.
        cur.execute(sql_script)

        print("Database schema initialized successfully.")

    except psycopg2.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Schema file not found at {schema_file_path}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        print(f"Loading environment variables from {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print("No .env file found, relying on environment variables set externally.")

    initialize_database()
