import logging
import psycopg2
from src.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT # Adjusted import

# Configure logging for this module
logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.

    Returns:
        psycopg2.connection: Database connection object or None if connection fails
    """
    try:
        connection = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        logger.info("Successfully connected to the database")
        return connection
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return None