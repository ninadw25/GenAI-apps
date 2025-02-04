import cassio
from typing import Tuple, Optional

def init_astradb_connection(token: str, database_id: str) -> bool:
    """Initialize connection to AstraDB."""
    try:
        cassio.init(token=token, database_id=database_id)
        return True
    except Exception as e:
        print(f"Error connecting to AstraDB: {e}")
        return False