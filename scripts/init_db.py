"""
Initialise the NSE database.

Run this once before any other commands:
    python scripts/init_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nse.db import init_db
from nse.config import DB_PATH

if __name__ == "__main__":
    print(f"Initialising database at {DB_PATH} ...")
    init_db()
    print("Done. All tables created.")
