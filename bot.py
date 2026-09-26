"""
Vera Signal Engine — Main Entry Point
magicpin AI Challenge submission bot.

To run:
    uvicorn bot:app --host 0.0.0.0 --port 8080

Or:
    python bot:app
"""
import sys
from pathlib import Path

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).parent))

from app.api import app
from app.config import config

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Vera Signal Engine on {config.HOST}:{config.PORT}...")
    uvicorn.run("bot:app", host=config.HOST, port=config.PORT, reload=False)
