"""
Configuration module — reads from environment or .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv(Path(__file__).parent.parent / ".env", override=False)


class Config:
    # LLM Provider: "openai" | "anthropic" | "gemini" | "deepseek" | "groq" | "openrouter" | "none"
    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai")

    # API keys
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")

    # Model overrides (optional)
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "")

    # Internal timeouts
    LLM_TIMEOUT_SECONDS: int = int(os.environ.get("LLM_TIMEOUT_SECONDS", "20"))
    TICK_TIMEOUT_SECONDS: int = int(os.environ.get("TICK_TIMEOUT_SECONDS", "25"))

    # Dataset path
    DATASET_DIR: Path = Path(os.environ.get("DATASET_DIR", "")) or (
        Path(__file__).parent.parent / "dataset" / "expanded"
    )

    # Fallback: if dataset/expanded not found, try dataset/
    @classmethod
    def dataset_dir(cls) -> Path:
        expanded = Path(__file__).parent.parent / "dataset" / "expanded"
        if expanded.exists():
            return expanded
        seed = Path(__file__).parent.parent / "dataset"
        return seed

    # Bot metadata
    TEAM_NAME: str = os.environ.get("TEAM_NAME", "Vera Signal Engine")
    TEAM_MEMBERS: str = os.environ.get("TEAM_MEMBERS", "Arpit")
    CONTACT_EMAIL: str = os.environ.get("CONTACT_EMAIL", "arpitks595@gmail.com")
    VERSION: str = "1.0.0"

    # Server
    PORT: int = int(os.environ.get("PORT", "8080"))
    HOST: str = os.environ.get("HOST", "0.0.0.0")

    # Max actions per tick
    MAX_ACTIONS_PER_TICK: int = 20


config = Config()
