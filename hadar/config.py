import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import time
from typing import List

load_dotenv()

GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
]
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]

_current_groq_key_index = 0
_groq_clients = [Groq(api_key=key) for key in GROQ_API_KEYS]

def get_groq_client() -> Groq:
    global _current_groq_key_index
    if not _groq_clients:
        raise ValueError("No Groq API keys configured")
    client = _groq_clients[_current_groq_key_index]
    _current_groq_key_index = (_current_groq_key_index + 1) % len(_groq_clients)
    return client

def rotate_groq_key():
    """Manually rotate to next Groq API key."""
    global _current_groq_key_index
    _current_groq_key_index = (_current_groq_key_index + 1) % len(_groq_clients)
    print(f"🔄 Rotated to Groq API key {_current_groq_key_index + 1}/{len(_groq_clients)}")

# Model configurations
MODEL_DEBATER_A = "llama-3.3-70b-versatile"
MODEL_DEBATER_B = "llama-3.3-70b-versatile"
MODEL_JUDGE = "kimi-k2-instruct-0905"
MODEL_SAFETY_GUARD = "llama-guard-4-12b"

CONSISTENCY_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

CONSISTENCY_TIMEOUT = 20
SIMILARITY_THRESHOLD_BASE = 0.88
BAYESIAN_ALPHA = 2.0
BAYESIAN_BETA = 1.0

# Output directories
OUTPUT_DIR = Path("hadar_results")
LOGS_DIR = Path("hadar_logs")

MAX_DEBATE_ROUNDS = 3
MAX_WORDS_PER_TURN = 200

ENABLE_HADAR = True

# Visualization settings
DPI = 300
FIGURE_SIZE = (12, 8)
STYLE = "seaborn-v0_8-whitegrid"

def ensure_directories():
    OUTPUT_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "metrics").mkdir(exist_ok=True)
    (OUTPUT_DIR / "consistency_analysis").mkdir(exist_ok=True)
    (OUTPUT_DIR / "hallucination_analysis").mkdir(exist_ok=True)
    (OUTPUT_DIR / "aggregate_metrics").mkdir(exist_ok=True)
    (OUTPUT_DIR / "consistency_scores").mkdir(exist_ok=True)

def validate_api_keys() -> bool:
    if not GROQ_API_KEYS:
        print("❌ Error: No Groq API keys found")
        print("\nPlease set at least one API key in the .env file:")
        print("  GROQ_API_KEY=your_groq_key_here")
        print("  GROQ_API_KEY_2=your_second_groq_key_here")
        return False
    
    print("\n✓ API keys validated")
    print(f"✓ Groq API keys: {len(GROQ_API_KEYS)} configured")
    for i, key in enumerate(GROQ_API_KEYS, 1):
        print(f"  Key {i}: {'*' * 20}{key[-8:]}")
    print()
    return True
