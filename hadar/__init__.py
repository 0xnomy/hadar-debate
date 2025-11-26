from .config import (
    get_groq_client,
    rotate_groq_key,
    validate_api_keys,
    ensure_directories,
    OUTPUT_DIR,
    LOGS_DIR,
    CONSISTENCY_MODELS,
    ENABLE_HADAR,
)

from .consistency import apply_hadar_layer, init_session_log, save_session_log

from .analysis import analyze_debate_metrics, HadarAnalyzer

from .visualizer import HadarVisualizer

from .adaptive import select_models_for_topic, run_adaptive_debate, classify_topic

from .debate import run_debate, debate_turn, judge_round, save_transcript, hadar_metrics, DEBATE_TOPIC

__version__ = "2.0.0"
__author__ = "Research Team"

__all__ = [
    # Config
    "get_groq_client",
    "rotate_groq_key",
    "validate_api_keys",
    "ensure_directories",
    "OUTPUT_DIR",
    "LOGS_DIR",
    # Consistency
    "apply_hadar_layer",
    "init_session_log",
    "save_session_log",
    # Analysis
    "analyze_debate_metrics",
    "HadarAnalyzer",
    # Visualization
    "HadarVisualizer",
    # Adaptive
    "select_models_for_topic",
    "run_adaptive_debate",
    "classify_topic",
    # Debate
    "run_debate",
    "debate_turn",
    "judge_round",
    "save_transcript",
    "hadar_metrics",
    "DEBATE_TOPIC",
]
