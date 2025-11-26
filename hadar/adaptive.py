"""
HADAR Adaptive Model Selection Module

Provides intelligent model selection based on topic classification
and optimized debate configuration for different domains.
"""

import time
from typing import List, Dict, Tuple

from .config import (
    get_groq_client,
    CONSISTENCY_MODELS,
    MODEL_DEBATER_A,
    MODEL_DEBATER_B,
    MODEL_JUDGE,
    ensure_directories,
)
from .consistency import (
    apply_hadar_layer,
    init_session_log,
    save_session_log,
)


# Topic classification categories
TOPIC_CATEGORIES = {
    'scientific': ['science', 'physics', 'chemistry', 'biology', 'research', 'experiment', 'hypothesis', 'theory'],
    'technical': ['technology', 'computer', 'software', 'programming', 'algorithm', 'ai', 'machine learning', 'data'],
    'philosophical': ['philosophy', 'ethics', 'morality', 'consciousness', 'existence', 'meaning', 'truth', 'reality'],
    'historical': ['history', 'war', 'civilization', 'ancient', 'revolution', 'empire', 'century', 'era'],
    'social': ['society', 'culture', 'politics', 'economics', 'policy', 'government', 'democracy', 'rights'],
    'general': []  # Default category
}

# Model specializations for different topic types
MODEL_SPECIALIZATIONS = {
    'scientific': {
        'primary_models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
        'consistency_boost': 0.05
    },
    'technical': {
        'primary_models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
        'consistency_boost': 0.03
    },
    'philosophical': {
        'primary_models': ['llama-3.3-70b-versatile'],
        'consistency_boost': 0.0
    },
    'historical': {
        'primary_models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
        'consistency_boost': 0.02
    },
    'social': {
        'primary_models': ['llama-3.3-70b-versatile'],
        'consistency_boost': 0.0
    },
    'general': {
        'primary_models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
        'consistency_boost': 0.0
    }
}


def classify_topic(topic: str) -> str:
    """
    Classify a debate topic into a category.
    
    Args:
        topic: The debate topic string
        
    Returns:
        Category string (scientific, technical, philosophical, etc.)
    """
    topic_lower = topic.lower()
    
    for category, keywords in TOPIC_CATEGORIES.items():
        if category == 'general':
            continue
        for keyword in keywords:
            if keyword in topic_lower:
                return category
    
    return 'general'


def select_models_for_topic(topic: str, num_models: int = 7) -> List[str]:
    """
    Select optimal consistency models based on topic classification.
    
    Args:
        topic: The debate topic
        num_models: Maximum number of models to select
        
    Returns:
        List of model identifiers for consistency checking
    """
    category = classify_topic(topic)
    specialization = MODEL_SPECIALIZATIONS.get(category, MODEL_SPECIALIZATIONS['general'])
    
    # Start with specialized primary models
    selected = specialization['primary_models'].copy()
    
    # Add remaining models from general pool
    for model in CONSISTENCY_MODELS:
        if model not in selected and len(selected) < num_models:
            selected.append(model)
    
    print(f"  📋 Topic category: {category}")
    print(f"  🔧 Selected {len(selected)} consistency models")
    
    return selected[:num_models]


def run_adaptive_debate(
    topic: str,
    consistency_models: List[str] = None,
    max_rounds: int = 5,
    visualize: bool = True
) -> Tuple[List[Dict], List[str]]:
    """
    Run a debate with adaptive model selection and HADAR verification.
    
    Args:
        topic: The debate topic
        consistency_models: Optional list of models for consistency checking
        max_rounds: Maximum number of debate rounds
        visualize: Whether to generate visualizations
        
    Returns:
        Tuple of (metrics_list, transcript_list)
    """
    from .debate import debate_turn, judge_round, save_transcript
    
    ensure_directories()
    init_session_log()
    
    if consistency_models is None:
        consistency_models = select_models_for_topic(topic)
    
    category = classify_topic(topic)
    consistency_boost = MODEL_SPECIALIZATIONS.get(category, {}).get('consistency_boost', 0.0)
    base_threshold = 0.65 + consistency_boost
    
    groq_client = get_groq_client()
    
    debate_history = []
    metrics_list = []
    transcript = []
    
    print(f"\n{'='*60}")
    print(f"ADAPTIVE HADAR DEBATE")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print(f"Category: {category}")
    print(f"Consistency Threshold: {base_threshold:.2f}")
    print(f"{'='*60}\n")
    
    transcript.append(f"ADAPTIVE HADAR DEBATE TRANSCRIPT")
    transcript.append(f"{'='*50}")
    transcript.append(f"Topic: {topic}")
    transcript.append(f"Category: {category}")
    transcript.append(f"{'='*50}\n")
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- ROUND {round_num} ---\n")
        transcript.append(f"\nROUND {round_num}")
        transcript.append("-" * 20)
        
        # Debater A's turn
        print("Debater A (FOR) speaking...")
        response_a = debate_turn(MODEL_DEBATER_A, "for", debate_history, topic)
        
        if response_a:
            # Apply HADAR layer
            corrected_a, metrics_a = apply_hadar_layer(
                groq_client, response_a, topic, round_num, "A",
                consistency_models, threshold=base_threshold, visualize=visualize
            )
            
            metrics_a['round'] = round_num
            metrics_a['debater'] = 'A'
            metrics_list.append(metrics_a)
            
            debate_history.append({"role": "Debater A", "content": corrected_a})
            transcript.append(f"\nDEBATER A (FOR):\n{corrected_a}")
        
        time.sleep(1)  # Rate limiting
        
        # Debater B's turn
        print("\nDebater B (AGAINST) speaking...")
        response_b = debate_turn(MODEL_DEBATER_B, "against", debate_history, topic)
        
        if response_b:
            # Apply HADAR layer
            corrected_b, metrics_b = apply_hadar_layer(
                groq_client, response_b, topic, round_num, "B",
                consistency_models, threshold=base_threshold, visualize=visualize
            )
            
            metrics_b['round'] = round_num
            metrics_b['debater'] = 'B'
            metrics_list.append(metrics_b)
            
            debate_history.append({"role": "Debater B", "content": corrected_b})
            transcript.append(f"\nDEBATER B (AGAINST):\n{corrected_b}")
        
        time.sleep(1)  # Rate limiting
    
    # Final judgment
    print("\n--- FINAL JUDGMENT ---\n")
    if len(debate_history) >= 2:
        final_judgment = judge_round(topic, debate_history[-2].get('content', ''), 
                                     debate_history[-1].get('content', ''))
        transcript.append(f"\n{'='*50}")
        transcript.append(f"FINAL JUDGMENT:\n{final_judgment}")
    else:
        final_judgment = "Insufficient debate rounds for judgment."
        transcript.append(f"\n{final_judgment}")
    
    print(f"\n{final_judgment}")
    
    # Save outputs
    save_transcript(topic, debate_history, final_judgment)
    save_session_log()
    
    return metrics_list, transcript


__all__ = ["classify_topic", "select_models_for_topic", "run_adaptive_debate"]
