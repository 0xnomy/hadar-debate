"""
HADAR Debate Module

Core debate orchestration including:
- Debate turn management
- Judge evaluation
- Transcript generation
- HADAR integration
"""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .config import (
    get_groq_client,
    MODEL_DEBATER_A,
    MODEL_DEBATER_B,
    MODEL_JUDGE,
    CONSISTENCY_MODELS,
    MAX_DEBATE_ROUNDS,
    MAX_WORDS_PER_TURN,
    ENABLE_HADAR,
    OUTPUT_DIR,
    ensure_directories,
)
from .consistency import (
    apply_hadar_layer,
    init_session_log,
    save_session_log,
)


# Default debate topic
DEBATE_TOPIC = "Would humanity pass its own Turing test?"

# Global metrics storage for backward compatibility
hadar_metrics: List[Dict] = []


def debate_turn(
    model_id: str,
    role: str,
    history: List[Dict],
    topic: str,
    max_tokens: int = 500
) -> str:
    """
    Execute a single debate turn.
    
    Args:
        model_id: The LLM model to use
        role: 'for' or 'against' the topic
        history: Previous debate exchanges
        topic: The debate topic
        max_tokens: Maximum response tokens
        
    Returns:
        The debater's response text
    """
    groq_client = get_groq_client()
    
    # Build conversation context
    position = "in favor of" if role == "for" else "against"
    
    system_prompt = f"""You are participating in a formal debate about: "{topic}"

Your position: You are arguing {position} this proposition.

Guidelines:
- Present clear, logical arguments
- Use evidence and examples to support your points
- Address counterarguments from the opposing side
- Be concise but thorough (aim for {MAX_WORDS_PER_TURN} words or less)
- Maintain a respectful, academic tone"""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add debate history
    for entry in history:
        role_map = {"Debater A": "assistant" if role == "for" else "user",
                    "Debater B": "assistant" if role == "against" else "user"}
        msg_role = role_map.get(entry.get("role"), "user")
        messages.append({"role": msg_role, "content": entry.get("content", "")})
    
    # Add prompt for this turn
    if history:
        messages.append({
            "role": "user",
            "content": f"Please respond to the previous argument and continue making your case {position} the topic."
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Please present your opening argument {position} the topic: {topic}"
        })
    
    try:
        response = groq_client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️ Error in debate turn ({model_id}): {e}")
        return ""


def judge_round(topic: str, response_a: str, response_b: str) -> str:
    """
    Have the judge evaluate a debate round.
    
    Args:
        topic: The debate topic
        response_a: Debater A's response
        response_b: Debater B's response
        
    Returns:
        Judge's evaluation text
    """
    groq_client = get_groq_client()
    
    judge_prompt = f"""You are an impartial debate judge evaluating arguments about: "{topic}"

DEBATER A (FOR):
{response_a}

DEBATER B (AGAINST):
{response_b}

Please evaluate both arguments based on:
1. Logical coherence and reasoning
2. Use of evidence and examples
3. Addressing of counterarguments
4. Persuasiveness and clarity

Provide a brief assessment of each debater's performance and declare which argument was stronger in this round."""

    try:
        response = groq_client.chat.completions.create(
            model=MODEL_JUDGE,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️ Error in judge evaluation: {e}")
        return "Unable to evaluate this round due to an error."


def save_transcript(
    topic: str,
    history: List[Dict],
    final_judgment: str,
    filename: str = "debate_transcript.txt"
) -> None:
    """
    Save the debate transcript to a file.
    
    Args:
        topic: The debate topic
        history: Full debate history
        final_judgment: Judge's final evaluation
        filename: Output filename
    """
    output_path = Path(OUTPUT_DIR) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("LLM DEBATE TRANSCRIPT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Topic: {topic}\n")
        f.write(f"Debater A: {MODEL_DEBATER_A}\n")
        f.write(f"Debater B: {MODEL_DEBATER_B}\n")
        f.write(f"Judge: {MODEL_JUDGE}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n\n")
        
        round_num = 0
        for i, entry in enumerate(history):
            if entry.get('role') == 'Debater A' and (i == 0 or history[i-1].get('role') != 'Debater A'):
                round_num += 1
                f.write(f"\nROUND {round_num}\n")
                f.write("-" * 20 + "\n\n")
            
            role = entry.get('role', 'Unknown')
            content = entry.get('content', '')
            position = "(FOR)" if role == "Debater A" else "(AGAINST)"
            f.write(f"{role.upper()} {position}:\n{content}\n\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("FINAL JUDGMENT:\n")
        f.write(final_judgment + "\n")
    
    print(f"  📝 Transcript saved: {output_path}")


def run_debate(
    topic: str = None,
    max_rounds: int = None,
    enable_hadar: bool = None
) -> Tuple[List[Dict], str]:
    """
    Run a full debate session.
    
    Args:
        topic: The debate topic (uses default if None)
        max_rounds: Number of debate rounds (uses config default if None)
        enable_hadar: Whether to apply HADAR layer (uses config default if None)
        
    Returns:
        Tuple of (metrics_list, final_judgment)
    """
    global hadar_metrics
    
    if topic is None:
        topic = DEBATE_TOPIC
    if max_rounds is None:
        max_rounds = MAX_DEBATE_ROUNDS
    if enable_hadar is None:
        enable_hadar = ENABLE_HADAR
    
    ensure_directories()
    init_session_log()
    
    groq_client = get_groq_client()
    debate_history = []
    metrics_list = []
    
    print(f"\n{'='*60}")
    print(f"LLM DEBATE SYSTEM")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print(f"Debater A: {MODEL_DEBATER_A}")
    print(f"Debater B: {MODEL_DEBATER_B}")
    print(f"Judge: {MODEL_JUDGE}")
    print(f"HADAR Enabled: {enable_hadar}")
    print(f"{'='*60}\n")
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- ROUND {round_num} ---\n")
        
        # Debater A's turn
        print("Debater A (FOR) speaking...")
        response_a = debate_turn(MODEL_DEBATER_A, "for", debate_history, topic)
        
        if response_a and enable_hadar:
            corrected_a, metrics_a = apply_hadar_layer(
                groq_client, response_a, topic, round_num, "A",
                CONSISTENCY_MODELS
            )
            metrics_a['round'] = round_num
            metrics_a['debater'] = 'A'
            metrics_list.append(metrics_a)
            response_a = corrected_a
        
        if response_a:
            debate_history.append({"role": "Debater A", "content": response_a})
            print(f"\n{response_a[:200]}..." if len(response_a) > 200 else f"\n{response_a}")
        
        time.sleep(1)  # Rate limiting
        
        # Debater B's turn
        print("\nDebater B (AGAINST) speaking...")
        response_b = debate_turn(MODEL_DEBATER_B, "against", debate_history, topic)
        
        if response_b and enable_hadar:
            corrected_b, metrics_b = apply_hadar_layer(
                groq_client, response_b, topic, round_num, "B",
                CONSISTENCY_MODELS
            )
            metrics_b['round'] = round_num
            metrics_b['debater'] = 'B'
            metrics_list.append(metrics_b)
            response_b = corrected_b
        
        if response_b:
            debate_history.append({"role": "Debater B", "content": response_b})
            print(f"\n{response_b[:200]}..." if len(response_b) > 200 else f"\n{response_b}")
        
        time.sleep(1)  # Rate limiting
    
    # Final judgment
    print("\n--- FINAL JUDGMENT ---\n")
    if len(debate_history) >= 2:
        final_judgment = judge_round(topic, debate_history[-2].get('content', ''),
                                     debate_history[-1].get('content', ''))
    else:
        final_judgment = "Insufficient debate content for judgment."
    
    print(final_judgment)
    
    # Save outputs
    save_transcript(topic, debate_history, final_judgment)
    save_session_log()
    
    # Update global metrics
    hadar_metrics = metrics_list
    
    return metrics_list, final_judgment


__all__ = ["run_debate", "debate_turn", "judge_round", "save_transcript", "hadar_metrics", "DEBATE_TOPIC"]
