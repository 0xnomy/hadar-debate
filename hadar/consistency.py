import os
import re
import gc
import time
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from functools import wraps, lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

from .config import LOGS_DIR

# Initialize sentence transformer for semantic similarity
_sentence_model = None
_session_log = {
    'session_start': None,
    'rounds': [],
    'model_latencies': {},
    'thresholds': [],
    'correction_acceptance_rates': []
}

# Initialize logging directory
_log_dir = Path(LOGS_DIR)
_log_dir.mkdir(exist_ok=True)

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model

def init_session_log():
    global _session_log
    _session_log = {
        'session_start': datetime.now().isoformat(),
        'rounds': [],
        'model_latencies': {},
        'thresholds': [],
        'correction_acceptance_rates': []
    }

def save_session_log():
    global _session_log, _log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _log_dir / f"session_{timestamp}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(_session_log, f, indent=2)
    print(f"  📝 Session log saved: {log_file}")


def retry_with_exponential_backoff(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorator to retry API calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a rate limit error
                    if 'rate_limit' in error_str.lower() or '429' in error_str:
                        if attempt < max_retries - 1:
                            print(f"  ⏳ Rate limit hit, retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                            continue
                    # For other errors or final retry, raise
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
                    delay *= 2
            return None
        return wrapper
    return decorator


def sentence_similarity(a: str, b: str) -> float:
    model = get_sentence_model()
    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()


def generate_paraphrased_prompts(topic: str, original_response: str = None) -> List[str]:
    if original_response:
        # Context-aware paraphrasing for fact-checking (optimized: reduced from 5 to 3 prompts)
        prompts = [
            f"Verify the factual accuracy of this claim about '{topic}': {original_response}",
            f"Fact-check this assertion regarding '{topic}': {original_response}",
            f"Evaluate the truthfulness of: {original_response} (in context of '{topic}')"
        ]
    else:
        # General topic paraphrasing (optimized: reduced from 5 to 3 prompts)
        prompts = [
            f"What are the key facts about: {topic}",
            f"Provide factual information regarding: {topic}",
            f"What are the verifiable facts concerning: {topic}"
        ]
    
    return prompts


def get_model_responses(groq_client: Groq, models: List[str], prompts: List[str], 
                       max_tokens: int = 300, use_rotation: bool = False) -> Dict[str, List[str]]:
    results = {model: [] for model in models}
    failed_models = set()
    
    if use_rotation:
        from .config import get_groq_client as get_rotating_client
        groq_client = get_rotating_client()
    
    @retry_with_exponential_backoff(max_retries=2, initial_delay=2.0)
    def query_model(model: str, prompt: str) -> Tuple[str, str, str]:
        """Helper function to query a single model with retry logic."""
        try:
            # Handle models with known context length issues
            if 'prompt-guard' in model.lower():
                # Truncate prompt for prompt-guard model (has smaller context)
                truncated_prompt = prompt[:500] if len(prompt) > 500 else prompt
                adjusted_max_tokens = min(max_tokens, 150)
            else:
                truncated_prompt = prompt
                adjusted_max_tokens = max_tokens
            
            # Add minor random jitter delay to prevent API overload
            time.sleep(0.1 + np.random.uniform(0, 0.1))
            
            response = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": truncated_prompt}],
                max_tokens=adjusted_max_tokens,
                temperature=0.3  # Lower temperature for consistency
            )
            content = response.choices[0].message.content.strip()
            return (model, prompt, content)
        except Exception as e:
            error_str = str(e)
            if 'reduce the length' in error_str.lower():
                print(f"  ⚠️  Error querying {model}: Context length exceeded, skipping")
            else:
                print(f"  ⚠️  Error querying {model}: {e}")
            return (model, prompt, "")
    
    # Reduced parallelism to avoid overwhelming API (max 3 concurrent for better reliability)
    max_workers = 3
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # Submit tasks
        for model in models:
            if model not in failed_models:
                for prompt in prompts:
                    future = executor.submit(query_model, model, prompt)
                    futures[future] = (model, prompt)
        
        # Collect results with timeout
        for future in as_completed(futures, timeout=60):
            model, prompt = futures[future]
            try:
                returned_model, returned_prompt, response = future.result(timeout=30)
                if response:
                    results[returned_model].append(response)
                else:
                    failed_models.add(returned_model)
            except TimeoutError:
                print(f"  ⚠️  Timeout querying {model}, skipping")
                failed_models.add(model)
            except Exception as e:
                print(f"  ⚠️  Unexpected error with {model}: {e}")
                failed_models.add(model)
    
    # Log summary of failed models
    if failed_models:
        print(f"  ℹ️  Models with errors (excluded from analysis): {', '.join(failed_models)}")
    
    # Ensure we have at least some responses
    successful_models = [m for m in models if m not in failed_models and results.get(m)]
    if len(successful_models) < 2:
        print(f"  ⚠️  Warning: Only {len(successful_models)} model(s) responded successfully")
    
    return results


def split_into_sentences(text: str) -> List[str]:
    # Simple sentence splitter - handles basic punctuation
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def compute_consistency_matrix(responses: Dict[str, List[str]], 
                               original_response: str) -> Dict[str, float]:
    # Split original response into sentences
    original_sentences = split_into_sentences(original_response)
    
    if not original_sentences:
        return {}
    
    # Collect all responses from all models
    all_responses = []
    for model_responses in responses.values():
        all_responses.extend(model_responses)
    
    if not all_responses:
        return {sent: 0.0 for sent in original_sentences}
    
    # For each sentence in original response, compute average similarity
    # to all model responses
    sentence_scores = {}
    model = get_sentence_model()
    
    for sent in original_sentences:
        sent_embedding = model.encode(sent, convert_to_tensor=True)
        similarities = []
        
        for response in all_responses:
            # Split response into sentences and find best matching sentence
            response_sentences = split_into_sentences(response)
            for resp_sent in response_sentences:
                resp_embedding = model.encode(resp_sent, convert_to_tensor=True)
                sim = util.cos_sim(sent_embedding, resp_embedding).item()
                
                # Normalize semantic similarity: all-MiniLM-L6-v2 typically ranges 0.3-0.7
                # Map to [0, 1] range for better interpretability
                sim_normalized = (sim - 0.3) / 0.4
                sim_normalized = np.clip(sim_normalized, 0.0, 1.0)
                
                similarities.append(sim_normalized)
        
        # Average consistency score for this sentence
        avg_score = np.mean(similarities) if similarities else 0.0
        sentence_scores[sent] = float(avg_score)
    
    return sentence_scores


def detect_hallucinations(consistency_scores: Dict[str, float], 
                         threshold: float = 0.65) -> Tuple[List[str], float]:
    global _session_log
    
    if not consistency_scores:
        return [], threshold
    
    scores = np.array(list(consistency_scores.values()))
    
    # Bayesian adaptive thresholding 2.0
    if len(scores) >= 3:
        baseline = np.mean(scores)
        uncertainty = np.std(scores)
        
        # Confidence factor: higher when uncertainty is low relative to baseline
        confidence_factor = np.clip(1 - (uncertainty / (baseline + 1e-6)), 0.5, 1.0)
        
        # Adaptive threshold scales with confidence
        adaptive_threshold = baseline * confidence_factor
        
        # Ensure threshold is reasonable (between 0.3 and 0.7)
        effective_threshold = np.clip(adaptive_threshold, 0.3, 0.7)
    else:
        effective_threshold = threshold
    
    # Log threshold for analysis
    _session_log['thresholds'].append({
        'threshold': float(effective_threshold),
        'baseline': float(np.mean(scores)),
        'uncertainty': float(np.std(scores))
    })
    
    hallucinated = []
    for sentence, score in consistency_scores.items():
        if score < effective_threshold:
            hallucinated.append(sentence)
    
    return hallucinated, effective_threshold


def correct_hallucinations(anchor_model: str, groq_client: Groq, 
                          sentences: List[str], topic: str,
                          original_scores: Dict[str, float],
                          model_responses: Dict[str, List[str]]) -> Tuple[List[str], List[bool]]:
    global _session_log
    
    corrected = []
    accepted = []
    acceptance_count = 0
    
    for i, sentence in enumerate(sentences):
        original_score = original_scores.get(sentence, 0.0)
        
        prompt = f"""The following statement was made in a debate about "{topic}":

        "{sentence}"

        This statement has been flagged as potentially inaccurate or unsupported.

        Please rewrite it to be factually accurate and verifiable. If the claim cannot be verified, 
        state it more cautiously with appropriate qualifiers (e.g., "may", "could", "some studies suggest").

        Provide ONLY the corrected sentence, without explanation."""

        # Add delay between correction requests to avoid rate limiting
        if i > 0:
            time.sleep(0.5)
        
        max_retries = 3
        corrected_sentence = None
        
        for attempt in range(max_retries):
            try:
                response = groq_client.chat.completions.create(
                    model=anchor_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.2
                )
                corrected_sentence = response.choices[0].message.content.strip()
                break
            except Exception as e:
                error_str = str(e)
                if 'rate_limit' in error_str.lower() or '429' in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  ⏳ Rate limit on correction, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                if attempt == max_retries - 1:
                    print(f"  ⚠️  Error correcting sentence with {anchor_model}: {e}")
                    corrected_sentence = f"Some studies suggest that {sentence.lower()}"
        
        # Multi-stage validation: check if correction improves consistency
        if corrected_sentence:
            # Re-check similarity against all model responses
            new_similarities = []
            model = get_sentence_model()
            corrected_emb = model.encode(corrected_sentence, convert_to_tensor=True)
            
            all_responses = []
            for model_resp in model_responses.values():
                all_responses.extend(model_resp)
            
            for response in all_responses:
                response_sentences = split_into_sentences(response)
                for resp_sent in response_sentences:
                    resp_emb = model.encode(resp_sent, convert_to_tensor=True)
                    sim = util.cos_sim(corrected_emb, resp_emb).item()
                    # Normalize
                    sim_normalized = (sim - 0.3) / 0.4
                    sim_normalized = np.clip(sim_normalized, 0.0, 1.0)
                    new_similarities.append(sim_normalized)
            
            new_score = np.mean(new_similarities) if new_similarities else 0.0
            
            # Accept correction only if new_score > old_score + 0.05
            if new_score > original_score + 0.05:
                corrected.append(corrected_sentence)
                accepted.append(True)
                acceptance_count += 1
            else:
                # Keep original
                corrected.append(sentence)
                accepted.append(False)
        else:
            corrected.append(sentence)
            accepted.append(False)
    
    # Log acceptance rate
    acceptance_rate = acceptance_count / len(sentences) if sentences else 0.0
    _session_log['correction_acceptance_rates'].append(float(acceptance_rate))
    
    return corrected, accepted


def replace_low_consistency_segments(original_response: str, 
                                    hallucinated_sents: List[str],
                                    corrected_sents: List[str]) -> str:
    corrected_response = original_response
    
    for original, corrected in zip(hallucinated_sents, corrected_sents):
        # Replace the hallucinated sentence with corrected version
        corrected_response = corrected_response.replace(original, corrected)
    
    return corrected_response


def visualize_consistency_matrix(consistency_scores: Dict[str, float], 
                                round_num: int, debater: str,
                                output_dir: str = "hadar_results") -> None:
    if not consistency_scores:
        return
    
    # Create consistency_scores subfolder
    consistency_dir = Path(output_dir) / "consistency_scores"
    consistency_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    sentences = list(consistency_scores.keys())
    scores = list(consistency_scores.values())
    
    # Truncate long sentences for display
    display_sentences = [s[:60] + "..." if len(s) > 60 else s for s in sentences]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(sentences) * 0.5)))
    
    # Create heatmap data
    data = np.array(scores).reshape(-1, 1)
    
    # Plot heatmap
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Consistency Score'},
                yticklabels=display_sentences, xticklabels=['Score'],
                ax=ax)
    
    ax.set_title(f'Sentence Consistency Scores - Round {round_num}, Debater {debater}')
    ax.set_xlabel('')
    
    # Add threshold line
    threshold = 0.65
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure in consistency_scores folder
    filename = consistency_dir / f'consistency_round{round_num}_debater{debater}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close specific figure
    plt.close('all')  # Close all figures
    gc.collect()  # Force garbage collection
    
    print(f"  📊 Consistency heatmap saved: {filename}")


def apply_hadar_layer(groq_client: Groq, debater_response: str, 
                      debate_topic: str, round_num: int, debater: str,
                      consistency_models: List[str],
                      anchor_model: str = "llama-3.3-70b-versatile",
                      threshold: float = 0.65,
                      visualize: bool = True) -> Tuple[str, Dict[str, any]]:
    """Apply HADAR consistency verification layer to debater response."""
    print(f"  🔍 Running HADAR consistency check for Debater {debater}...")
    
    start_time = time.time()
    global _session_log
    
    metrics = {
        'mean_consistency': 0.0,
        'hallucinated_count': 0,
        'corrected_count': 0,
        'total_sentences': 0,
        'acceptance_ratio': 0.0,
        'adaptive_threshold': threshold
    }
    
    try:
        # Step 1: Generate paraphrased prompts
        prompts = generate_paraphrased_prompts(debate_topic, debater_response)
        
        # Step 2: Get responses from multiple models
        print(f"    → Querying {len(consistency_models)} models with {len(prompts)} prompts...")
        model_start = time.time()
        responses = get_model_responses(groq_client, consistency_models, prompts)
        model_latency = time.time() - model_start
        
        # Log latency
        _session_log['model_latencies'][f'round_{round_num}_debater_{debater}'] = float(model_latency)
        
        # Step 3: Compute consistency matrix
        print(f"    → Computing sentence-level consistency scores...")
        consistency_scores = compute_consistency_matrix(responses, debater_response)
        
        if not consistency_scores:
            print(f"    ⚠️  No consistency scores computed, using original response")
            return debater_response, metrics
        
        # Update metrics
        metrics['total_sentences'] = len(consistency_scores)
        metrics['mean_consistency'] = float(np.mean(list(consistency_scores.values())))
        
        print(f"    → Mean consistency score: {metrics['mean_consistency']:.3f}")
        
        # Step 4: Detect hallucinations with Bayesian adaptive thresholding
        hallucinated_sents, adaptive_threshold = detect_hallucinations(consistency_scores, threshold)
        metrics['hallucinated_count'] = len(hallucinated_sents)
        metrics['adaptive_threshold'] = float(adaptive_threshold)
        
        print(f"    → Adaptive threshold: {adaptive_threshold:.3f}")
        
        if hallucinated_sents:
            print(f"    ⚠️  Detected {len(hallucinated_sents)} low-consistency sentence(s)")
            
            # Step 5: Correct hallucinations with multi-stage validation
            print(f"    → Applying corrections using {anchor_model}...")
            corrected_sents, acceptance_flags = correct_hallucinations(
                anchor_model, groq_client, hallucinated_sents, debate_topic,
                consistency_scores, responses
            )
            
            accepted_count = sum(acceptance_flags)
            metrics['corrected_count'] = accepted_count
            metrics['acceptance_ratio'] = float(accepted_count / len(hallucinated_sents)) if hallucinated_sents else 0.0
            
            print(f"    ✅ Accepted {accepted_count}/{len(hallucinated_sents)} corrections (ratio: {metrics['acceptance_ratio']:.2f})")
            
            # Step 6: Replace segments
            corrected_response = replace_low_consistency_segments(
                debater_response, hallucinated_sents, corrected_sents
            )
            
            # Step 7: Recalculate post-correction consistency
            print(f"    → Rechecking consistency after corrections...")
            post_correction_scores = compute_consistency_matrix(responses, corrected_response)
            
            if post_correction_scores:
                metrics['post_mean_consistency'] = float(np.mean(list(post_correction_scores.values())))
                consistency_gain = metrics['post_mean_consistency'] - metrics['mean_consistency']
                
                # Tanh-based correction efficiency (bounded in [-1, 1])
                metrics['correction_efficiency'] = float(np.tanh(
                    consistency_gain / (metrics['hallucinated_count'] + 1e-6)
                ))
                
                print(f"    📈 Post-correction consistency: {metrics['post_mean_consistency']:.3f} (+{consistency_gain:.3f})")
                print(f"    📊 Correction efficiency: {metrics['correction_efficiency']:.3f}")
            else:
                metrics['post_mean_consistency'] = metrics['mean_consistency']
                metrics['correction_efficiency'] = 0.0
        else:
            print(f"    ✅ All sentences meet consistency threshold")
            corrected_response = debater_response
            metrics['post_mean_consistency'] = metrics['mean_consistency']
            metrics['correction_efficiency'] = 0.0
            metrics['acceptance_ratio'] = 0.0
        
        # Step 8: Log sentence-level scores for visualization
        metrics['sentence_scores'] = list(consistency_scores.values())
        
        # Step 9: Log round data
        total_time = time.time() - start_time
        _session_log['rounds'].append({
            'round': round_num,
            'debater': debater,
            'duration': float(total_time),
            'metrics': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v 
                       for k, v in metrics.items() if k != 'sentence_scores'}
        })
        
        # Step 10: Visualize (optional)
        if visualize:
            visualize_consistency_matrix(consistency_scores, round_num, debater)
        
        return corrected_response, metrics
        
    except Exception as e:
        print(f"    ❌ Error in HADAR layer: {e}")
        print(f"    → Falling back to original response")
        return debater_response, metrics

# Backward compatibility aliases
apply_finch_zk_layer = apply_hadar_layer
