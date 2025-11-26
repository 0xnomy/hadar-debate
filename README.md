# HADAR - Hallucination Adaptive Detection & Automated Rectification

A multi-model consistency verification system for detecting and correcting hallucinations in LLM outputs through cross-model debate. Built for **Groq-hosted LLMs** (LLaMA, Mixtral, etc.) leveraging Groq's fast inference API.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure API keys in .env
GROQ_API_KEY=your_key_here

# Run
python main.py
```

## Core Concept

HADAR detects hallucinations by querying multiple LLMs with paraphrased fact-checking prompts, then computing semantic similarity between responses. Sentences with low cross-model consistency are flagged and corrected.

```
Response → Paraphrased Prompts → Multi-Model Verification → Consistency Scoring → Correction
```

## Key Algorithm

**Bayesian Adaptive Thresholding:**
```python
threshold = clip(mean(scores) × (1 - std/mean), 0.3, 0.7)
```
- Low consistency + high variance → more sensitive detection
- High consistency + low variance → less sensitive detection

**Correction Validation:**
- Corrections accepted only if `new_score > old_score + 0.05`

## Project Structure

```
hadar-debate/
├── main.py              # Entry point (interactive CLI)
├── hadar/
│   ├── config.py        # API keys, model configs
│   ├── consistency.py   # Core HADAR detection engine
│   ├── debate.py        # Debate orchestration
│   ├── adaptive.py      # Topic-aware model selection
│   ├── analysis.py      # Metrics computation
│   └── visualizer.py    # Visualization generation
├── hadar_results/       # Output: transcripts, metrics, plots
└── hadar_logs/          # Session logs (JSON)
```

## Usage

### Interactive Mode
```bash
python main.py
# Select: 1) Standard Debate  2) Adaptive Debate  3) Exit
```

### Programmatic
```python
from hadar import run_debate, apply_hadar_layer, analyze_debate_metrics

# Run full debate
metrics, judgment = run_debate("Is consciousness computable?")

# Or apply HADAR to any text
from hadar import get_groq_client, CONSISTENCY_MODELS
corrected, metrics = apply_hadar_layer(
    get_groq_client(), 
    text, topic, round_num=1, debater="A",
    consistency_models=CONSISTENCY_MODELS
)
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Mean Consistency** | Avg sentence-level consistency (0-1) |
| **Hallucination Rate** | % sentences below threshold |
| **Correction Efficiency** | `tanh(Δconsistency / hallucination_count)` |
| **HII** | `(1 - hallucination_rate) × mean_consistency` |

## Configuration

Edit `hadar/config.py`:
```python
MODEL_DEBATER_A = "llama-3.3-70b-versatile"
MODEL_DEBATER_B = "llama-3.3-70b-versatile"
MODEL_JUDGE = "kimi-k2-instruct-0905"
CONSISTENCY_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
ENABLE_HADAR = True
MAX_DEBATE_ROUNDS = 3
```

## Dependencies

```
groq, python-dotenv, sentence-transformers, numpy, pandas, matplotlib, seaborn, torch
```

## Output

- `hadar_results/debate_transcript.txt` - Full debate text
- `hadar_results/metrics/*.csv|json` - Aggregate & detailed metrics
- `hadar_results/consistency_scores/*.png` - Per-round heatmaps
- `hadar_logs/session_*.json` - Timing, thresholds, acceptance rates

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No API keys | Add `GROQ_API_KEY` to `.env` |
| Rate limits | Add `GROQ_API_KEY_2`, `GROQ_API_KEY_3` for rotation |
| Context exceeded | Automatic truncation (logged as warning) |

## Citation

```bibtex
@software{hadar2025,
  title={HADAR: Hallucination Detection via Agentic Reasoning},
  year={2025}
}
```

## License

[Your License]

---
*Last Updated: November 2025*
