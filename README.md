# gemini_ABCD

Can an LLM learn a sequential reward task purely from sparse feedback, and generalise after the reward locations are shuffled?

## The Task

A Gemini model navigates a 3×3 grid by replying N / E / S / W. Four goal locations (A → B → C → D → A …) are hidden at fixed grid positions. The agent receives **+1** when it reaches the current goal, **+0** otherwise. It is told nothing about the goal structure. It must discover locations and sequence from reward alone.

After each 50-step session the model is asked to reflect, then goal locations are **scrambled** to new positions.  

## Setup

```bash
conda activate abcd-meta
pip install -r Code/requirements.txt
```

Place your Gemini API key in `api_key.txt` (gitignored).

## Running

```bash
python Code/gemini_agent.py    # runs the experiment, writes logs/<timestamp>/
python Code/analysis.py        # analyses the most recent run
python Code/analysis.py logs/2026-04-22_120000   # or a specific run
```

Key settings at the top of `gemini_agent.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `gemini-3.1-flash-lite-preview` | Gemini model ID |
| `NUM_SESSIONS` | `9` | Sessions per experiment |
| `STEPS_PER_SESSION` | `50` | Steps per session |
| `REQUEST_DELAY` | `8.0 s` | Delay between API calls (free tier: 15 RPM) |

## Output

Each run writes to `logs/<timestamp>/`:

```
experiment_log.jsonl   # one JSON record per step + one per reflection
```

Each step record contains agent position, current goal label, goal node, action, reward, and the raw model response.


## Repo structure

```
Code/
  ABCD.py            # environment (grid, rewards, scramble)
  gemini_agent.py    # experiment runner
  analysis.py        # trajectory analysis and plots
  requirements.txt
logs/                # created at runtime, gitignored
api_key.txt          # gitignored
```