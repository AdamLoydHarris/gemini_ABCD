import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

from google import genai


from ABCD import ABCD, random_task

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "gemini-3.1-flash-lite-preview"  # free tier: 15 RPM, 1500 RPD
STEPS_PER_SESSION = 50
NUM_SESSIONS = 9
REQUEST_DELAY = 8.0   # seconds between API calls (~7 RPM, comfortable under 15 RPM free limit)
MAX_BACKOFF = 60.0
LOG_DIR = Path(__file__).parent.parent / "logs"

INTRO_PROMPT = """You are playing a navigation game on a 3x3 grid.

Rules:
- You are shown as 'O' on the grid; all other cells are 'X'
- Each turn you must reply with exactly one letter: N, E, S, or W (for North, East, South, West)
- Moving into a wall keeps you in place
- After each move you will receive a reward of +0 or +1
- Your goal is to collect as many +1 rewards as possible over 50 steps
- The reward pattern is determined by a hidden task that you must try to figure out and exploit

Ready to begin."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_call_with_backoff(chat, message: str, delay: float) -> tuple[str, float]:
    """Send a message, return (response_text, updated_delay)."""
    time.sleep(delay)
    while True:
        try:
            response = chat.send_message(message)
            return response.text, REQUEST_DELAY
        except Exception as e:
            err = str(e)
            if "429" in err:
                # Daily quota exhausted — retrying won't help
                if "PerDay" in err or "per_day" in err.lower():
                    raise RuntimeError(f"Daily quota exhausted: {err[:300]}") from e
                wait = min(delay * 2, MAX_BACKOFF)
                print(f"  [rate limit] sleeping {wait:.0f}s ...")
                time.sleep(wait)
                delay = wait
            elif "503" in err or "500" in err or \
                    "unavailable" in err.lower() or "internal error" in err.lower():
                wait = min(delay * 2, MAX_BACKOFF)
                print(f"  [server error] sleeping {wait:.0f}s ...")
                time.sleep(wait)
                delay = wait
            else:
                raise


def parse_action(raw: str) -> tuple[str, bool]:
    """Extract N/E/S/W from model response; return (action, parse_error)."""
    cleaned = raw.strip().upper()
    for ch in cleaned:
        if ch in ('N', 'E', 'S', 'W'):
            return ch, False
    fallback = random.choice(['N', 'E', 'S', 'W'])
    return fallback, True


def log_record(log_fh, record: dict):
    log_fh.write(json.dumps(record) + '\n')
    log_fh.flush()


def step_prompt(step: int, grid: str, last_reward: int, session_total: int) -> str:
    return (
        f"Step {step}/{STEPS_PER_SESSION} | "
        f"Last reward: +{last_reward} | "
        f"Session total: {session_total}\n\n"
        f"{grid}\n\n"
        f"What is your next move? Reply with exactly one letter: N, E, S, or W."
    )


def reflection_prompt(session_total: int) -> str:
    return (
        f"The session is now over ({STEPS_PER_SESSION} steps completed). "
        f"Your session total was {session_total}/{STEPS_PER_SESSION}.\n\n"
        "Please reflect: What do you think is happening in this game? "
        "What patterns have you noticed? What is your strategy going forward?"
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    # Load API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        key_file = Path(__file__).parent.parent / "api_key.txt"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY env var or place key in api_key.txt"
        )

    client = genai.Client(api_key=api_key)

    run_dir = LOG_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "experiment_log.jsonl"

    task = random_task()
    env = ABCD(task)

    with log_file.open('w') as log_fh:
        for session in range(1, NUM_SESSIONS + 1):
            print(f"\n{'='*50}")
            print(f"Session {session}/{NUM_SESSIONS}  |  task: {env.task}")
            print(f"{'='*50}")

            chat = client.chats.create(model=MODEL)
            delay = REQUEST_DELAY

            # Send introduction
            _, delay = api_call_with_backoff(chat, INTRO_PROMPT, delay)

            session_total = 0
            last_reward = 0

            for step in range(1, STEPS_PER_SESSION + 1):
                grid = env.render_grid()
                prompt = step_prompt(step, grid, last_reward, session_total)

                raw_response, delay = api_call_with_backoff(chat, prompt, delay)
                action, parse_error = parse_action(raw_response)

                result = env.step(action)
                last_reward = result['reward']
                session_total += last_reward

                record = {
                    "type": "step",
                    "session": session,
                    "step": step,
                    "task": env.task,
                    "agent_node": result['node'],
                    "agent_rc": list(result['rc']),
                    "current_goal": result['current_goal'],
                    "goal_node": result['goal_node'],
                    "goal_rc": list(result['goal_rc']),
                    "action": action,
                    "reward": last_reward,
                    "session_total": session_total,
                    "grid": grid,
                    "llm_raw_response": raw_response,
                    "parse_error": parse_error,
                }
                log_record(log_fh, record)

                status = f"  step {step:2d}: action={action} reward={last_reward} total={session_total}"
                if parse_error:
                    status += " [PARSE ERROR]"
                print(status)

            # Post-session reflection
            refl_text, delay = api_call_with_backoff(
                chat, reflection_prompt(session_total), delay
            )
            log_record(log_fh, {
                "type": "reflection",
                "session": session,
                "session_total": session_total,
                "reflection": refl_text,
            })
            print(f"\nReflection:\n{refl_text}")

            # Scramble for next session
            if session < NUM_SESSIONS:
                env.scramble()

    print(f"\nExperiment complete. Log saved to {log_file}")


if __name__ == "__main__":
    run_experiment()
