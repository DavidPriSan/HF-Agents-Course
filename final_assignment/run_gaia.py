"""
Local runner for the Unit 4 final assignment (no Space / OAuth needed).

    python run_gaia.py                  # answer all questions, cache them in answers.json
    python run_gaia.py --limit 3        # try only the first 3 questions
    python run_gaia.py --retry-errors   # re-run questions that failed
    python run_gaia.py --submit         # submit the cached answers
"""
import argparse
import json
import os
from pathlib import Path

import requests

from agent import GaiaAgent

API_URL = "https://agents-course-unit4-scoring.hf.space"
HERE = Path(__file__).parent
ANSWERS_FILE = HERE / "answers.json"
FILES_DIR = HERE / "files"
DEFAULT_USERNAME = "DavidPriSan"
DEFAULT_AGENT_CODE = "https://github.com/DavidPriSan/HF-Agents-Course/tree/main/final_assignment"


def fetch_questions():
    response = requests.get(f"{API_URL}/questions", timeout=30)
    response.raise_for_status()
    return response.json()


def download_file(task_id: str, file_name: str):
    if not file_name:
        return None
    FILES_DIR.mkdir(exist_ok=True)
    path = FILES_DIR / file_name
    if not path.exists():
        response = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return str(path)


def load_answers():
    if ANSWERS_FILE.exists():
        return json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
    return {}


def save_answers(answers):
    ANSWERS_FILE.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")


def run(limit, retry_errors):
    questions = fetch_questions()[:limit] if limit else fetch_questions()
    answers = load_answers()
    agent = GaiaAgent()

    for i, item in enumerate(questions, 1):
        task_id = item["task_id"]
        cached = answers.get(task_id)
        if cached and not (retry_errors and cached["answer"].startswith("AGENT ERROR")):
            continue

        print(f"\n[{i}/{len(questions)}] {item['question'][:100]}")
        try:
            file_path = download_file(task_id, item.get("file_name"))
            answer = agent(item["question"], file_path)
        except Exception as e:
            answer = f"AGENT ERROR: {e}"
        print(f"  -> {answer}")

        # Save after every question so a crash doesn't lose progress
        answers[task_id] = {"question": item["question"], "answer": answer}
        save_answers(answers)

    print(f"\nDone. {len(answers)} answers cached in {ANSWERS_FILE}. Review them, then run with --submit.")


def submit(username, agent_code):
    answers = load_answers()
    if not answers:
        print("No cached answers. Run without --submit first.")
        return
    payload = {
        "username": username,
        "agent_code": agent_code,
        "answers": [
            {"task_id": task_id, "submitted_answer": data["answer"]}
            for task_id, data in answers.items()
        ],
    }
    print(f"Submitting {len(payload['answers'])} answers as '{username}'...")
    response = requests.post(f"{API_URL}/submit", json=payload, timeout=120)
    if not response.ok:
        print(f"Submission failed ({response.status_code}): {response.text[:500]}")
        return
    result = response.json()
    print(f"Score: {result.get('score')}% ({result.get('correct_count')}/{result.get('total_attempted')} correct)")
    print(result.get("message"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Submit cached answers instead of running the agent")
    parser.add_argument("--limit", type=int, help="Only run the first N questions")
    parser.add_argument("--retry-errors", action="store_true", help="Re-run questions whose answer was an error")
    parser.add_argument("--username", default=os.getenv("HF_USERNAME", DEFAULT_USERNAME))
    parser.add_argument("--agent-code", default=DEFAULT_AGENT_CODE)
    args = parser.parse_args()

    if args.submit:
        submit(args.username, args.agent_code)
    else:
        run(args.limit, args.retry_errors)
