
import json
import sys
from pathlib import Path

# Add the src folder to Python's import path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag import answer_question

QUESTIONS_FILE = Path(__file__).parent / "test_questions.json"
RESULTS_FILE = Path(__file__).parent / "test_results.json"


def main():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    results = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        print("\n" + "=" * 70)
        print(f"TEST {question_id}: {question}")

        try:
            answer = answer_question(question)

            print("\nANSWER:")
            print(answer)

            results.append({
                "id": question_id,
                "question": question,
                "answer": str(answer),
                "status": "COMPLETED"
            })

        except Exception as error:
            print(f"\nERROR: {error}")

            results.append({
                "id": question_id,
                "question": question,
                "answer": None,
                "status": "ERROR",
                "error": str(error)
            })

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Finished {len(results)} tests.")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
