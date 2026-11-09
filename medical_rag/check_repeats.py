"""
Repeat-run check for medical_rag (read-only; writes nothing).

Run from the project root (the folder containing src/ and data/):
    python check_repeats.py
    python check_repeats.py "What is the recommended adult dosage of metformin?" 5 "starting dose|850 mg"

Arguments (all optional):
  1. question
  2. number of runs (default 5)
  3. phrases that must appear TOGETHER in one claim line, separated by "|"
     (default "starting dose|850 mg")

A run counts as CLAIM OK only if one claim line contains every phrase, so a
"850 mg" in the dose-increase claim does not count. A run counts as SAFE if
either the claim is present or the starting-dose warning is shown.

It calls answer_question() several times. It does not change any files, but
each run calls Ollama.
"""
import contextlib
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import rag  # noqa: E402

question = sys.argv[1] if len(sys.argv) > 1 else "What is the recommended adult dosage of metformin?"
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
phrases = tuple((sys.argv[3] if len(sys.argv) > 3 else "starting dose|850 mg").split("|"))


def claim_has_all(answer, phrases):
    lines = [ln for ln in answer.splitlines() if ln.startswith("- ")]
    return any(all(p.lower() in ln.lower() for p in phrases) for ln in lines)


print(f"QUESTION: {question}")
print(f"RUNS: {runs}   CLAIM MUST HAVE (one line): {list(phrases)}\n")

claim_ok = 0
warned = 0
safe = 0
noted = 0
for i in range(1, runs + 1):
    # Keep the library's own logging out of the summary.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        answer = rag.answer_question(question)
    has_claim = claim_has_all(answer, phrases)
    has_warning = rag.STARTING_DOSE_WARNING in answer
    has_note = rag.OMITTED_NOTE in answer
    claim_ok += has_claim
    warned += has_warning
    safe += has_claim or has_warning
    noted += has_note
    first_line = answer.splitlines()[0][:90] if answer else "(empty)"
    print(f"run {i}: claim={has_claim!s:5}  warning={has_warning!s:5}  "
          f"note={has_note!s:5}  first line: {first_line}")

print(f"\nSUMMARY: {claim_ok}/{runs} answers had the claim with {list(phrases)} in one line; "
      f"{warned}/{runs} showed the starting-dose warning; "
      f"{safe}/{runs} were safe (claim or warning); {noted}/{runs} showed the 'left out' note.")