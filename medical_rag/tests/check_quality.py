
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag  # noqa: E402

CONTRA = "4 CONTRAINDICATIONS"
CHECKS = [
    # (question, allowed cited sections, required phrase groups, label)
    # A phrase group is a tuple: ALL its phrases must appear in the SAME claim
    # line. A plain string is checked against the whole answer.
    ("What is metformin used for?",
     {"1 INDICATIONS AND USAGE"}, ["type 2 diabetes"],
     "indication cites the indication section"),
    ("What is the recommended adult dosage of metformin?",
     {"2.1 Adult Dosage"}, [("starting dose", "850 mg")],
     "adult dose cites adult section and keeps the starting-dose claim"),
    ("What warnings and precautions are listed for metformin?",
     {"WARNING: LACTIC ACIDOSIS"}, ["lactic acidosis"],
     "warnings cite the lactic acidosis warning, not other sections"),
    ("What are the contraindications for metformin?",
     {CONTRA, "8.6 Renal Impairment"}, ["hypersensitivity"],
     "contraindications cite section 4 (renal impairment allowed as support)"),
    ("What adverse reactions are listed for metformin?",
     {"6.1 Clinical Studies Experience", "6.2 Postmarketing Experience"},
     ["Cholestatic",
      # Every item in the "≥ 1% to ≤ 5%" group must survive grouping.
      "Abnormal stools", "Hypoglycemia", "Myalgia", "Lightheaded", "Dyspnea",
      "Nail disorder", "Rash", "Sweating increased", "Taste disorder",
      "Chest discomfort", "Chills", "Flu syndrome", "Flushing", "Palpitation"],
     "adverse reactions keep the liver-injury safety claim"),
]


def missing_phrases(answer, claims, required):
    """Return the required phrase groups that are not satisfied."""
    missing = []
    for item in required:
        if isinstance(item, tuple):
            ok = any(all(p.lower() in line.lower() for p in item) for line in claims)
            if not ok:
                missing.append(" + ".join(item) + " (in one claim)")
        elif item.lower() not in answer.lower():
            missing.append(item)
    return missing


def parse_sources(answer):
    """Map source IDs (S1, S2) to section names from the Sources block."""
    return dict(re.findall(
        r"^\[(S\d+)\] .*? - (.+?)(?: \(http|$)", answer, flags=re.M))


def claim_lines(answer):
    return [ln for ln in answer.splitlines() if ln.startswith("- ")]


failures = 0
for question, allowed, required, label in CHECKS:
    answer = rag.answer_question(question)
    sources = parse_sources(answer)
    cited = set(sources.values())
    claims = claim_lines(answer)
    print(f"\n{question}")
    print(f"  claims: {len(claims)}   cited sections: {sorted(cited) or 'none'}")

    problems = []
    if not claims:
        problems.append("no claims returned")
    if not cited <= allowed:
        problems.append(f"cited sections outside {sorted(allowed)}")
    for phrase in missing_phrases(answer, claims, required):
        problems.append(f"missing '{phrase}'")
    for line in claims:
        for sid in re.findall(r"\[(S\d+)\]", line):
            if sid not in sources:
                problems.append(f"{sid} cited but not in Sources")

    if problems:
        failures += 1
        print(f"  FAIL  {label}: " + "; ".join(problems))
    else:
        print(f"  PASS  {label}")

print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} questions passed all checks.")
sys.exit(1 if failures else 0)