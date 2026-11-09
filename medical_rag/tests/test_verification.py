"""
Offline tests for rag.py logic: number/unit checks, the retry and note
behaviour, and the dosage no-evidence rule.

They do NOT need Ollama, the embedding model, or the Chroma database:
those are replaced with small stand-ins before rag.py is loaded.

Run from the project root:
    python -m unittest tests/test_verification.py -v
"""
import importlib.util
import json
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeCollection:
    def count(self):
        return 0

    def get(self, include=None):
        return {"metadatas": []}


class _FakeClient:
    def __init__(self, path=None):
        pass

    def get_collection(self, name):
        return _FakeCollection()


def _install_stubs():
    stubs = {
        "chromadb": {"PersistentClient": _FakeClient},
        "dotenv": {"load_dotenv": lambda: None},
        "ollama": {"chat": lambda **kwargs: None},
        "sentence_transformers": {
            "SentenceTransformer": lambda *a, **k: None,
            "CrossEncoder": lambda *a, **k: None,
        },
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module


_install_stubs()
_spec = importlib.util.spec_from_file_location("rag", os.path.join(ROOT, "src", "rag.py"))
rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag)

PASSAGE = (
    "Recommended starting dose for adults is 500 mg (5 mL) orally twice a day "
    "or 850 mg (8.5 mL) once a day. The maximum dose is 2,550 mg (25.5 mL) per day. "
    "Diarrhea was reported in 53% of patients."
)
GOOD = {"text": "The recommended starting dose is 500 mg (5 mL) twice a day or 850 mg (8.5 mL) once a day.",
        "source_ids": ["S1"]}
BAD_NUMBER = {"text": "The starting dose is 500 mg (5 mL) twice a day or 85,0 mg (8.5 mL) once a day.",
              "source_ids": ["S1"]}
MAX = {"text": "The maximum dose is 2,550 mg (25.5 mL) per day.", "source_ids": ["S1"]}
UNSUPPORTED = {"text": "Metformin treats high blood pressure.", "source_ids": ["S1"]}


class TestNumberAndUnitChecks(unittest.TestCase):
    def test_matching_numbers_and_units_pass(self):
        self.assertTrue(rag.numbers_supported("The dose is 500 mg (5 mL).", PASSAGE))

    def test_wrong_unit_is_rejected(self):
        # Source says 500 mg; the claim says 500 mL.
        self.assertFalse(rag.numbers_supported("The dose is 500 mL.", PASSAGE))

    def test_wrong_number_is_rejected(self):
        self.assertFalse(rag.numbers_supported("The dose is 85,0 mg.", PASSAGE))

    def test_percent_is_checked(self):
        self.assertTrue(rag.numbers_supported("Diarrhea occurred in 53% of patients.", PASSAGE))
        self.assertFalse(rag.numbers_supported("Diarrhea occurred in 35% of patients.", PASSAGE))

    def test_thousands_comma_is_handled(self):
        self.assertTrue(rag.numbers_supported("The maximum is 2550 mg per day.", PASSAGE))

    def test_unlisted_words_after_numbers_are_ignored(self):
        # "times" is not a checked unit, so a paraphrase does not fail.
        self.assertTrue(rag.numbers_supported("Given 2 times a day.", "given 2 times a day"))


class TestGenerationAndNote(unittest.TestCase):
    def run_answer(self, generations, verdicts=None,
                   question="What is the recommended adult dosage of metformin?",
                   passage=PASSAGE):
        calls = {"generate": 0, "prompts": []}

        def fake_call_llm(messages, schema=None):
            if schema is rag.CLAIM_SCHEMA:
                index = min(calls["generate"], len(generations) - 1)
                calls["generate"] += 1
                calls["prompts"].append(messages[1]["content"])
                return json.dumps({"claims": generations[index]})
            text = messages[1]["content"]
            for claim_text, verdict in (verdicts or {}).items():
                if claim_text in text:
                    return json.dumps({"verdict": verdict, "reason": "test"})
            return json.dumps({"verdict": "SUPPORTED", "reason": "test"})

        rag.call_llm = fake_call_llm
        rag.retrieve_documents = lambda q, top_k=3: [{
            "document": passage, "source": "DailyMed", "title": "Metformin Hydrochloride",
            "section": "2.1 Adult Dosage", "url": "https://example.test/label"}]
        answer = rag.answer_question(question)
        self.prompts = calls["prompts"]
        return answer, calls["generate"]

    def test_clean_run_has_no_note_and_one_attempt(self):
        answer, attempts = self.run_answer([[GOOD, MAX]])
        self.assertNotIn(rag.OMITTED_NOTE, answer)
        self.assertEqual(attempts, 1)
        self.assertIn(rag.DOSAGE_DISCLAIMER, answer)

    def test_number_error_is_retried_once_and_fixed(self):
        answer, attempts = self.run_answer([[BAD_NUMBER, MAX], [GOOD, MAX]])
        self.assertEqual(attempts, 2)
        self.assertIn("850 mg (8.5 mL)", answer)
        self.assertNotIn("85,0", answer)
        self.assertNotIn(rag.OMITTED_NOTE, answer)

    def test_retry_with_same_claim_count_is_kept(self):
        # Attempt 1 keeps only the maximum-dose claim; the retry keeps the
        # starting dose but drops a different claim. Same count, so the retry wins.
        answer, attempts = self.run_answer(
            [[BAD_NUMBER, MAX], [GOOD, UNSUPPORTED]],
            verdicts={UNSUPPORTED["text"]: "INSUFFICIENT_EVIDENCE"})
        self.assertEqual(attempts, 2)
        self.assertIn("850 mg (8.5 mL)", answer)
        self.assertNotIn(rag.STARTING_DOSE_WARNING, answer)

    def test_persistent_number_error_shows_note(self):
        answer, attempts = self.run_answer([[BAD_NUMBER, MAX], [BAD_NUMBER, MAX]])
        self.assertEqual(attempts, 2)
        self.assertIn(rag.OMITTED_NOTE, answer)
        self.assertNotIn("85,0", answer)
        self.assertIn("2,550 mg", answer)

    def test_other_rejection_is_not_retried_but_noted(self):
        answer, attempts = self.run_answer(
            [[GOOD, UNSUPPORTED]], verdicts={UNSUPPORTED["text"]: "INSUFFICIENT_EVIDENCE"})
        self.assertEqual(attempts, 1)
        self.assertIn(rag.OMITTED_NOTE, answer)
        self.assertNotIn("high blood pressure", answer)

    def test_nothing_verified_returns_no_evidence(self):
        answer, _ = self.run_answer([[BAD_NUMBER]])
        self.assertEqual(answer, rag.NO_EVIDENCE)

    def test_retry_gets_feedback_naming_the_rejected_claim(self):
        self.run_answer([[BAD_NUMBER, MAX], [GOOD, MAX]])
        self.assertNotIn("Rejected claim", self.prompts[0])
        self.assertIn("Rejected claim", self.prompts[1])
        self.assertIn(BAD_NUMBER["text"], self.prompts[1])
        self.assertIn("Source text for S1", self.prompts[1])

    def test_no_feedback_when_nothing_was_rejected_for_numbers(self):
        self.run_answer([[GOOD, MAX]])
        self.assertNotIn("Rejected claim", self.prompts[0])

    def test_starting_dose_warning_when_claim_is_lost(self):
        # The model keeps only the maximum-dose claim; the source has a starting dose.
        answer, _ = self.run_answer([[MAX]])
        self.assertIn(rag.STARTING_DOSE_WARNING, answer)
        self.assertLess(answer.index(rag.STARTING_DOSE_WARNING), answer.index("2,550 mg"))

    def test_no_starting_dose_warning_when_claim_is_kept(self):
        answer, _ = self.run_answer([[GOOD, MAX]])
        self.assertNotIn(rag.STARTING_DOSE_WARNING, answer)

    def test_no_starting_dose_warning_for_non_dosage_question(self):
        answer, _ = self.run_answer(
            [[MAX]], question="What is metformin used for?")
        self.assertNotIn(rag.STARTING_DOSE_WARNING, answer)

    def test_no_starting_dose_warning_when_source_has_no_starting_dose(self):
        passage = "The maximum dose is 2,550 mg (25.5 mL) per day."
        answer, _ = self.run_answer([[MAX]], passage=passage)
        self.assertNotIn(rag.STARTING_DOSE_WARNING, answer)


LIST_PASSAGE = (
    "Additionally, the following adverse reactions were reported in ≥ 1% to ≤ 5% of "
    "patients treated with metformin HCl tablets and were more commonly reported than "
    "placebo: abnormal stools, hypoglycemia, myalgia, lightheaded, dyspnea, nail disorder, "
    "rash, sweating increased, taste disorder, chest discomfort, chills, flu syndrome, "
    "flushing, palpitation."
)
LIST_CLAIM = {"text": "Abnormal stools were reported in ≥ 1% to ≤ 5% of patients.", "source_ids": ["S1"]}
RASH_CLAIM = {"text": "Rash was reported in ≥ 1% to ≤ 5% of patients.", "source_ids": ["S1"]}
ALL_ITEMS_CLAIM = {"text": ("Abnormal stools, hypoglycemia, myalgia, lightheaded, dyspnea, nail disorder, "
                            "rash, sweating increased, taste disorder, chest discomfort, chills, "
                            "flu syndrome, flushing, and palpitation were reported in ≥ 1% to ≤ 5% "
                            "of patients."), "source_ids": ["S1"]}


class TestListCompleteness(unittest.TestCase):
    def sources(self, passage=LIST_PASSAGE):
        return {"S1": {"content": passage, "title": "t", "section": "6.1", "source": "DailyMed", "url": ""}}

    def test_parses_the_real_label_list(self):
        items = rag.list_items_from_passages(self.sources())
        self.assertEqual(len(items), 14)
        self.assertIn("abnormal stools", items)
        self.assertIn("palpitation", items)
        self.assertNotIn("and palpitation", items)

    def test_missing_item_is_found(self):
        missing = rag.missing_list_items([LIST_CLAIM], self.sources())
        self.assertIn("hypoglycemia", missing)
        self.assertNotIn("abnormal stools", missing)

    def test_all_items_in_one_claim_means_nothing_missing(self):
        self.assertEqual(rag.missing_list_items([ALL_ITEMS_CLAIM], self.sources()), [])

    def test_word_order_does_not_matter(self):
        claim = {"text": "Increased sweating, hypoglycemia, myalgia, lightheaded, dyspnea, "
                         "nail disorder, rash, taste disorder, chest discomfort, chills, "
                         "flu syndrome, flushing, palpitation and abnormal stools.",
                 "source_ids": ["S1"]}
        self.assertEqual(rag.missing_list_items([claim], self.sources()), [])

    def run_answer(self, generations, question, verdicts=None):
        calls = {"generate": 0, "prompts": []}

        def fake_call_llm(messages, schema=None):
            if schema is rag.CLAIM_SCHEMA:
                index = min(calls["generate"], len(generations) - 1)
                calls["generate"] += 1
                calls["prompts"].append(messages[1]["content"])
                return json.dumps({"claims": generations[index]})
            return json.dumps({"verdict": "SUPPORTED", "reason": "test"})

        rag.call_llm = fake_call_llm
        rag.retrieve_documents = lambda q, top_k=3: [{
            "document": LIST_PASSAGE, "source": "DailyMed", "title": "Metformin Hydrochloride",
            "section": "6.1 Clinical Studies Experience", "url": "https://example.test/label"}]
        answer = rag.answer_question(question)
        self.prompts = calls["prompts"]
        return answer, calls["generate"]

    ADR = "What adverse reactions are listed for metformin?"

    def test_retry_fills_in_the_missing_items(self):
        # First attempt covers one item; the retry names all of them.
        answer, attempts = self.run_answer([[LIST_CLAIM], [ALL_ITEMS_CLAIM]], self.ADR)
        self.assertEqual(attempts, 2)
        self.assertIn("rash", answer.lower())
        self.assertNotIn("are listed in the drug label but are not", answer)

    def test_retry_feedback_names_the_missing_items(self):
        self.run_answer([[LIST_CLAIM], [ALL_ITEMS_CLAIM]], self.ADR)
        self.assertIn("left these reactions out", self.prompts[1])
        self.assertIn("rash", self.prompts[1])

    def test_note_shown_when_still_missing_after_retry(self):
        answer, attempts = self.run_answer([[LIST_CLAIM], [LIST_CLAIM]], self.ADR)
        self.assertEqual(attempts, 2)
        self.assertIn("are listed in the drug label but are not in the answer", answer)
        self.assertIn("rash", answer)
        self.assertIn("hypoglycemia", answer)

    def test_no_list_check_for_other_questions(self):
        answer, attempts = self.run_answer([[LIST_CLAIM]], "What is metformin used for?")
        self.assertEqual(attempts, 1)
        self.assertNotIn("are listed in the drug label but are not", answer)


class TestDosageNoEvidence(unittest.TestCase):
    """
    Uses the real candidate scores from the check_retrieval.py run on the
    adult dosage question. Metadata for the other chunks is generic.
    """
    SECTIONS = [
        ("2.2 Pediatric Dosage", 7.192), ("2.1 Adult Dosage", 6.388),
        ("12.3 Pharmacokinetics", 4.157), ("8.5 Geriatric Use", 1.053),
        ("12.3 Pharmacokinetics", 4.277), ("16 HOW SUPPLIED/STORAGE AND HANDLING", 1.175),
        ("12.3 Pharmacokinetics", -0.471), ("12.3 Pharmacokinetics", -0.989),
        ("8.4 Pediatric Use", 4.519), ("12.3 Pharmacokinetics", -1.145),
        ("4 CONTRAINDICATIONS", -0.830), ("10 OVERDOSAGE", -1.295),
        ("2.3 Recommendations for Use in Renal Impairment", -0.412),
        ("8.6 Renal Impairment", 0.354), ("WARNING: LACTIC ACIDOSIS", -3.384),
    ]

    def retrieve(self, sections):
        class Collection:
            def count(inner):
                return len(sections)

            def query(inner, **kwargs):
                metas = [{"title": "Metformin Hydrochloride", "source": "DailyMed",
                          "section": s, "document_id": "e087", "url": "u",
                          "chunk_index": i} for i, (s, _) in enumerate(sections)]
                return {"documents": [[f"text of {s}" for s, _ in sections]],
                        "metadatas": [metas],
                        "distances": [[0.5] * len(sections)]}

        class Encoder:
            def encode(inner, text):
                return types.SimpleNamespace(tolist=lambda: [0.0])

        class Reranker:
            def predict(inner, pairs):
                return [score for _, score in sections]

        rag.collection = Collection()
        rag.embedding_model = Encoder()
        rag.reranker = Reranker()
        rag.DRUG_KEYS = {"metformin"}
        return rag.retrieve_documents("What is the recommended adult dosage of metformin?")

    def test_adult_dosage_selected_and_pediatric_excluded(self):
        selected = self.retrieve(self.SECTIONS)
        self.assertEqual([c["section"] for c in selected], ["2.1 Adult Dosage"])

    def test_no_adult_section_returns_no_evidence(self):
        without_adult = [s for s in self.SECTIONS if s[0] != "2.1 Adult Dosage"]
        self.assertEqual(self.retrieve(without_adult), [])


if __name__ == "__main__":
    unittest.main()