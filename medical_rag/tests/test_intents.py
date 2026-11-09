import unittest
from src.intent import (
    detect_medicine_intent,
    get_preferred_sections,
)

class TestIntentDetection(unittest.TestCase):

    def test_dosage(self):
        self.assertEqual(
            detect_medicine_intent(
                "What is the recommended adult dosage of metformin?"
            ),
            "dosage",
        )

    def test_contraindication(self):
        self.assertEqual(
            detect_medicine_intent(
                "What are the contraindications of metformin?"
            ),
            "contraindication",
        )

    def test_indication(self):
        self.assertEqual(
            detect_medicine_intent("What is metformin used for?"),
            "indication",
        )

    def test_adverse_reactions(self):
        self.assertEqual(
            detect_medicine_intent(
                "What are the side effects of metformin?"
            ),
            "adverse_reactions",
        )

    def test_drug_interactions(self):
        self.assertEqual(
            detect_medicine_intent(
                "Does metformin interact with other medicines?"
            ),
            "drug_interactions",
        )

    def test_mechanism(self):
        self.assertEqual(
            detect_medicine_intent("How does metformin work?"),
            "mechanism",
        )

    def test_pregnancy(self):
        self.assertEqual(
            detect_medicine_intent(
                "Can metformin be used during pregnancy?"
            ),
            "pregnancy",
        )

    def test_renal(self):
        self.assertEqual(
            detect_medicine_intent(
                "Can metformin be used with kidney problems?"
            ),
            "renal",
        )

    def test_overdose(self):
        self.assertEqual(
            detect_medicine_intent(
                "What happens in a metformin overdose?"
            ),
            "overdose",
        )

    def test_dosage_sections(self):
        self.assertEqual(
            get_preferred_sections("dosage"),
            [
                "dosage and administration",
                "adult dosage",
                "pediatric dosage",
            ],
        )

    def test_adult_dosage_boost(self):
        from src.rag import section_boost

        self.assertEqual(
            section_boost(
                "What is the recommended adult dosage of metformin?",
                "2.1 Adult Dosage",
            ),
            15.0,
        )

    def test_pediatric_dosage_not_boosted_for_adult_question(self):
        from src.rag import section_boost

        self.assertEqual(
            section_boost(
                "What is the recommended adult dosage of metformin?",
                "2.2 Pediatric Dosage",
            ),
            0.0,
        )

    def test_pediatric_dosage_boost(self):
        from src.rag import section_boost

        self.assertEqual(
            section_boost(
                "What is the recommended pediatric dosage of metformin?",
                "2.2 Pediatric Dosage",
            ),
            15.0,
        )



if __name__ == "__main__":
    unittest.main()