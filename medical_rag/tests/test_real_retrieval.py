import unittest
from src import rag


class TestRealDocumentRetrieval(unittest.TestCase):

    def test_metformin_dosage_retrieval(self):
        question = "What is the recommended adult dosage of metformin?"

        results = rag.retrieve_documents(question, top_k=3)

        # The database should return relevant documents.
        self.assertGreater(
            len(results),
            0,
            "No documents were retrieved from the real database."
        )

        # Check the returned document metadata.
        sections = [
            result.get("section", "").lower()
            for result in results
        ]

        print("\nReal retrieved sections:", sections)

        self.assertTrue(
            any("dosage" in section for section in sections),
            f"No dosage section was retrieved. Sections: {sections}"
        )


if __name__ == "__main__":
    unittest.main()