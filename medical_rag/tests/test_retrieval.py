
import unittest
from unittest.mock import patch, MagicMock

from src import rag


class TestDocumentRetrieval(unittest.TestCase):

    def setUp(self):
        self.question = "What is the recommended adult dosage of metformin?"

        self.fake_documents = [
            "Metformin adult dosage information.",
            "Metformin pediatric dosage information.",
            "Unrelated information about another medicine.",
        ]

        self.fake_metadatas = [
            {
                "title": "Metformin",
                "source": "DailyMed",
                "section": "2.1 Adult Dosage",
                "document_id": "doc1",
                "url": "https://example.com/metformin",
                "chunk_index": 0,
            },
            {
                "title": "Metformin",
                "source": "DailyMed",
                "section": "Pediatric Dosage",
                "document_id": "doc1",
                "url": "https://example.com/metformin",
                "chunk_index": 1,
            },
            {
                "title": "Another Medicine",
                "source": "DailyMed",
                "section": "Dosage",
                "document_id": "doc2",
                "url": "https://example.com/other",
                "chunk_index": 0,
            },
        ]

    @patch("src.rag.deduplicate_documents", side_effect=lambda docs, max_chunks_per_document: docs)
    @patch("src.rag.medicine_name_matches")
    @patch("src.rag.section_boost")
    def test_returns_relevant_documents(
        self, mock_boost, mock_medicine_match, mock_deduplicate
    ):
        mock_medicine_match.side_effect = [
            True, True, False
        ]
        mock_boost.return_value = 10.0

        rag.collection = MagicMock()
        rag.collection.query.return_value = {
            "documents": [self.fake_documents],
            "metadatas": [self.fake_metadatas],
            "distances": [[0.1, 0.2, 0.3]],
        }

        rag.embedding_model = MagicMock()
        rag.embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2]

        rag.reranker = MagicMock()
        rag.reranker.predict.return_value = [8.0, 7.0, 9.0]

        with patch.object(rag, "MIN_RERANKER_SCORE", 0.0):
            results = rag.retrieve_documents(self.question, top_k=3)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["section"], "2.1 Adult Dosage")
        self.assertEqual(results[1]["section"], "Pediatric Dosage")
        self.assertGreaterEqual(
            results[0]["final_score"], results[1]["final_score"]
        )

    @patch("src.rag.deduplicate_documents", side_effect=lambda docs, max_chunks_per_document: docs)
    @patch("src.rag.medicine_name_matches", return_value=True)
    @patch("src.rag.section_boost", return_value=0.0)
    def test_rejects_low_reranker_scores(
        self, mock_boost, mock_medicine_match, mock_deduplicate
    ):
        rag.collection = MagicMock()
        rag.collection.query.return_value = {
            "documents": [["Irrelevant text"]],
            "metadatas": [[self.fake_metadatas[0]]],
            "distances": [[0.9]],
        }

        rag.embedding_model = MagicMock()
        rag.embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2]

        rag.reranker = MagicMock()
        rag.reranker.predict.return_value = [-10.0]

        with patch.object(rag, "MIN_RERANKER_SCORE", 0.0):
            results = rag.retrieve_documents(self.question, top_k=3)

        self.assertEqual(results, [])

    def test_returns_empty_when_database_has_no_documents(self):
        rag.collection = MagicMock()
        rag.collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        rag.embedding_model = MagicMock()
        rag.embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2]

        results = rag.retrieve_documents(self.question)

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
