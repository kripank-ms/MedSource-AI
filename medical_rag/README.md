# 💬 Medical RAG

**Medical RAG is the Retrieval-Augmented Generation component of [MedSource AI](../README.md).**

It answers medical questions by retrieving relevant information from reference sources and providing that context to a local language model.

The system retrieves relevant passages, asks the local language model to generate claims grounded in those passages, verifies the claims against their cited sources, and displays the sources used for the answer.

> **Educational information only. This system is not medical advice and does not replace a doctor or pharmacist.**

---

## 🧠 How It Works

```text
User Question
      │
      ▼
Intent Detection
      │
      ▼
Question Embedding
      │
      ▼
ChromaDB Vector Search
      │
      ▼
Candidate Passages
      │
      ▼
CrossEncoder Reranking
      │
      ▼
Best Relevant Passages
      │
      ▼
Ollama / Qwen3 8B
      │
      ▼
Generated Claims
      │
      ▼
Claim Verification
      │
      ▼
Grounded Response
      │
      ▼
Cited Sources
```

---

## 🔎 Retrieval Pipeline

1. Detect the question's intent, such as:
   - `dosage`
   - `indication`
   - `adverse_reactions`

2. Embed the user's question.

3. Search ChromaDB for candidate passages.

4. Rerank candidates using a CrossEncoder model.

5. Apply intent-specific section boosting.

6. Remove inappropriate dosing sections when the population does not match.

7. Keep the best relevant passages.

8. Send the retrieved context to the local LLM.

9. Generate claims based only on the retrieved passages.

10. Verify each claim against its cited source.

11. Remove unsupported claims.

12. Retry once when a numerical claim fails verification.

13. Return the grounded response with numbered sources.

---

## 🧩 RAG Technology Stack

| Component | Technology |
|---|---|
| Vector database | ChromaDB |
| Embedding model | all-MiniLM-L6-v2 |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| LLM runtime | Ollama |
| Language model | Qwen3 8B |
| Backend integration | FastAPI |
| Source documents | DailyMed / MedlinePlus |
| Programming language | Python |

### MongoDB vs ChromaDB

MedSource AI uses two different databases for different purposes:

```text
MongoDB
   │
   └── User/account information


ChromaDB
   │
   └── RAG document chunks + embeddings
```

MongoDB is **not** used as the RAG vector database.

---

## 📁 Project Structure

```text
medical_rag/
├── src/
│   ├── rag.py
│   ├── chunker.py
│   ├── parse_dailymed.py
│   ├── parse_medlineplus.py
│   ├── parse_genetics.py
│   ├── dailymed_chunks.py
│   ├── create_genetics_chunks.py
│   ├── vector_store.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_groundedness.py
│   └── loaders/
│       ├── JSON loaders
│       ├── TXT loaders
│       └── XML loaders
│
├── tests/
│   ├── check_quality.py
│   ├── test_verification.py
│   ├── run_tests.py
│   ├── test_questions.json
│   └── test_results.json
│
├── data/
│   ├── chroma_db/
│   ├── processed/
│   └── evaluation files
│
├── rag_api.py
├── check_repeats.py
├── requirements-lock-utf8.txt
└── README.md
```

> The local `data/` directory can contain generated indexes and source data and should not be uploaded when it is excluded by the project's `.gitignore`.

---

## ⚙️ Setup

Install Python 3 and create a virtual environment.

Install the RAG dependencies:

```bash
pip install -r requirements-lock-utf8.txt
```

Install and start Ollama, then download the model:

```bash
ollama pull qwen3:8b
```

The first run may download the Hugging Face models:

```text
all-MiniLM-L6-v2
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## ▶️ Running the RAG System

Run commands from the appropriate project directory so relative paths resolve correctly.

For the standalone RAG module:

```bash
python src/rag.py
```

---

## 🧪 Testing

Offline verification tests:

```bash
python -m unittest tests/test_verification.py -v
```

Run the test questions:

```bash
python tests/run_tests.py
```

Run quality checks:

```bash
python tests/check_quality.py
```

Check response consistency:

```bash
python check_repeats.py
```

---

## 🔐 Environment Variables

Optional configuration can be supplied through environment variables.

Examples:

```env
OLLAMA_MODEL=qwen3:8b
MIN_RERANKER_SCORE=-6.0
EMERGENCY_NUMBER=your_emergency_number
```

Never commit secrets or private configuration values to GitHub.

---

## ⚠️ Known Limitations

- Only the currently defined test questions have been evaluated.
- Some source datasets and documents are not included in the repository.
- The LLM verifier may not reject every unsupported claim.
- Numerical verification provides an additional safety check.
- The local model can occasionally generate incorrect numerical formatting.
- Emergency detection uses keyword matching and can produce false alarms.
- Medical correctness has not been established through clinical validation.
- Some data scripts may require path adjustments when executed from different directories.

---

## 🛡️ Medical Disclaimer

This RAG system is an **educational and research project**.

Its responses:

- may contain errors;
- are not medical diagnoses;
- are not a substitute for professional medical advice;
- should not be used to make treatment or medication decisions.

Always verify medical information with an appropriate healthcare professional and the original medical source.

---

## 🔗 Part of MedSource AI

The RAG system is integrated into the larger **MedSource AI** FastAPI application.

For the complete project documentation, see:

```text
../README.md
```

The complete MedSource AI project combines:

```text
Machine Learning
       │
       ├── Disease Prediction
       │
       ├── Medical Image Analysis
       │
       └── RAG Medical Chatbot
                    │
                    ▼
              Ollama + Qwen3
```
