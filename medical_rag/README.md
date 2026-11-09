# Medical RAG

A learning project that answers medical questions using text retrieved from
reference sources (DailyMed drug labels and MedlinePlus health topics). It
retrieves relevant passages, asks a local language model to write claims
grounded in those passages, checks each claim against its cited source, and
shows the sources.

**This is educational information only. It is not medical advice and does not
replace a doctor or pharmacist.** Dosing and safety details come from the drug
label text in the index. Always check the label or ask a healthcare professional.

## How it works

1. Detect the question's intent (for example `dosage`, `indication`, `adverse_reactions`).
2. Embed the question and search ChromaDB for candidate passages.
3. Rerank the candidates with a CrossEncoder model.
4. Boost preferred label sections for the intent, remove dosing sections for the
   wrong population (adult or pediatric), and keep the best 3 passages.
5. Ask the local model (Ollama) for claims, using only the passages.
6. Verify each claim: numbers and units must appear in the cited source, and an
   LLM check must say the source supports the claim.
7. Remove failed claims, retry once when a number was wrong, and show a warning
   or note when information was left out.
8. Print the answer with numbered sources and URLs.

Main code: `src/rag.py` (function `answer_question()` is the entry point).

## Project layout

```
medical_rag/
├── src/
│   ├── rag.py                 # question answering pipeline (main file)
│   ├── intent.py              # older intent module, see Known issues
│   ├── chunker.py             # splits documents into chunks
│   ├── parse_dailymed.py      # downloads/parses DailyMed labels
│   ├── parse_medlineplus.py   # parses MedlinePlus XML topics
│   ├── parse_genetics.py      # parses genetics content
│   ├── dailymed_chunks.py     # builds DailyMed chunks
│   ├── create_genetics_chunks.py
│   ├── vector_store.py        # writes chunks into ChromaDB
│   ├── embeddings.py          # embedding helper
│   ├── evaluate_retrieval.py  # retrieval evaluation
│   ├── evaluate_groundedness.py
│   └── loaders/               # JSON, TXT and XML loaders
├── tests/
│   ├── check_quality.py       # quality checks for the 5 metformin questions (needs models)
│   ├── test_verification.py   # offline unit tests (no models needed)
│   ├── run_tests.py           # runs test questions and writes test_results.json
│   ├── test_questions.json
│   └── test_results.json
├── data/
│   ├── chroma_db/             # the ChromaDB vector index
│   ├── processed/             # chunk and document JSON files
│   └── evaluation_questions.json, retrieval_evaluation_labeled_ready.csv
├── check_repeats.py           # runs one question several times to measure consistency
├── requirements.txt
└── README.md
```

Raw source files (for example the MedlinePlus XML) are not in the repository.
The scripts expect them under `data/raw/`.

## Setup

1. Install Python 3 (your cache files show Python 3.13; any recent Python 3 should work).
2. Install Ollama and download the model:
   ```
   ollama pull qwen3:8b
   ```
3. Create a virtual environment and install the packages:
   ```
   python -m venv env
   env\Scripts\activate
   pip install -r requirements.txt
   ```

No API keys are needed. Optional settings are read from environment variables
(all have defaults): `OLLAMA_MODEL`, `MIN_RERANKER_SCORE`, `EMERGENCY_NUMBER`.

The first run downloads two Hugging Face models (`all-MiniLM-L6-v2` and
`cross-encoder/ms-marco-MiniLM-L-6-v2`). The log warning about unauthenticated
requests is harmless. Setting `HF_TOKEN` removes it and speeds up downloads.

## Running

Run all commands from the project root (the folder that contains `src/` and `data/`),
because the database path is relative.

Ask a question interactively:
```
python src/rag.py
```

Tests:
```
python -m unittest tests/test_verification.py -v   # offline, no models needed
python tests/run_tests.py                          # runs test questions, needs models
python tests/check_quality.py                      # checks answers against label text, needs models
python check_repeats.py                            # repeats one question and reports consistency
```

## What the tests mean

- **Offline unit tests** check the logic: number and unit matching, the retry,
  the warning, and the dosage rules. They use stand-in models.
- **`check_quality.py`** checks each answer against the label text in the index:
  the right section is cited, and required facts appear in one claim. It does not
  judge medical correctness.
- **`run_tests.py`** only shows that the code ran. A `COMPLETED` status does not
  mean the answer is accurate.

## Known limitations

- Only the five metformin questions are tested. MedlinePlus and genetics content
  have not been tested.
- The LLM verifier has not yet rejected a claim in testing. The number-and-unit
  check is the stronger check.
- The local model sometimes writes numbers wrongly (for example `85,0 mg` for
  `850 mg`). The retry and the starting-dose warning are the safety net.
- Emergency detection uses keyword matching, so it can give false alarms
  (for example "are seizures a side effect?").
- The answers have not been reviewed by a medical professional.
- Some data scripts use paths such as `medical_rag/data/...`, which only work if the
  folder is named `medical_rag` and run from its parent folder. These should be made relative.

## Known issues to fix

- `tests/test_citations.py` imports `validate_claim_support`, which no longer exists in `rag.py`.
- `src/intent.py` duplicates the intent logic in `rag.py`. Only `tests/test_intents.py` uses it.
