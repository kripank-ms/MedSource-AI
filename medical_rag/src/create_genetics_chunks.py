import json
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")


INPUT_FILE = "medical_rag/data/processed/medlineplus_genetics_documents.json"
OUTPUT_FILE = "medical_rag/data/processed/medlineplus_genetics_chunks.json"

MAX_WORDS = 300
OVERLAP_SENTENCES = 1


def sentence_chunks(
    text,
    max_words=300,
    overlap_sentences=1
):
    sentences = sent_tokenize(text)

    chunks = []

    current_chunk = []
    current_word_count = 0

    for sentence in sentences:

        sentence_word_count = len(
            sentence.split()
        )

        if (
            current_word_count
            + sentence_word_count
            <= max_words
        ):

            current_chunk.append(sentence)

            current_word_count += (
                sentence_word_count
            )

        else:

            # Save current chunk
            if current_chunk:
                chunks.append(
                    " ".join(current_chunk)
                )

            # Keep overlap
            current_chunk = (
                current_chunk[
                    -overlap_sentences:
                ]
            )

            current_word_count = sum(
                len(s.split())
                for s in current_chunk
            )

            # Start adding new sentence
            current_chunk.append(sentence)

            current_word_count += (
                sentence_word_count
            )

    # Save final chunk
    if current_chunk:
        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


def chunk_document(document):

    text = document["text"]
    metadata = document["metadata"]

    chunks = sentence_chunks(
        text,
        max_words=MAX_WORDS,
        overlap_sentences=OVERLAP_SENTENCES
    )

    results = []

    document_id = metadata["id"]

    for index, chunk in enumerate(chunks):

        if not chunk.strip():
            continue

        chunk_metadata = metadata.copy()

        chunk_metadata["chunk_index"] = index

        chunk_id = (
            f"genetics_"
            f"{document_id}_"
            f"{index:04d}"
        )

        results.append({
            "chunk_id": chunk_id,
            "text": chunk,
            "metadata": chunk_metadata
        })

    return results


# ---------------------------------------
# Load Genetics documents
# ---------------------------------------

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    documents = json.load(file)


print(
    "Genetics documents loaded:",
    len(documents)
)


# ---------------------------------------
# Create chunks
# ---------------------------------------

all_chunks = []

for document in documents:

    chunks = chunk_document(
        document
    )

    all_chunks.extend(chunks)


# ---------------------------------------
# Save chunks
# ---------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------
# Validation
# ---------------------------------------

word_counts = [
    len(chunk["text"].split())
    for chunk in all_chunks
]

empty_chunks = sum(
    1
    for chunk in all_chunks
    if not chunk["text"].strip()
)

unique_ids = len(
    set(
        chunk["chunk_id"]
        for chunk in all_chunks
    )
)


print()
print("==============================")
print("GENETICS CHUNKING COMPLETE")
print("==============================")

print(
    "Documents:",
    len(documents)
)

print(
    "Chunks:",
    len(all_chunks)
)

print(
    "Empty chunks:",
    empty_chunks
)

print(
    "Unique chunk IDs:",
    unique_ids
)

print(
    "Shortest chunk:",
    min(word_counts),
    "words"
)

print(
    "Longest chunk:",
    max(word_counts),
    "words"
)

print(
    "Average chunk:",
    round(
        sum(word_counts) / len(word_counts),
        2
    ),
    "words"
)

print(
    "Output:",
    OUTPUT_FILE
)


print()
print("First chunk:")
print("------------------------------")

print(
    json.dumps(
        all_chunks[0],
        indent=2,
        ensure_ascii=False
    )
)