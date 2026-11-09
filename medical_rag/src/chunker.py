import json
import nltk
from nltk.tokenize import sent_tokenize


# Download tokenizer data
nltk.download("punkt")
nltk.download("punkt_tab")


INPUT_FILE = "medical_rag/data/processed/medlineplus_documents.json"
OUTPUT_FILE = "medical_rag/data/processed/medlineplus_chunks.json"


MAX_WORDS = 300
OVERLAP_SENTENCES = 1


def sentence_chunks( text, max_words=300, overlap_sentences=1):

    sentences = sent_tokenize(text)

    chunks = []

    current_chunk = []
    current_word_count = 0

    for sentence in sentences:

        sentence_word_count = len(
            sentence.split()
        )

        if (current_word_count + sentence_word_count <= max_words):

            current_chunk.append(sentence)

            current_word_count += (sentence_word_count)

        else:

            if current_chunk:

                chunks.append(" ".join(current_chunk))

            current_chunk = (current_chunk[-overlap_sentences:])

            current_word_count = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)

            current_word_count += (
                sentence_word_count
            )

    if current_chunk:

        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_document(document):

    text = document["text"]

    metadata = document["metadata"]

    chunks = sentence_chunks(
        text,
        max_words=MAX_WORDS,
        overlap_sentences=OVERLAP_SENTENCES
    )

    result = []

    document_id = metadata["id"]

    for index, chunk in enumerate(chunks):

        if not chunk.strip():
            continue

        chunk_metadata = metadata.copy()

        chunk_metadata["chunk_index"] = index

        chunk_id = (
            f"{document_id}_{index:04d}"
        )

        result.append({

            "chunk_id": chunk_id,

            "text": chunk,

            "metadata": chunk_metadata

        })

    return result


with open(INPUT_FILE,"r",encoding="utf-8") as file:

    documents = json.load(file)


print("Documents loaded:", len(documents))


all_chunks = []


for document in documents:

    chunks = chunk_document(document)

    all_chunks.extend(chunks)


with open(OUTPUT_FILE,"w", encoding="utf-8") as file:

    json.dump(all_chunks, file, indent=2, ensure_ascii=False)


print()
print("==============================")
print("Chunking complete")
print("==============================")

print("Documents:", len(documents))

print("Chunks:", len(all_chunks))

print("Saved:", OUTPUT_FILE)