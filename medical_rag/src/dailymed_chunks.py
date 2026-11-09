import json
import re
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


INPUT_FILE = "medical_rag/data/processed/dailymed_metformin_documents.json"
OUTPUT_FILE = "medical_rag/data/processed/dailymed_metformin_chunks.json"

MAX_WORDS = 300
OVERLAP_SENTENCES = 1


# Sections we do not need for normal medical Q&A retrieval
EXCLUDED_SECTIONS = {
    "PRINCIPAL DISPLAY PANEL - 16 fl. oz. (473 mL) BOTTLE",
    "PRINCIPAL DISPLAY PANEL - 16 fl. oz. (473 mL) CARTON"
}


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_parent_section(section_title, all_section_titles):
    """
    Check whether a section is a parent section.

    Example:
        2 DOSAGE AND ADMINISTRATION
        2.1 Adult Dosage

    The first section is a parent because 2.1 exists.
    """

    match = re.match(r"^(\d+)\s+", section_title)

    if not match:
        return False

    section_number = match.group(1)

    for title in all_section_titles:

        child_match = re.match(
            rf"^{re.escape(section_number)}\.\d+\s+",
            title
        )

        if child_match:
            return True

    return False


def sentence_chunks(text, max_words=300, overlap_sentences=1):

    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:

        sentence = clean_text(sentence)

        if not sentence:
            continue

        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count <= max_words:

            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        else:

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            current_chunk = current_chunk[-overlap_sentences:]

            current_word_count = sum(
                len(s.split())
                for s in current_chunk
            )

            current_chunk.append(sentence)
            current_word_count += sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


with open(INPUT_FILE, "r", encoding="utf-8") as file:
    documents = json.load(file)


# Get all section names
all_section_titles = [
    document["metadata"]["section"]
    for document in documents
]


all_chunks = []


for document in documents:

    text = document["text"]
    metadata = document["metadata"]

    section = metadata["section"]

    # Skip packaging/display-panel information
    if section in EXCLUDED_SECTIONS:
        continue

    # Skip parent sections when child sections exist
    if is_parent_section(section, all_section_titles):
        print("Skipping parent section:", section)
        continue
    
    chunks = sentence_chunks(
        text,
        max_words=MAX_WORDS,
        overlap_sentences=OVERLAP_SENTENCES
    )

    for index, chunk in enumerate(chunks):

        if not chunk.strip():
            continue

        chunk_metadata = {
            "source": metadata["source"],
            "set_id": metadata["set_id"],
            "section": section,
            "url": metadata["url"],
            "drug": "Metformin Hydrochloride",
            "chunk_index": index
        }

        chunk_id = (
            f"dailymed_metformin_"
            f"{index:04d}_"
            f"{len(all_chunks):04d}"
        )

        all_chunks.append({
            "chunk_id": chunk_id,
            "text": chunk,
            "metadata": chunk_metadata
        })


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        all_chunks,
        file,
        indent=2,
        ensure_ascii=False
    )


print("\n" + "=" * 60)
print("DAILYMED CHUNKING COMPLETE")
print("=" * 60)

print("Sections processed:", len(documents))
print("Chunks created:", len(all_chunks))
print("Output:", OUTPUT_FILE)

if all_chunks:

    word_counts = [
        len(chunk["text"].split())
        for chunk in all_chunks
    ]

    print("Shortest chunk:", min(word_counts), "words")
    print("Longest chunk:", max(word_counts), "words")
    print(
        "Average chunk:",
        round(sum(word_counts) / len(word_counts), 2),
        "words"
    )

print("\nSections included:")

sections = sorted(
    set(chunk["metadata"]["section"] for chunk in all_chunks)
)

for section in sections:
    print("-", section)