import json
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. File paths
# ============================================================

HEALTH_TOPIC_FILE = (
    "data/processed/medlineplus_chunks.json"
)

GENETICS_FILE = (
    "data/processed/medlineplus_genetics_chunks.json"
)

DAILYMED_FILE = (
    "data/processed/dailymed_metformin_chunks.json"
)


# ============================================================
# 2. Chroma settings
# ============================================================

CHROMA_PATH = "data/chroma_db"

COLLECTION_NAME = "medical_documents"

BATCH_SIZE = 100


# ============================================================
# 3. Load embedding model
# ============================================================

print("=" * 60)
print("Loading embedding model...")
print("=" * 60)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 4. Connect to Chroma
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# ============================================================
# 5. Delete old collection
# ============================================================

print("\nChecking existing Chroma collection...")

try:

    client.delete_collection(
        name=COLLECTION_NAME
    )

    print("Old collection deleted.")

except Exception:

    print("No existing collection found.")


# ============================================================
# 6. Create fresh collection
# ============================================================

collection = client.create_collection(
    name=COLLECTION_NAME
)

print("Fresh Chroma collection created.")


# ============================================================
# 7. Load all three datasets
# ============================================================

print("\nLoading datasets...")


with open(
    HEALTH_TOPIC_FILE,
    "r",
    encoding="utf-8"
) as file:

    health_chunks = json.load(file)


with open(
    GENETICS_FILE,
    "r",
    encoding="utf-8"
) as file:

    genetics_chunks = json.load(file)


with open(
    DAILYMED_FILE,
    "r",
    encoding="utf-8"
) as file:

    dailymed_chunks = json.load(file)


print()
print("MedlinePlus Health Topics:", len(health_chunks))
print("MedlinePlus Genetics:", len(genetics_chunks))
print("DailyMed:", len(dailymed_chunks))

total_chunks = (
    len(health_chunks)
    + len(genetics_chunks)
    + len(dailymed_chunks)
)

print("Total chunks:", total_chunks)


# ============================================================
# 8. Function to add chunks
# ============================================================

def add_chunks(
    chunks,
    source_name
):

    print()
    print("-" * 60)
    print("Adding:", source_name)
    print("Chunks:", len(chunks))
    print("-" * 60)

    for start in range(
        0,
        len(chunks),
        BATCH_SIZE
    ):

        batch = chunks[
            start:start + BATCH_SIZE
        ]

        texts = []
        embeddings = []
        metadatas = []
        ids = []

        for chunk in batch:

            text = chunk["text"]

            metadata = chunk["metadata"]

            chunk_id = chunk["chunk_id"]


            # ----------------------------------------
            # Create embedding
            # ----------------------------------------

            embedding = model.encode(
                text
            ).tolist()


            texts.append(text)

            embeddings.append(
                embedding
            )


            # ====================================================
            # MedlinePlus Health Topics
            # ====================================================

            if source_name == "MedlinePlus":

                metadatas.append({

                    "source": "MedlinePlus",

                    "title": metadata.get(
                        "title",
                        ""
                    ),

                    "url": metadata.get(
                        "url",
                        ""
                    ),

                    "language": metadata.get(
                        "language",
                        "English"
                    ),

                    "document_id": metadata.get(
                        "id",
                        ""
                    ),

                    "chunk_index": metadata.get(
                        "chunk_index",
                        0
                    )
                })


            # ====================================================
            # MedlinePlus Genetics
            # ====================================================

            elif source_name == "MedlinePlus Genetics":

                metadatas.append({

                    "source": "MedlinePlus Genetics",

                    "record_type": metadata.get(
                        "record_type",
                        ""
                    ),

                    "title": metadata.get(
                        "title",
                        ""
                    ),

                    "url": metadata.get(
                        "url",
                        ""
                    ),

                    "language": "English",

                    "document_id": metadata.get(
                        "id",
                        ""
                    ),

                    "chunk_index": metadata.get(
                        "chunk_index",
                        0
                    )
                })


            # ====================================================
            # DailyMed
            # ====================================================

            elif source_name == "DailyMed":

                metadatas.append({

                    "source": "DailyMed",

                    "title": metadata.get(
                        "drug",
                        ""
                    ),

                    "url": metadata.get(
                        "url",
                        ""
                    ),

                    "language": "English",

                    "document_id": metadata.get(
                        "set_id",
                        ""
                    ),

                    "section": metadata.get(
                        "section",
                        ""
                    ),

                    "chunk_index": metadata.get(
                        "chunk_index",
                        0
                    )
                })


            ids.append(
                chunk_id
            )


        # ========================================================
        # Add batch to Chroma
        # ========================================================

        collection.add(

            ids=ids,

            documents=texts,

            embeddings=embeddings,

            metadatas=metadatas

        )


        print(
            f"Added {min(start + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)}"
        )


# ============================================================
# 9. Add all three sources
# ============================================================

add_chunks(
    health_chunks,
    "MedlinePlus"
)

add_chunks(
    genetics_chunks,
    "MedlinePlus Genetics"
)

add_chunks(
    dailymed_chunks,
    "DailyMed"
)


# ============================================================
# 10. Final result
# ============================================================

print()
print("=" * 60)
print("CHROMA REBUILD COMPLETE")
print("=" * 60)

print(
    "MedlinePlus chunks:",
    len(health_chunks)
)

print(
    "Genetics chunks:",
    len(genetics_chunks)
)

print(
    "DailyMed chunks:",
    len(dailymed_chunks)
)

print(
    "Expected total:",
    total_chunks
)

print(
    "Actual Chroma records:",
    collection.count()
)

print("=" * 60)