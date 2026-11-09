import json
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder



QUESTIONS_FILE = "medical_rag/data/evaluation_questions.json"


print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


client = chromadb.PersistentClient(
    path="medical_rag/data/chroma_db"
)

collection = client.get_collection(
    name="medical_documents"
)



with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
    questions = json.load(file)

print("Evaluation questions:", len(questions))


def recall_at_k(results, expected_title, k):
    """
    Returns 1 if the expected document
    appears in the first k results.
    Otherwise returns 0.
    """

    for result in results[:k]:

        if result["title"].lower() == expected_title.lower():
            return 1

    return 0


def reciprocal_rank(results, expected_title):


    for rank, result in enumerate(results, start=1):

        if result["title"].lower() == expected_title.lower():
            return 1 / rank

    return 0

def deduplicate_documents(reranked_results,max_chunks_per_document=2):
    """
    Keep at most a fixed number of chunks
    from the same original document.
    """

    document_counts = {}
    final_results = []

    for result in reranked_results:

        document_id = result.get("document_id")

        # Safety fallback
        if document_id is None:
            document_id = (
                "unique_" +
                str(result.get("chunk_index", len(final_results)))
            )

        count = document_counts.get(document_id, 0)

        if count >= max_chunks_per_document:
            continue

        final_results.append(result)

        document_counts[document_id] = count + 1

    return final_results




evaluation_results = []


for item in questions:

    question = item["question"]
    expected_title = item["expected_topic"]

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("EXPECTED:", expected_title)
    print("=" * 70)


    # --------------------------------------------------
    # CHROMA RETRIEVAL
    # --------------------------------------------------

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"language": "English"},
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]


    # --------------------------------------------------
    # CREATE CHROMA RESULTS
    # --------------------------------------------------

    chroma_results = []

    for i in range(len(documents)):

        chroma_results.append({
            "title": metadatas[i]["title"],
            "distance": distances[i],
            "document": documents[i]
        })

    print("\nAll ChromaDB candidate titles:")

    for rank, result in enumerate(chroma_results, start=1):
        print(rank, result["title"])


    # --------------------------------------------------
    # RERANK
    # --------------------------------------------------

    pairs = [
        [question, document]
        for document in documents
    ]

    scores = reranker.predict(pairs)


    reranked_results = []

    for i in range(len(documents)):

        reranked_results.append({
            "title": metadatas[i]["title"],
            "score": float(scores[i]),
            "document": documents[i],
            "document_id": metadatas[i].get("document_id"),
            "chunk_index": metadatas[i].get("chunk_index")
        })


    reranked_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    final_results = deduplicate_documents(
        reranked_results,
        max_chunks_per_document=2
    )

    # --------------------------------------------------
    # CALCULATE CHROMA METRICS
    # --------------------------------------------------

    chroma_recall_3 = recall_at_k(
        chroma_results,
        expected_title,
        3
    )

    chroma_recall_5 = recall_at_k(
        chroma_results,
        expected_title,
        5
    )

    chroma_recall_10 = recall_at_k(
        chroma_results,
        expected_title,
        10
    )

    chroma_mrr = reciprocal_rank(
        chroma_results,
        expected_title
    )


    # --------------------------------------------------
    # CALCULATE RERANKER METRICS
    # --------------------------------------------------
    reranker_recall_3 = recall_at_k(
        final_results, expected_title, 3
    )

    reranker_recall_5 = recall_at_k(
        final_results, expected_title, 5
    )

    reranker_recall_10 = recall_at_k(
        final_results, expected_title, 10
    )

    reranker_mrr = reciprocal_rank(
        final_results, expected_title
    )


    # --------------------------------------------------
    # FIND RERANKER RANK
    # --------------------------------------------------

    reranker_rank = None

    for rank, result in enumerate(
        final_results,
        start=1
    ):

        if (
            result["title"].lower()
            == expected_title.lower()
        ):

            reranker_rank = rank
            break


    # --------------------------------------------------
    # PRINT QUESTION RESULTS
    # --------------------------------------------------

    # Print the top ChromaDB result
    print("\nTop ChromaDB result:")
    print("Title:", chroma_results[0]["title"])
    print("Document:", chroma_results[0]["document"])

    # Print the top reranked result
    print("\nTop reranked result:")
    print("Title:", final_results[0]["title"])
    print("Document:", final_results[0]["document"])
    print("  Recall@3 :", chroma_recall_3)
    print("  Recall@5 :", chroma_recall_5)
    print("  Recall@10:", chroma_recall_10)
    print("  MRR      :", round(chroma_mrr, 4))


    print("\nReranker:")
    print("  Recall@3 :", reranker_recall_3)
    print("  Recall@5 :", reranker_recall_5)
    print("  Recall@10:", reranker_recall_10)
    print("  MRR      :", round(reranker_mrr, 4))
    print("  Correct rank:", reranker_rank)


    print("\nChecking whether A1C exists in ChromaDB...")

    a1c_results = collection.get(
        where={"title": "A1C"},
        include=["metadatas"]
    )

    print("Matching documents:", len(a1c_results["ids"]))

    for metadata in a1c_results["metadatas"]:
        print(metadata)

    
import numpy as np

print("\n" + "=" * 60)
print("A1C DOCUMENT DIAGNOSTIC")
print("=" * 60)

# 1. Retrieve the exact A1C document stored in ChromaDB
a1c_record = collection.get(
    where={"title": "A1C"},
    include=["documents", "metadatas", "embeddings"]
)

print("Number of A1C records:", len(a1c_record["ids"]))

if len(a1c_record["ids"]) > 0:
    stored_text = a1c_record["documents"][0]
    stored_metadata = a1c_record["metadatas"][0]
    stored_embedding = np.array(a1c_record["embeddings"][0])

    print("\nMetadata:", stored_metadata)
    print("\nStored A1C document text:")
    print(stored_text[:2000])

    print("\nStored embedding dimensions:", len(stored_embedding))
    print("Stored embedding norm:", np.linalg.norm(stored_embedding))

    # 2. Test the question directly against the A1C record
    question = "What is HbA1c?"
    query_embedding = embedding_model.encode(question).tolist()

    direct_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        where={"title": "A1C"},
        include=["metadatas", "distances"]
    )

    print("\nDirect A1C search distance:")
    print(direct_result["distances"][0][0])

    # 3. Check the document's position among more search candidates
    candidate_count = min(collection.count(), 100)

    all_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        where={"language": "English"},
        include=["metadatas", "distances"]
    )

    titles = all_results["metadatas"][0]
    a1c_ranks = [
        i + 1 for i, metadata in enumerate(titles)
        if metadata.get("title", "").lower() == "a1c"
    ]

    print("\nA1C rank among up to 100 candidates:", a1c_ranks)
    if not a1c_ranks:
        print("A1C was not found among these candidates.")
else:
    print("A1C document was not found.")


import numpy as np

record = collection.get(
    where={"title": "A1C"},
    include=["documents", "embeddings"]
)

stored_embedding = np.array(record["embeddings"][0])

fresh_embedding = embedding_model.encode(
    record["documents"][0],
    normalize_embeddings=True
)

print("Stored dimensions:", stored_embedding.shape)
print("Fresh dimensions:", fresh_embedding.shape)

print(
    "Cosine similarity:",
    np.dot(stored_embedding, fresh_embedding)
    / (
        np.linalg.norm(stored_embedding)
        * np.linalg.norm(fresh_embedding)
    )
)

