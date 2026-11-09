"""
Read-only retrieval check for medical_rag.

Run from the project root (the folder that contains src/ and data/):
    python check_retrieval.py
    python check_retrieval.py "What is metformin used for?"

It does not write or change any files. It prints the top candidates
from Chroma, their reranker scores, section boosts, and what survives
the filters in rag.retrieve_documents().
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  
import src.rag as rag 

question = sys.argv[1] if len(sys.argv) > 1 else "What is metformin used for?"
intent = rag.detect_intent(question)
print(f"QUESTION: {question}")
print(f"INTENT:   {intent}")
print(f"PREFERRED SECTIONS: {rag.PREFERRED_SECTIONS.get(intent, [])}")
print(f"CANDIDATES={rag.CANDIDATES}  TOP_K={rag.TOP_K}  MIN_RERANKER_SCORE={rag.MIN_RERANKER_SCORE}")
print()

total = rag.collection.count()
emb = rag.embedding_model.encode(question).tolist()
res = rag.collection.query(
    query_embeddings=[emb],
    n_results=min(rag.CANDIDATES, total),
    include=["documents", "metadatas", "distances"],
)
docs = res["documents"][0]
mds = res["metadatas"][0]
dists = res["distances"][0]
scores = rag.reranker.predict([[question, d] for d in docs]) if docs else []

print("CANDIDATES FROM CHROMA (rank, distance, reranker, boost, filters):")
print(f"{'#':>2}  {'dist':>6}  {'rerank':>7}  {'boost':>5}  {'drug':>4}  {'thr':>3}  source | section | title")
for i, (d, md, dist, s) in enumerate(zip(docs, mds, dists, scores), 1):
    md = md or {}
    raw = float(s)
    drug_ok = rag.medicine_name_matches(question, md.get("title", ""), md.get("source", ""))
    thr_ok = raw >= rag.MIN_RERANKER_SCORE
    boost = rag.section_boost(question, md.get("section"), intent)
    print(f"{i:>2}  {dist:6.3f}  {raw:7.3f}  {boost:5.1f}  {'ok' if drug_ok else 'NO':>4}  {'ok' if thr_ok else 'NO':>3}  "
          f"{md.get('source','')} | {md.get('section')} | {md.get('title','')[:40]}")

print()
selected = rag.retrieve_documents(question)
print(f"SELECTED BY retrieve_documents() ({len(selected)} documents):")
for item in selected:
    print(f"  final={item['final_score']:.3f}  source={item['source']}  section={item['section']}  title={item['title'][:40]}")
