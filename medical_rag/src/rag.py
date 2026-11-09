import json
import logging
import os
import re
from pathlib import Path
import time
import chromadb
from dotenv import load_dotenv
from ollama import chat
from sentence_transformers import CrossEncoder, SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
log = logging.getLogger("medical_rag")

CHROMA_PATH = str(PROJECT_DIR / "data" / "chroma_db")
COLLECTION_NAME = "medical_documents"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "qwen3:4b")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


MIN_RERANKER_SCORE = float(os.getenv("MIN_RERANKER_SCORE", "-6.0"))
SECTION_BOOST_MATCH = 5.0
CANDIDATES = 15
TOP_K = 3
MAX_CHUNKS_PER_DOC = 2
EMERGENCY_NUMBER = os.getenv("EMERGENCY_NUMBER", "112")

# --------------------------------
# 1. Models and database
# --------------------------------
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

db_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = db_client.get_collection(name=COLLECTION_NAME)

# Drug names known from label titles, e.g. "Metformin Hydrochloride" -> "metformin".
STOP_KEYS = {"drug", "medicine", "medication", "and", "the"}


def load_drug_keys():
    data = collection.get(include=["metadatas"])
    keys = set()
    for md in data["metadatas"] or []:
        title = (md or {}).get("title", "")
        if title:
            key = title.split()[0].lower()
            if key not in STOP_KEYS:
                keys.add(key)
    return keys


DRUG_KEYS = load_drug_keys()


# --------------------------------
# 2. Helpers
# --------------------------------
def contains_phrase(text, phrases):
    """Whole-phrase match, so 'take' does not match 'mistake'."""
    return any(re.search(rf"\b{re.escape(p)}\b", text) for p in phrases)


# --------------------------------
# 3. Safety checks
# --------------------------------
OVERDOSE_PHRASES = [
    "i took too much", "i accidentally took", "i took an overdose",
    "someone took too much", "someone overdosed", "overdose just happened",
    "suspected overdose",
]
URGENT_PHRASES = [
    "unconscious", "not responding", "can't breathe", "cannot breathe",
    "severe difficulty breathing", "stopped breathing", "chest pain",
    "seizure", "severe bleeding",
]
SELF_HARM_PHRASES = [
    "kill myself", "end my life", "suicide", "suicidal", "hurt myself",
    "self harm", "self-harm",
]


def detect_emergency(question):
    q = question.lower()
    if any(p in q for p in OVERDOSE_PHRASES + URGENT_PHRASES):
        return "medical"
    if any(p in q for p in SELF_HARM_PHRASES):
        return "self_harm"
    return None


def emergency_message(kind):
    if kind == "self_harm":
        return (
            "I'm really sorry you're going through this. You don't have to "
            "face it alone. Please contact a local crisis helpline or "
            f"emergency services ({EMERGENCY_NUMBER}) now, or reach out to "
            "someone you trust who can be with you."
        )
    return (
        "This may be a medical emergency. Contact emergency services "
        f"({EMERGENCY_NUMBER}) or local poison control now, or go to the "
        "nearest emergency department."
    )


# --------------------------------
# 4. Intent detection
# --------------------------------
INTENT_RULES = [
    ("contraindication", ["contraindication", "contraindications",
                          "should not use", "shouldn't use", "should not be used"]),
    ("overdose", ["overdose", "overdosing", "overdosage", "too much"]),
    ("indication", ["indication", "indications", "what is it used for",
                    "what does it treat", "used for", "purpose"]),
    ("adverse_reactions", ["side effect", "side effects", "adverse effect",
                           "adverse effects", "reaction", "reactions"]),
    ("drug_interactions", ["interaction", "interactions", "interact", "interact with"]),
    ("mechanism", ["how does", "how do", "how it works", "mechanism",
                   "mechanism of action"]),
    ("pregnancy", ["pregnancy", "pregnant"]),
    ("renal", ["kidney", "renal"]),
    ("dosage", ["dose", "doses", "dosage", "how much", "how many", "how often", "take"]),
    ("warnings", ["warning", "warnings", "warning and precaution", "warnings and precautions", "precautions", "boxed warning", "black box warning"]),
]


def detect_intent(question):
    q = question.lower()
    for intent, phrases in INTENT_RULES:
        if contains_phrase(q, phrases):
            return intent
    return "general"


PREFERRED_SECTIONS = {
    "dosage": ["dosage and administration", "adult dosage", "pediatric dosage"],
    "indication": ["indications and usage"],
    "contraindication": ["contraindications", "renal impairment"],
    "mechanism": ["mechanism of action"],
    "adverse_reactions": ["adverse reactions", "clinical studies experience",
                          "postmarketing experience"],
    "drug_interactions": ["drug interactions"],
    "pregnancy": ["pregnancy"],
    "renal": ["renal impairment"],
    "overdose": ["overdosage"],
    "warnings": ["warnings and precautions", "warning: lactic acidosis", "lactic acidosis",
                 "vitamin b 12", "hypoglycemia"],}


# --------------------------------
# 5. Retrieval scoring
# --------------------------------
def section_boost(question, section, intent):
    q = question.lower()
    sec = (section or "").lower()

    asks_adult = contains_phrase(q, ["adult", "adults"])
    asks_pediatric = contains_phrase(q, ["pediatric", "paediatric", "child", "children"])

    if intent == "dosage" and (asks_adult or asks_pediatric):
        wanted = "adult dosage" if asks_adult else "pediatric dosage"
        unwanted = "pediatric dosage" if asks_adult else "adult dosage"
        if wanted in sec:
            return SECTION_BOOST_MATCH
        if unwanted in sec:
            return 0.0

    if any(p in sec for p in PREFERRED_SECTIONS.get(intent, [])):
        return SECTION_BOOST_MATCH / 2

    return 0.0


def is_wrong_population(question, intent, section):
    """
    For dosage questions only: if the question names adults but not children,
    drop pediatric dosing sections (and the reverse). Applied before the
    on-topic filter and its fallback, so a fallback cannot bring them back.
    """
    if intent != "dosage":
        return False
    q = question.lower()
    sec = (section or "").lower()
    asks_adult = contains_phrase(q, ["adult", "adults"])
    asks_pediatric = contains_phrase(q, ["pediatric", "paediatric", "child", "children"])
    if asks_adult and not asks_pediatric:
        return "pediatric" in sec
    if asks_pediatric and not asks_adult:
        return "adult" in sec
    return False


def medicine_name_matches(question, title, source=""):
    """
    For drug-label sources only: if the question names a known drug,
    keep only labels for that drug. Otherwise do not filter.
    """
    source_lower = (source or "").lower()
    is_label = "dailymed" in source_lower or "drug label" in source_lower
    if not is_label:
        return True

    q = question.lower()
    mentioned = {k for k in DRUG_KEYS if re.search(rf"\b{re.escape(k)}\b", q)}
    if not mentioned:
        return True

    title_key = (title or "").split()[0].lower() if title else ""
    return title_key in mentioned


def deduplicate_documents(results, max_chunks_per_document=MAX_CHUNKS_PER_DOC):
    """
    Limit how many chunks can come from the same place.

    The key is (source, document_id, section). This means one DailyMed
    label can give chunks from several different sections (for example
    indications AND adult dosage), while still allowing at most
    max_chunks_per_document chunks from the same section.
    """
    counts = {}
    final = []

    for r in results:
        doc_id = r.get("document_id")

        if doc_id is None:
            key = ("unique", r.get("chunk_index", len(final)))
        else:
            key = (r.get("source"), doc_id, r.get("section"))

        count = counts.get(key, 0)
        if count >= max_chunks_per_document:
            continue

        final.append(r)
        counts[key] = count + 1

    return final


def retrieve_documents(question, top_k=TOP_K):
    intent = detect_intent(question)

    total = collection.count()
    if total == 0:
        log.warning("Collection is empty.")
        return []

    query_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(CANDIDATES, total),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    if not documents:
        return []

    scores = reranker.predict([[question, d] for d in documents])

    candidates = []
    for doc, md, dist, raw_score in zip(documents, metadatas, distances, scores):
        md = md or {}
        raw_score = float(raw_score)

        if not medicine_name_matches(question, md.get("title", ""), md.get("source", "")):
            log.debug("Excluded drug mismatch: %s", md.get("title"))
            continue

        # Apply the threshold before the boost, so a boost cannot rescue a weak match.
        if raw_score < MIN_RERANKER_SCORE:
            continue

        boost = section_boost(question, md.get("section"), intent)
        log.debug("Title=%s Section=%s raw=%.3f boost=%.1f",
                  md.get("title"), md.get("section"), raw_score, boost)

        candidates.append({
            "document": doc,
            "document_id": md.get("document_id"),
            "title": md.get("title", ""),
            "source": md.get("source", ""),
            "section": md.get("section", ""),
            "url": md.get("url", ""),
            "chunk_index": md.get("chunk_index"),
            "chroma_distance": dist,
            "reranker_score": raw_score,
            "section_boost": boost,
            "final_score": raw_score + boost,
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    deduped = deduplicate_documents(candidates)

    # Remove dosing sections for the wrong population before the on-topic
    # filter, so its fallback cannot reintroduce them.
    before = len(deduped)
    deduped = [c for c in deduped
               if not is_wrong_population(question, intent, c.get("section"))]
    if len(deduped) < before:
        log.info("Excluded %d section(s) for the other population", before - len(deduped))

    preferred = PREFERRED_SECTIONS.get(intent, [])
    if preferred:
        on_topic = [
            c for c in deduped
            if any(p in (c.get("section") or "").lower() for p in preferred)
        ]
        # Use on-topic sections when there are any; otherwise fall back to everything.
        if on_topic:
            deduped = on_topic
        elif intent == "dosage":
            # Safety: do not answer a dose question from unrelated sections
            # (for example pharmacokinetics). Returning nothing lets the
            # caller say the sources do not contain the answer.
            log.warning("No dosage section among %d candidates; returning no evidence.",
                        len(deduped))
            return []
        else:
            # Visible in the logs: the answer may come from sections that do not
            # match the question's intent.
            log.warning(
                "No preferred section for intent=%s among %d candidates; "
                "falling back to all of them", intent, len(deduped))

    selected = deduped[:top_k]
    log.info("Intent=%s, selected %d documents", intent, len(selected))
    return selected


# --------------------------------
# 6. Context and sources
# --------------------------------
def build_context_and_sources(retrieved_docs):
    context_parts = []
    sources_by_id = {}
    source_ids_by_key = {}

    for item in retrieved_docs:
        text = item.get("document") or ""
        if not text:
            continue

        key = (item["source"], item["title"], item["section"], item["url"])

        if key not in source_ids_by_key:
            sid = f"S{len(sources_by_id) + 1}"
            source_ids_by_key[key] = sid
            sources_by_id[sid] = {
                "source": key[0], "title": key[1],
                "section": key[2], "url": key[3],
                "content": text,
            }
        else:
            sid = source_ids_by_key[key]
            if text not in sources_by_id[sid]["content"]:
                sources_by_id[sid]["content"] += "\n" + text

        context_parts.append(
            f"[{sid}]\nSource: {key[0]}\nTitle: {key[1]}\n"
            f"Section: {key[2]}\nContent:\n{text}"
        )

    return "\n\n".join(context_parts), sources_by_id


# --------------------------------
# 7. LLM calls
# --------------------------------
CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "source_ids"],
            },
        }
    },
    "required": ["claims"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"],
        },
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def call_llm(messages, schema=None,model=None):
    kwargs = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "think": False,
        "options": {"temperature": 0,"num_predict": 2000},
    }

    if schema is not None:
        kwargs["format"] = schema

    start_time = time.perf_counter()
    log.info("LLM call STARTED")

    try:
        response = chat(**kwargs)
        return response.message.content.strip()
    finally:
        elapsed = time.perf_counter() - start_time
        log.info("LLM call FINISHED in %.2f seconds", elapsed)


def create_claim_messages(question, context, feedback=""):
    system = (
        "You are a medical information assistant. Use only the supplied "
        "context. Never invent facts or source IDs."
    )
    user = f"""
You are a careful medical information extraction assistant.

Answer the user's exact question using ONLY the supplied context.

Rules:
1. Include only facts that directly answer the question.
2. Prefer the most relevant section for the detected intent.
3. Do not include unrelated information from other sections.
4. For a simple question, aim for 2-5 concise claims when the evidence supports that.
   Do not omit important facts or warnings just to meet this target.
5. Each claim must express one factual point.
6. Preserve all numbers, units, conditions, and qualifications exactly.
7. Do not combine information about different formulations
   (for example, tablets and oral solution) unless the context clearly supports it.
8. Do not correct, guess, or complete missing information using outside knowledge.
9. Every claim must include the source ID that supports it.
10. If the context does not answer the question, return an empty claims list.
11. Keep spelling and grammar clear without changing the source's meaning.
12. Keep every condition from the source sentence in the claim, such as "if", "when", "in patients with", or "suspected". Never drop a condition.
13. Do not add comparisons, percentages, or phrases such as "more commonly reported than placebo" unless the passage states them.
14. When the context lists many items that share the same frequency (for example several reactions each "reported in ≥ 1% to ≤ 5%"), you may combine them into ONE claim. That claim must name every item and keep the exact frequency. Never drop an item to shorten the answer.

{feedback}
Return only data that matches the required JSON schema.
CONTEXT:
{context}

QUESTION:
{question}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_claims(question, context, feedback=""):
    raw = call_llm(create_claim_messages(question, context, feedback), CLAIM_SCHEMA, model=VERIFY_MODEL)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error(
            "Model returned invalid JSON (%d chars) at char %d: %s. Tail: %r",
            len(raw), e.pos, e.msg, raw[-300:],
        )
        return []
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        log.error("Model returned unexpected structure.")
        return []
    return data["claims"]


def verify_with_llm(claim_text, passage):
    raw = call_llm(
        [
            {"role": "system", "content": (
                "Verify the claim strictly against the supplied passage. "
                "Treat the passage as the only source of truth. "
                "Check every number, unit, date, frequency, condition, "
                "negation, and relationship in the claim. "
                "SUPPORTED means the passage directly supports the entire claim "
                "without changing its meaning. "
                "CONTRADICTED means the passage directly conflicts with the claim. "
                "If any detail is missing, ambiguous, or not directly supported, "
                "return INSUFFICIENT_EVIDENCE. "
                "Do not use outside knowledge."            )},
            {"role": "user", "content": f"CLAIM:\n{claim_text}\n\nPASSAGE:\n{passage}"},
        ],
        VERIFY_SCHEMA,
        model=OLLAMA_MODEL,
    )
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return "INSUFFICIENT_EVIDENCE"
    verdict = result.get("verdict") if isinstance(result, dict) else None
    if verdict in {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}:
        return verdict
    return "INSUFFICIENT_EVIDENCE"


# --------------------------------
# 8. Verification
# --------------------------------
def _normalize_numbers(text):
    # Remove thousands commas so "2,550" becomes "2550".
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


NUMBER_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])")


# Number followed by an optional unit, e.g. "500 mg", "5 mL", "53%".
NUMBER_UNIT_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])\s*([A-Za-zµμ%][A-Za-zµμ/%]*)?"
)
# Units checked against the source. Other words after a number are ignored,
# so a harmless word like "times" in a paraphrase cannot cause a rejection.
CHECKED_UNITS = {
    "mg", "g", "mcg", "µg", "μg", "kg", "ml", "l", "meq", "mmol", "%",
    "mg/dl", "mcg/ml", "mmol/l", "mmol/liter", "ml/min", "min", "hours",
    "hour", "days", "day", "weeks", "week", "months", "month", "years", "year",
}


def numbers_supported(claim_text, passage):
    """
    Every number in the claim must appear in the passage as a whole number,
    and every checked unit next to it must match the source too.
    '2500' is not accepted just because '500' appears somewhere in the passage.
    '500 mg' is not accepted when the passage only says '500 mL'.
    """
    passage_norm = _normalize_numbers(passage)
    claim_norm = _normalize_numbers(claim_text)
    for num in NUMBER_RE.findall(claim_norm):
        if not re.search(rf"(?<![\d.]){re.escape(num)}(?![\d.])", passage_norm):
            return False
    for num, unit in NUMBER_UNIT_RE.findall(claim_norm):
        if unit.lower() not in CHECKED_UNITS:
            continue
        pattern = rf"(?<![\d.]){re.escape(num)}(?![\d.])\s*{re.escape(unit)}(?![A-Za-z])"
        if not re.search(pattern, passage_norm, flags=re.IGNORECASE):
            return False
    return True


def clean_claims(claims, sources_by_id):
    valid = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = claim.get("text")
        ids = claim.get("source_ids")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(ids, list) or not ids:
            continue
        ids = list(dict.fromkeys(ids))
        if not all(isinstance(i, str) and i in sources_by_id for i in ids):
            continue
        valid.append({"text": text.strip(), "source_ids": ids})
    return valid


def verify_claim(claim_text, source_ids, sources_by_id):
    """A claim passes only if every cited source passes both checks."""
    for sid in source_ids:
        passage = sources_by_id[sid]["content"]

        if not numbers_supported(claim_text, passage):
            return False, f"number in claim not found in {sid}"

        verdict = verify_with_llm(claim_text, passage)
        if verdict != "SUPPORTED":
            return False, f"{sid}: {verdict}"

    return True, "all cited sources support the claim"


# --------------------------------
# 9. Answer pipeline
# --------------------------------
NO_EVIDENCE = (
    "The available sources do not provide enough verified information "
    "to answer this question."
)
DOSAGE_DISCLAIMER = (
    "This is general information from the supplied drug labels, not medical "
    "advice. Doses depend on the individual (for example kidney function, age "
    "and other medicines). Confirm with a doctor or pharmacist."
)
OMITTED_NOTE = (
    "Note: some information from the sources could not be verified and was "
    "left out. Check the drug label for the complete details."
)
MAX_GENERATION_ATTEMPTS = 2  # one retry, only when a number check fails
STARTING_DOSE_WARNING = (
    "Warning: the starting dose could not be verified from the sources. "
    "Check the drug label for the exact starting dose before using this information."
)


def build_repair_feedback(rejected_claims, sources_by_id):
    """
    Tell the model exactly which claims failed the number check and show it
    the source text they cite, so the retry can copy the numbers correctly.
    """
    lines = [
        "A previous attempt was rejected because some numbers or units did not "
        "match the source. Rewrite these claims so every number and unit is "
        "copied exactly as it is written in the CONTEXT:"
    ]
    for claim in rejected_claims:
        lines.append(f'- Rejected claim: "{claim["text"]}"')
        for sid in claim["source_ids"]:
            lines.append(f"  Source text for {sid}: {sources_by_id[sid]['content']}")
    lines.append("Do not retype numbers from memory. Copy every digit, comma and unit.")
    return "\n".join(lines)


# A label sentence that lists reactions after a frequency, e.g.
# "...reported in ≥ 1% to ≤ 5% of patients ...: abnormal stools, rash, chills."
LIST_SENTENCE_RE = re.compile(r"reported in[^:]{0,200}?%[^:]{0,200}?:\s*([^.]+)\.")
MISSING_ITEMS_NOTE = (
    "Note: these reactions are listed in the drug label but are not in the answer "
    "above: {items}. Check the drug label for the complete list."
)


def list_items_from_passages(sources_by_id):
    """
    Items from label lists such as "reported in ≥ 1% to ≤ 5% ...: a, b, and c."
    Only lists with at least 3 items count, so a single item is not treated as a list.
    """
    items = []
    for src in sources_by_id.values():
        for m in LIST_SENTENCE_RE.finditer(src.get("content", "")):
            raw = m.group(1).replace(", and ", ", ").replace(" and ", ", ")
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) >= 3:
                items.extend(p for p in parts if p not in items)
    return items


def missing_list_items(verified, sources_by_id):
    """
    Label list items that no verified claim mentions. An item counts as
    mentioned when all of its words appear in one claim, so "increased sweating"
    matches "sweating increased".
    """
    claims_text = [c["text"].lower() for c in verified]
    missing = []
    for item in list_items_from_passages(sources_by_id):
        words = re.findall(r"[a-z0-9]+", item.lower())
        if not any(all(re.search(rf"\b{re.escape(w)}\b", t) for w in words) for t in claims_text):
            missing.append(item)
    return missing


def build_missing_feedback(missing):
    return (
        "A previous attempt left these reactions out of the answer, but they are "
        "in the source list: " + ", ".join(missing) + ". Include each of them. "
        "You may combine them into one claim that names every item and keeps the "
        "exact frequency."
    )


def generate_and_verify(question, context, sources_by_id):
    """
    Generate claims and keep only the ones that pass verification.

    Returns (verified, dropped, missing):
      dropped - how many claims were removed for any reason
      missing - label list items (adverse reactions only) that no kept claim mentions

    Retries once when a number check failed or when list items are missing,
    with feedback naming the problem. The better attempt is kept: fewer missing
    items first, then more verified claims. A tie goes to the retry.
    """
    check_lists = detect_intent(question) == "adverse_reactions"
    best = None  # (score, verified, dropped, missing)
    feedback = ""
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        raw_claims = generate_claims(question, context, feedback)[:6]
        claims = clean_claims(raw_claims, sources_by_id)
        dropped = len(raw_claims) - len(claims)
        number_rejects = []

        verified = []
        for claim in claims:
            try:
                ok, reason = verify_claim(claim["text"], claim["source_ids"], sources_by_id)
            except Exception:
                log.exception("Verification failed for claim: %s", claim["text"])
                dropped += 1
                continue
            log.info("%s | %s | %s (attempt %d)",
                     "KEEP" if ok else "REJECT", reason, claim["text"], attempt)
            if ok:
                verified.append(claim)
            else:
                dropped += 1
                if reason.startswith("number in claim"):
                    number_rejects.append(claim)

        missing = missing_list_items(verified, sources_by_id) if check_lists else []
        score = (len(missing), -len(verified))
        if best is None or score <= best[0]:
            best = (score, verified, dropped, missing)

        feedback_parts = []
        if number_rejects:
            log.info("A number check failed; retrying once with feedback about the rejected claims.")
            feedback_parts.append(build_repair_feedback(number_rejects, sources_by_id))
        if missing:
            log.info("Label list items missing from the answer: %s", ", ".join(missing))
            feedback_parts.append(build_missing_feedback(missing))
        if not feedback_parts:
            break
        feedback = "\n".join(feedback_parts)

    _, best_verified, best_dropped, best_missing = best
    return best_verified, best_dropped, best_missing


def starting_dose_missing(question, context, verified):
    """
    True when the question is about dosage, the sources contain a starting
    dose, and none of the verified claims states it. The dosage check is a
    simple phrase test, so it only fires when the sources themselves say
    'starting dose'.
    """
    if detect_intent(question) != "dosage":
        return False
    if "starting dose" not in context.lower():
        return False
    return not any("starting dose" in c["text"].lower() for c in verified)


def answer_question(question):
    question = (question or "").strip()
    if not question:
        return "Please enter a question."

    emergency = detect_emergency(question)
    if emergency:
        return emergency_message(emergency)

    retrieved = retrieve_documents(question)
    context, sources_by_id = build_context_and_sources(retrieved)
    if not context.strip():
        return NO_EVIDENCE

    try:
        verified, dropped, missing = generate_and_verify(question, context, sources_by_id)
    except Exception as exc:  # e.g. Ollama not running
        log.exception("LLM call failed: %s", exc)
        return "The language model is unavailable right now. Please try again later."

    if not verified:
        return NO_EVIDENCE

    body = "\n".join(
        f"- {c['text']} "
        + " ".join(f"[{s}]" for s in c["source_ids"])
        for c in verified
    )

    cited = []
    for c in verified:
        for s in c["source_ids"]:
            if s not in cited:
                cited.append(s)

    def format_source(s):
        title = sources_by_id[s]["title"] or "Source"
        section = sources_by_id[s]["section"]
        url = sources_by_id[s]["url"]
        text = f"[{s}] {title}"
        if section:
            text += f"  {section}"
        if url:
            text += f" ({url})"
        return text

    sources_block = "\n".join(format_source(s) for s in cited)

    parts = []
    # Shown first: the answer must not look complete when the starting dose is missing.
    if starting_dose_missing(question, context, verified):
        parts += [STARTING_DOSE_WARNING, ""]
    parts.append(body)
    if dropped:
        parts += ["", OMITTED_NOTE]
    if missing:
        parts += ["", MISSING_ITEMS_NOTE.format(items=", ".join(missing))]
    parts += ["", "Sources:", sources_block]
    if detect_intent(question) == "dosage":
        parts += ["", DOSAGE_DISCLAIMER]
    return "\n".join(parts)


if __name__ == "__main__":
    question = input("Ask a medical question: ")
    print("\n ------------------ANSWER-----------------------\n" + answer_question(question))