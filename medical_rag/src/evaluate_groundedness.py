import json
from ollama import chat


OLLAMA_MODEL = "qwen3:8b"


def evaluate_groundedness(question, context, answer):

    prompt = f"""
You are evaluating a medical RAG answer.

Your job is NOT to answer the medical question.

Determine whether the ANSWER is supported by the CONTEXT.

Rules:
1. Check each important factual claim in the answer.
2. A claim is SUPPORTED only if the context provides information that supports it.
3. A claim is UNSUPPORTED if the context does not provide enough evidence.
4. Do not use your own medical knowledge.
5. Do not judge whether the medical information is true in general.
6. Focus only on whether it is supported by the provided context.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Return ONLY valid JSON in this format:

{{
    "grounded": true,
    "score": 0.0,
    "supported_claims": [],
    "unsupported_claims": []
}}

The score should be between 0 and 1.
"""


    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response["message"]["content"]

    return result

if __name__ == "__main__":

    question = "What is prediabetes?"

    context = """
Prediabetes

What is prediabetes? Prediabetes means that your blood glucose,
or blood sugar, levels are higher than normal but not high enough
to be called diabetes.

If you have prediabetes, you are more likely to develop type 2
diabetes, heart disease, and stroke.

If you make some lifestyle changes now, you may be able to delay
or prevent type 2 diabetes.
"""

    answer = """
Prediabetes means that your blood glucose levels are higher than
normal but not high enough to be called diabetes.

People with prediabetes are more likely to develop type 2 diabetes,
heart disease, and stroke.

Lifestyle changes may help delay or prevent type 2 diabetes.
"""

    result = evaluate_groundedness(
        question,
        context,
        answer
    )

    print("\n")
    print("=" * 60)
    print("GROUNDEDNESS EVALUATION")
    print("=" * 60)
    print(result)