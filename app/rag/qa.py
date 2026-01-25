from openai import OpenAI
from .prompt import build_prompt
from .retriever import retrieve_context
from .vector_store import VectorStore

def answer_question(
    question: str,
    vector_store: VectorStore,
    k: int = 5
) -> dict:
    from openai import OpenAI
    client = OpenAI()

    contexts = retrieve_context(question, vector_store, k=k)
    prompt = build_prompt(question, contexts)

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You answer questions using provided context only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            c["metadata"] for c in contexts
        ]
    }
