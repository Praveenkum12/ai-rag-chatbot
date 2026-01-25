from typing import List, Dict


def build_prompt(
    question: str,
    contexts: List[Dict]
) -> str:
    context_text = "\n\n".join(
        [f"- {c['text']}" for c in contexts]
    )

    prompt = f"""
    You are a helpful assistant.
    Answer the question using ONLY the context below.
    If the answer is not contained in the context, say:
    "I don't know based on the provided documents."

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """
    return prompt.strip()
