from anthropic import Anthropic

from src.config.settings import (
    get_anthropic_api_key,
    CLAUDE_MODEL,
    MAX_OUTPUT_TOKENS,
)


class ClaudeGenerator:

    def __init__(self):

        api_key = get_anthropic_api_key()
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Run using Infisical."
            )

        self.client = Anthropic(api_key=api_key)

    def generate_answer(
        self,
        query,
        context
    ):

        prompt = f"""
You are a highly capable PDF-based RAG (Retrieval-Augmented Generation) assistant.

Your mandate is to answer the user's question using ONLY the retrieved context provided below.

Follow these critical operational rules:

1. STRICT GROUNDING: Base your answer entirely on the retrieved context. Do not use any external or training knowledge to answer the question.
2. CITATION PROTOCOL: For every factual claim you make, you MUST cite the source by adding `[Source X]` at the end of the sentence or claim (e.g., "...this model uses 8 attention heads [Source 1]"). The index `X` must match the corresponding "Source X" block in the retrieved context below.
3. NOT FOUND PROTOCOL: If the retrieved context contains absolutely no relevant information, or insufficient information to answer the question, respond exactly with: "Not found in the uploaded PDFs." Do not attempt to guess or answer from general knowledge.

Retrieved Context:
{context}

User Question:
{query}

Answer:
"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.content[0].text