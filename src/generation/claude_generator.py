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

Your mandate is to answer the user's question using the retrieved context provided below. If the context is incomplete or only partially relevant, you should supplement it with your own internal knowledge to provide a complete answer, ensuring you strictly and explicitly separate document facts from your own inferences.

Follow these critical operational rules:

1. HYBRID PARTIAL INFORMATION PROTOCOL: If the retrieved context contains some relevant information (even if it is small, partial, or does not directly/fully answer the question), you MUST:
    * First, explain the facts explicitly written in the retrieved text, citing the sources using `[Source X]` at the end of the sentence or claim (where `X` matches the "Source X" block in the context).
    * Second, explicitly state what is missing from the document context (e.g., "The provided documents do not contain details about [missing aspect]").
    * Third, answer the remaining part of the question using your internal/general knowledge. You MUST introduce this section by explicitly stating: "However, based on general knowledge..." and do not attribute this part to any document source (to avoid hallucination).

2. ZERO-MATCH PROTOCOL: If the retrieved context is completely empty, OR if the user's question is completely unrelated to the topics/entities covered by the retrieved context (e.g. asking about general knowledge like "what is the capital of France", "how to bake cookies", or other non-ML/LLM topics that have absolutely no representation in the retrieved text), respond exactly with: "No context present in the uploaded PDFs." Do not attempt to answer or guess from general knowledge in this case.

3. CITATION PROTOCOL: For every factual claim you make from the retrieved documents, you MUST cite the source by adding `[Source X]` at the end of the sentence or claim (e.g., "...this model uses 8 attention heads [Source 1]"). The index `X` must match the corresponding "Source X" block in the retrieved context below. Do not use citations for claims based on your general knowledge.

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