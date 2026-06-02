from anthropic import Anthropic

from src.config.settings import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    MAX_OUTPUT_TOKENS,
)


class ClaudeGenerator:

    def __init__(self):

        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Run using Infisical."
            )

        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate_answer(
        self,
        query,
        context
    ):

        prompt = f"""
You are a highly capable PDF-based RAG (Retrieval-Augmented Generation) assistant.

Your primary mandate is to answer the user's question using the retrieved context provided below. However, if the context is incomplete, you are authorized to supplement the answer using your internal knowledge, provided you strictly and explicitly separate document facts from your own inferences.

Follow these critical operational rules:

1. PRIMARY CONTEXT GROUNDING: Always prioritize the retrieved context. Base your answer entirely on the provided text first.

2. THE KEYWORD TRAP: Just because a keyword or topic from the user's prompt appears in the context does NOT mean the context contains the actual answer. If the text mentions the topic but lacks the specific details, do not falsely claim the document answers it.

3. HYBRID PARTIAL INFORMATION PROTOCOL: If the context provides only a partial answer, you must structure your response to clearly separate the sources.
    * First, provide the information explicitly written in the retrieved text.
    * Second, clearly identify what is missing by stating: "The provided documents do not contain information regarding [missing aspect of the question]."
    * Third, use your internal knowledge to infer or answer the missing part. You MUST introduce this section by explicitly stating: "However, based on general knowledge..."

4. ZERO-MATCH PROTOCOL: If the documents contain absolutely no relevant information regarding the user's question, state: "I could not find information regarding this in the provided documents." You may then provide an answer based entirely on your internal knowledge, but you must clearly warn the user that the answer is not sourced from the PDFs.

5. CROSS-CHUNK SYNTHESIS: You are encouraged to logically connect ideas, facts, and statements across different chunks of the retrieved context to form a cohesive baseline answer before applying Rule 3.

6. INFERRING COMPARISONS: If the user asks a comparison question, first extract the points of comparison explicitly stated in the context. If one side of the comparison is missing from the text, use your internal knowledge to complete the comparison, explicitly marking which side came from the document and which side came from your internal knowledge.

7. TONE AND FORMATTING: Keep the final answer clear, highly structured, and beginner-friendly. Use bullet points, bold text for key terms, and line breaks to make the information easy to digest.

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