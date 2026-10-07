"""Prompt text for grounded, cited answers.

Retrieval only measures topical similarity, so a chunk about the right topic
can still lack the answer. These prompts make the model refuse in that case.
"""

from llama_index.core.prompts import PromptTemplate

NO_RELEVANT_CONTEXT_MESSAGE = (
    "I couldn't find relevant information in the indexed lecture documents."
)

GROUNDING_RULES = f"""\
You answer questions using ONLY the numbered sources below.

Rules:
- Use only facts explicitly stated in the sources. Never use outside knowledge
  and never guess.
- Do not reinterpret an unrelated fact as the answer. For example, a currency
  is not an animal, and a city is not a person.
- Cite every fact with its source number, such as [1].
- If the sources do not explicitly contain the answer, reply with exactly this
  sentence and nothing else: "{NO_RELEVANT_CONTEXT_MESSAGE}"
"""

# Not f-strings: LlamaIndex fills these {placeholders} at query time.
CITATION_QA_TEMPLATE = PromptTemplate(
    GROUNDING_RULES
    + """
------
{context_str}
------
Question: {query_str}
Answer:"""
)

CITATION_REFINE_TEMPLATE = PromptTemplate(
    GROUNDING_RULES
    + """
Existing answer: {existing_answer}
Refine the existing answer using the additional sources below. If they do not
help, repeat the existing answer unchanged.
------
{context_msg}
------
Question: {query_str}
Answer:"""
)
