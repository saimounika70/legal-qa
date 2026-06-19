from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert legal assistant specialising in Indian law.
You help users understand Indian court judgments, RTI documents, and legal texts.

When answering:
1. Base your answer ONLY on the provided document excerpts
2. Cite specific parts by saying "According to the document..." or "The court held that..."
3. If the answer isn't in the excerpts, say so clearly — do not hallucinate
4. Use plain English — avoid unnecessary legal jargon
5. If relevant, mention the specific legal sections or case citations found in the text

You are helping ordinary citizens understand complex legal documents."""


def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    Include similarity scores so the model knows which chunks are most relevant.
    """
    context_parts = []
    for chunk in chunks:
        similarity_pct = int(chunk['similarity'] * 100)
        context_parts.append(
            f"[Excerpt {chunk['rank']} — {similarity_pct}% relevance]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


def answer_question(
    question: str,
    chunks: list[dict],
    chat_history: list[dict] = None
) -> dict:
    """
    Generate an answer using retrieved chunks as context.
    
    What's happening (RAG pipeline):
    1. chunks = retrieved from vector store (the R in RAG)
    2. We format them into a prompt (the A = Augmented)
    3. LLM generates answer grounded in those chunks (the G = Generation)
    
    The LLM never sees the full document — only the relevant chunks.
    This is why RAG works: the LLM can't hallucinate facts not in the chunks.
    """
    context = build_context(chunks)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # add chat history for multi-turn conversation
    if chat_history:
        for msg in chat_history[-6:]:  # last 3 exchanges
            messages.append(msg)
    
    # the actual prompt
    user_message = f"""Based on the following excerpts from the legal document, 
please answer this question: {question}

DOCUMENT EXCERPTS:
{context}

Please provide a clear, accurate answer based only on the above excerpts.
If you quote directly, use quotation marks."""
    
    messages.append({"role": "user", "content": user_message})
    
    # call Groq API (LLaMA 3 70B — free)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,    # low temperature = more factual, less creative
        max_tokens=1024,
    )
    
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "sources": chunks,
        "model": "llama-3.3-70b-versatile",
        "tokens_used": response.usage.total_tokens
    }


def generate_document_summary(chunks: list[dict]) -> str:
    """
    Generate a structured summary of the entire document.
    Uses the top chunks as representative content.
    """
    # use first 8 chunks for summary (overview of document)
    sample_text = "\n\n".join([c["content"] for c in chunks[:8]])
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Provide a structured summary of this Indian legal document.

Format your response as:
**Case/Document Type:** [what kind of document]
**Parties Involved:** [petitioner vs respondent if applicable]
**Core Issue:** [what legal question is being decided]
**Court's Decision:** [what was decided]
**Key Legal Points:** [3-5 bullet points of important holdings]
**Relevant Laws Cited:** [IPC sections, Articles, Acts mentioned]

DOCUMENT EXCERPTS:
{sample_text}"""}
        ],
        temperature=0.2,
        max_tokens=800,
    )
    
    return response.choices[0].message.content