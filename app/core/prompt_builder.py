def build_prompt(
    query: str,
    context_chunks: list[str],
    history: list[dict],
    include_booking_instructions: bool = False,
) -> str:
    """Assemble the full LLM prompt from retrieved context, history, and query.

    Args:
        query: The user's current question.
        context_chunks: Top-k document chunks retrieved from Qdrant vector search.
        history: Prior conversation turns as [{role, content}, ...].
        include_booking_instructions: If True, inject booking JSON format into the prompt.

    Returns:
        A single formatted prompt string ready to send to the LLM.
    """
    if context_chunks:
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
        context_instruction = (
            "Answer using the document context below. "
            "You may also use the conversation history to understand the user's intent."
        )
    else:
        context_block = "No relevant documents found."
        context_instruction = (
            "No relevant documents were retrieved. "
            "Answer using the conversation history if available, or let the user know "
            "you need a document to be ingested first."
        )

    history_text = (
        "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
        if history
        else "No prior conversation."
    )

    booking_instructions = (
        "\nIf the user wants to book an interview and has provided their name, email, date, and time, "
        "respond with ONLY this JSON (no extra text, no markdown):\n"
        '{"intent": "booking", "name": "...", "email": "...", "date": "...", "time": "..."}\n'
        if include_booking_instructions
        else ""
    )

    return f"""You are a helpful assistant. {context_instruction}{booking_instructions}
--- DOCUMENT CONTEXT ---
{context_block}

--- CONVERSATION HISTORY ---
{history_text}

--- CURRENT QUESTION ---
User: {query}
Assistant:"""
