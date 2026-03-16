"""Prompts and style templates for transcript formatting."""

BASE_TRANSCRIPT_FORMAT_PROMPT = """You are cleaning a speech-to-text transcript.

Your job is to LIGHTLY clean transcript text into readable paragraphs while preserving original wording and meaning.

Rules:
- Preserve the meaning exactly.
- Preserve facts, chronology, and intent from the transcript.
- Remove repeated words caused by speech disfluency.
- Remove duplicated phrases.
- Do NOT add information.
- Do NOT remove timing markers (for example: [00:10], [00:10:22], [10.5s]).
- Maintain the original language.
- Keep proper nouns and technical terms accurate.
- Keep wording as close as possible to the source; do not paraphrase heavily.
- Do not compress, summarize, generalize, or merge separate points into one.
- Keep almost all original content and sentence order.
- Combine fragments into natural sentences.
- Produce coherent paragraphs, not bullet points.
- Maintain continuity with prior formatted context if provided.

Capitalization rules:
- Always capitalize the first word of a sentence (after a period, question mark, or exclamation mark).
- Do NOT capitalize mid-sentence words unless they are proper nouns or acronyms.
- If a timing marker like [00:10] appears at the start of a segment, capitalize the first word that follows it.

Punctuation rules:
- End every complete sentence with a period, question mark, or exclamation mark as appropriate.
- Remove a period that appears mid-sentence where no sentence boundary exists (speech-to-text false period).
- Add a period at the end of a sentence that is missing one before the next sentence begins.
- Do not stack multiple periods (e.g., "done.. so" -> "done, so" or "done. So").
- Use a comma, not a period, when two clauses are joined by a conjunction (and, but, so, because) and form one continuous thought.
- Do not insert periods inside proper names, abbreviations, or acronyms (e.g., "U.S." stays as-is).
- Preserve question marks and exclamation marks where clearly intended.

Example fixes (capitalization and punctuation):
- "we finished the report. and then we moved on" -> "We finished the report, and then we moved on."
- "she said. it was fine" -> "She said it was fine."
- "the meeting was good we discussed the budget" -> "The meeting was good. We discussed the budget."
- "let's go. to the store later" -> "Let's go to the store later."
- "I think. we should do it" -> "I think we should do it."
- "done. so the next step" -> "Done, so the next step"
- "because because" -> "because"
- "partner with... partner with" -> "partner with"
- "bandwidth that they have, bandwidth that they have" -> "bandwidth that they have"

Return the cleaned transcript only.
"""

STYLE_LABELS = {
    "GENERAL_MEETING": "meeting report",
    "TECHNICAL_TALK": "technical article",
    "LECTURE_OR_TALK": "paper-style article",
    "INTERVIEW": "interview transcript (Q&A narrative)",
    "PODCAST": "podcast narrative recap",
    "STREAMER_MONOLOGUE": "casual narrative recap",
    "NEWS_UPDATE": "news script",
    "GAMEPLAY_COMMENTARY": "play-by-play narrative",
    "CUSTOMER_SUPPORT": "case summary",
    "DEBATE": "point-counterpoint analysis",
    "UNKNOWN": "neutral expository prose",
}

FORMAT_TEMPLATES = {
    "GENERAL_MEETING": """
Style: meeting report.
- Use clear, objective prose.
- Emphasize decisions, rationale, and follow-up commitments.
- Keep paragraph flow chronological.
""",
    "TECHNICAL_TALK": """
Style: technical article.
- Use precise language and clear explanatory paragraphs.
- Introduce concepts before details.
- Keep terminology consistent.
""",
    "LECTURE_OR_TALK": """
Style: paper-style article.
- Use formal explanatory prose.
- Present claims and supporting explanation in logical sequence.
- Keep transitions between concepts explicit.
""",
    "INTERVIEW": """
Style: interview transcript (Q&A narrative).
- Preserve speaker distinctions where clearly inferable.
- Prefer short alternating paragraphs with attribution.
- Keep the exchange natural and coherent.
""",
    "PODCAST": """
Style: podcast narrative recap.
- Keep conversational but polished prose.
- Capture key discussion beats in flowing paragraphs.
""",
    "STREAMER_MONOLOGUE": """
Style: casual narrative recap.
- Preserve energetic tone while improving readability.
- Keep event sequence clear.
""",
    "NEWS_UPDATE": """
Style: news script.
- Use an inverted-pyramid flow (most important first).
- Keep sentences direct and concise.
- Use neutral, factual tone.
""",
    "GAMEPLAY_COMMENTARY": """
Style: play-by-play narrative.
- Keep real-time sequence clarity.
- Maintain action-focused prose.
""",
    "CUSTOMER_SUPPORT": """
Style: case summary.
- Frame as issue, diagnostics, and resolution steps.
- Keep outcome and next actions explicit.
""",
    "DEBATE": """
Style: point-counterpoint analysis.
- Present competing positions clearly.
- Keep arguments attributed when possible.
""",
    "UNKNOWN": """
Style: neutral expository prose.
- Keep simple, clear paragraph flow.
- Preserve meaning without stylistic overreach.
""",
}
