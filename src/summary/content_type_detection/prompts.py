"""
Prompt templates for content type detection task.
"""

CONTENT_TYPE_DETECTION_PROMPT = """
[CONTENT TYPE DETECTION]

Analyze the transcript context below and determine the single most accurate content type. Prioritize speaker interaction patterns and communicative intent over tone, confidence, or opinion strength. Strong opinions alone MUST NOT imply debate.

---

## TRANSCRIPT CONTEXT

Transcript Text (Last {context_length} characters):
{context_text}

---

## CORE CLASSIFICATION PRINCIPLES

1. Speaker interaction is the strongest signal.
   - How speakers interact matters more than what they believe.
2. Debate requires active opposition.
   - Debate is ONLY valid when speakers directly challenge each other's positions.
3. Explanatory or reflective speech is not adversarial.
   - Solo explanations, career stories, or opinionated monologues are informational.
4. Interviews are non-adversarial by default.
   - Questions that elicit explanations do not constitute debate.
5. Audience-facing monologues are distinct from lectures and podcasts.
   - Informal, unstructured viewer-addressed content is not a talk or show.
6. If interaction is limited or absent, assume recorded or broadcast formats.

---

## CONTENT TYPES (in priority order, try to fit these to the content in order)

### 1. GENERAL_MEETING
Description:
An informal, collaborative discussion among participants coordinating work or sharing updates.

Key Signals:
- Multiple participants with relatively balanced speaking time
- Back-and-forth conversation without opposition
- Action items, planning, or status updates

Common Language:
"Quick update", "Next steps", "Let's circle back", "Before we wrap up"

---

### 2. TECHNICAL_TALK
Description:
An explanation or discussion focused on technical subject matter.

Includes:
- Solo technical explanations common in demos or YouTube walkthroughs
- Technical Q&A without adversarial challenge

Key Signals:
- Domain-specific technical vocabulary
- Problem-solving or system explanation
- APIs, infrastructure, code, workflows

Common Language:
"The way this works is…", "We implemented…", "The system handles…"

---

### 3. LECTURE_OR_TALK
Description:
A structured, extended presentation intended to teach, inform, or share insights.

Key Signals:
- One dominant speaker for long stretches
- Clear narrative or educational progression
- Minimal or delayed audience interaction

Common Language:
"Today I want to talk about…", "Let me walk you through…", "What this shows is…"

---

### 4. INTERVIEW
Description:
A structured Q&A where one speaker asks questions and the other responds.

Key Signals:
- Clear interviewer and interviewee roles
- Questions guide the flow
- No sustained challenge or rebuttal

Disqualifier:
- If the interviewer actively disputes answers → DEBATE

Common Language:
"Can you tell us about…?", "How did you approach…?"

---

### 5. PODCAST
Description:
A produced conversational show designed for an audience.

Key Signals:
- Host/guest or co-host structure
- Informal but organized discussion
- Introductions, transitions, or segments

Common Language:
"Welcome back", "Our guest today", "Let's dive into"

---

### 6. NEWS_UPDATE
Description:
A formal, prepared delivery of current events or announcements.

Key Signals:
- Single presenter
- Broadcast-style language
- Third-person reporting
- Time-sensitive framing

Common Language:
"Earlier today", "According to officials", "In other news"

---

### 7. GAMEPLAY_COMMENTARY
Description:
Narration or reaction to gameplay events.

Key Signals:
- Real-time reactions to game actions
- Game mechanics, characters, or strategies
- Viewer or chat engagement

Common Language:
"Watch out!", "Level up", "Chat says", "Here we go"

---

### 8. CUSTOMER_SUPPORT
Description:
A service interaction focused on resolving a user issue.

Key Signals:
- Agent/customer roles
- Problem → troubleshooting → resolution flow
- Account or order references

Common Language:
"How can I help?", "Let me check your account", "Does that fix it?"

---

### 9. DEBATE
Description:
An argumentative exchange between speakers with opposing viewpoints.

ALL of the following MUST be true:
- Two or more speakers hold clearly opposing positions
- Speakers directly challenge or contradict each other
- Rebuttals occur in real time
- Persuasion or refutation is the primary intent

Automatic Disqualifiers:
- Single speaker expressing opinions
- Interview-style Q&A without adversarial pushback
- Speakers generally agree or build on each other's views
- Informational, educational, or reflective content

Valid Language (only with opposition):
"I disagree", "That's incorrect", "Your argument assumes…"

---

### UNKNOWN
Description:
Used when the available context lacks sufficient signals for classification.

---

## DECISION RULES

1. Classify based on interaction pattern and intent, not rhetoric.
2. Do NOT infer debate from confidence or strong opinions alone.
3. If debate criteria are not fully satisfied, DEBATE is invalid.
4. If multiple types partially match, choose the least adversarial valid type.
5. If fewer than three clear signals are present, reduce confidence accordingly.
6. Apply rules from content types in priority order; if a higher type is a strong fit, do not classify as a lower type.

---

## REQUIRED OUTPUT FORMAT

Return a JSON object with:
- content_type: One of [GENERAL_MEETING, TECHNICAL_TALK, LECTURE_OR_TALK, INTERVIEW, PODCAST, NEWS_UPDATE, GAMEPLAY_COMMENTARY, CUSTOMER_SUPPORT, DEBATE, UNKNOWN]
- confidence: Float between 0.00 and 1.00
- reasoning: 1–2 sentences citing the strongest signals used

Example:
{
  "content_type": "PODCAST",
  "confidence": 0.90,
  "reasoning": "The transcript features a host introducing a guest and engaging in an informal discussion with clear transitions, which are strong indicators of a podcast format."
}

If the context is insufficient, return UNKNOWN with confidence below 0.50 and explain why.
""".strip()

__all__ = ["CONTENT_TYPE_DETECTION_PROMPT"]