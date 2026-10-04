"""Ollama connection utilities for StudyLens."""

import json
import re
import requests


DEFAULT_MODEL = "gemma3:1b"
OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaConnectionError(Exception):
    """Raised when Ollama connection fails."""
    pass


class ModelNotFoundError(Exception):
    """Raised when the required model is not installed."""
    pass


def _extract_json_from_response(response: str) -> str:
    """
    Extract JSON from model response, handling markdown code fences.
    Returns the cleaned JSON string.
    """
    json_str = response.strip()
    # Remove markdown code fences
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    return json_str.strip()


def _try_parse_json(json_str: str) -> dict | list | None:
    """
    Try to parse JSON with multiple strategies.
    Returns parsed object or None if all strategies fail.
    """
    # Strategy 1: Try as-is
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Fix trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix duplicate keys (keep last occurrence)
    lines = json_str.split('\n')
    seen_keys = set()
    fixed_lines = []
    for line in lines:
        match = re.match(r'^(\s*)"([^"]+)":', line)
        if match:
            indent, key = match.groups()
            if key in seen_keys:
                continue
            seen_keys.add(key)
        fixed_lines.append(line)
    fixed = '\n'.join(fixed_lines)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Normalize quotes
    fixed = json_str.replace("'", '"')
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


EXPLAIN_SYSTEM_PROMPT = """You are a study assistant. Explain concepts using ONLY the provided study material.

Rules:
- Use only the supplied material; if it lacks information, say so
- Be concise and educational
- Preserve technical terminology
- Do not invent facts or use outside knowledge
- Use bullets for comparisons when helpful
- No unnecessary introductions or conclusions"""


EXPLAIN_USER_TEMPLATE = """Study material:
{context}

Question: {question}

Answer:"""


REVISE_SYSTEM_PROMPT = """You are a study assistant. Create a concise, exam-friendly revision sheet using ONLY the provided study material.

Rules:
- Identify key concepts, terms, and relationships from the material
- Prioritize emphasized, repeated, or defined information
- Preserve technical terminology exactly as used
- Each explanation: 1-3 sentences maximum
- Do not invent facts or include unsupported information
- Omit unsupported sections
- Return clean Markdown with clear headings"""


REVISE_USER_TEMPLATE = """Study material:
{context}

Create a compact revision sheet with these sections (only include supported ones):

## Important Concepts
### Concept Name
1-3 sentence explanation...

## Key Terms
- **Term:** Brief definition from the material

## Important Relationships
- **A vs B:** Brief comparison from the material

## Remember
- High-value fact, formula, or mechanism

## Common Confusions
- Easily confused concept with clarification
"""


QUIZ_SYSTEM_PROMPT = """You are a study assistant. Generate a quiz using ONLY the provided study material.

Rules:
- Use only the supplied material; do not invent facts
- Test understanding, not word matching
- Exactly one correct MCQ option
- Questions: concise, unambiguous, with topic
- No duplicate questions
- If material is thin, generate fewer high-quality questions
- Return valid JSON only - no extra commentary"""


QUIZ_USER_TEMPLATE = """Study material:
{context}

Generate a quiz with {num_questions} questions ({num_mcq} MCQs, {num_short} short-answer).

Return ONLY this JSON structure:
{{
  "questions": [
    {{
      "type": "mcq",
      "question": "Question text?",
      "options": ["A", "B", "C", "D"],
      "answer": 0,
      "explanation": "Brief explanation",
      "topic": "Topic name"
    }},
    {{
      "type": "short",
      "question": "Question text?",
      "expected_answer": "Expected answer",
      "explanation": "Brief explanation",
      "topic": "Topic name"
    }}
  ]
}}

Rules:
- EXACTLY {num_questions} questions ({num_mcq} MCQs, {num_short} short)
- MCQ: 4 options array, answer = 0/1/2/3
- Every question has "topic" and "explanation"
- Short answers need "expected_answer"
- No duplicate questions
- Output ONLY the JSON object
"""


def check_ollama_running(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Check if Ollama server is reachable."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """List installed models in Ollama."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except requests.RequestException:
        pass
    return []


def is_model_installed(model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL) -> bool:
    """Check if a specific model is installed."""
    models = list_models(base_url)
    return any(model in m for m in models)


def verify_ollama_setup(model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL) -> tuple[bool, str]:
    """
    Verify Ollama is running and the model is installed.
    Returns (success, message).
    """
    if not check_ollama_running(base_url):
        return False, (
            "Ollama is not running. Please start Ollama first:\n"
            "  - macOS/Linux: Run 'ollama serve' in a terminal\n"
            "  - Windows: Start Ollama from the Start menu"
        )

    if not is_model_installed(model, base_url):
        return False, (
            f"Model '{model}' is not installed. Please pull it first:\n"
            f"  ollama pull {model}"
        )

    return True, "Ollama is ready"


def generate_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Generate a completion from Ollama."""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.RequestException as e:
        raise OllamaConnectionError(f"Failed to generate completion: {e}")


def generate_explanation(
    question: str,
    context: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """
    Generate an explanation grounded in the provided context.
    Returns a message if no context is available.
    """
    if not context.strip():
        return "No relevant content found in the document for this question. Try rephrasing or asking about a different topic."
    
    prompt = f"{EXPLAIN_SYSTEM_PROMPT}\n\n{EXPLAIN_USER_TEMPLATE.format(context=context, question=question)}"
    return generate_completion(
        prompt,
        model=model,
        base_url=base_url,
        temperature=0.3,
        max_tokens=512,
    )


def generate_revision(
    context: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """
    Generate a revision sheet grounded in the provided context.
    Returns a message if no context is available.
    """
    if not context.strip():
        return "No content available to create a revision sheet."
    
    prompt = f"{REVISE_SYSTEM_PROMPT}\n\n{REVISE_USER_TEMPLATE.format(context=context)}"
    return generate_completion(
        prompt,
        model=model,
        base_url=base_url,
        temperature=0.2,
        max_tokens=768,
    )


def _extract_questions_regex(json_str: str) -> dict | None:
    """
    Extract questions from severely malformed JSON using regex.
    This is a last-resort fallback.
    """
    import re
    
    # Try to find question objects in the text
    questions = []
    
    # Find all type declarations
    type_matches = list(re.finditer(r'"type"\s*:\s*"(mcq|short)"', json_str))
    if not type_matches:
        return None
    
    for i, match in enumerate(type_matches):
        qtype = match.group(1)
        start = match.start()
        # Find the end of this question object (next type or end of array)
        end = type_matches[i + 1].start() if i + 1 < len(type_matches) else len(json_str)
        obj_text = json_str[start:end]
        
        # Extract fields using regex
        question_match = re.search(r'"question"\s*:\s*"([^"]*)"', obj_text)
        if not question_match:
            continue
        question = question_match.group(1)
        
        if qtype == "mcq":
            # Extract options - handle mixed quote styles
            options_match = re.search(r'"options"\s*:\s*\[(.*?)\]', obj_text, re.DOTALL)
            if not options_match:
                continue
            options_str = options_match.group(1)
            # Extract individual options - handle both single and double quotes
            opts = re.findall(r'["\']([^"\']*)["\']', options_str)
            if len(opts) != 4:
                continue
            
            # Extract answer - handle letter (A-D) or number (0-3)
            answer_match = re.search(r'"answer"\s*:\s*([0-3]|["\'][A-D]["\'])', obj_text)
            if not answer_match:
                continue
            answer_str = answer_match.group(1).strip('"\'')
            if answer_str in "ABCD":
                answer = "ABCD".index(answer_str)
            else:
                try:
                    answer = int(answer_str)
                except ValueError:
                    continue
            if not 0 <= answer <= 3:
                continue
            
            # Extract explanation
            expl_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', obj_text)
            explanation = expl_match.group(1) if expl_match else ""
            
            # Extract topic
            topic_match = re.search(r'"topic"\s*:\s*"([^"]*)"', obj_text)
            topic = topic_match.group(1) if topic_match else "General"
            
            questions.append({
                "type": "mcq",
                "question": question,
                "options": opts,
                "answer": answer,
                "explanation": explanation,
                "topic": topic,
            })
        
        elif qtype == "short":
            # Extract expected_answer
            ea_match = re.search(r'"expected_answer"\s*:\s*"([^"]*)"', obj_text)
            if not ea_match:
                continue
            expected_answer = ea_match.group(1)
            
            # Extract explanation
            expl_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', obj_text)
            explanation = expl_match.group(1) if expl_match else "Answer based on the study material."
            
            # Extract topic
            topic_match = re.search(r'"topic"\s*:\s*"([^"]*)"', obj_text)
            topic = topic_match.group(1) if topic_match else "General"
            
            questions.append({
                "type": "short",
                "question": question,
                "expected_answer": expected_answer,
                "explanation": explanation,
                "topic": topic,
            })
    
    if not questions:
        return None
    
    return {"questions": questions}


def parse_quiz_response(response: str) -> dict | None:
    """
    Parse and validate quiz JSON response from the model.
    Returns parsed dict or None if invalid.
    """
    if not response.strip():
        return None
    
    # Extract JSON from response
    json_str = _extract_json_from_response(response)
    
    # Try to parse with multiple strategies
    data = _try_parse_json(json_str)
    
    # Strategy 5: Regex-based extraction for severely malformed JSON
    if data is None:
        data = _extract_questions_regex(json_str)
    
    if data is None:
        return None
    
    # Validate structure
    if not isinstance(data, dict) or "questions" not in data:
        return None
    
    questions = data["questions"]
    if not isinstance(questions, list) or not questions:
        return None
    
    validated = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        qtype = q.get("type")
        if qtype == "mcq":
            # Validate MCQ - be strict on required fields
            required = ("question", "options", "answer", "explanation", "topic")
            if not all(k in q for k in required):
                continue
            if not isinstance(q["options"], list) or len(q["options"]) != 4:
                continue
            if not isinstance(q["answer"], int) or not (0 <= q["answer"] <= 3):
                continue
            # Ensure all options are strings
            if not all(isinstance(opt, str) for opt in q["options"]):
                continue
            validated.append({
                "type": "mcq",
                "question": str(q["question"]),
                "options": [str(opt) for opt in q["options"]],
                "answer": q["answer"],
                "explanation": str(q["explanation"]),
                "topic": str(q["topic"]),
            })
        elif qtype == "short":
            # Validate short answer - provide defaults for missing fields
            required = ("question", "expected_answer")
            if not all(k in q for k in required):
                continue
            validated.append({
                "type": "short",
                "question": str(q["question"]),
                "expected_answer": str(q["expected_answer"]),
                "explanation": str(q.get("explanation", "Answer based on the study material.")),
                "topic": str(q.get("topic", "General")),
            })
    
    if not validated:
        return None
    
    return {"questions": validated}


def generate_quiz(
    context: str,
    num_questions: int = 5,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> dict | None:
    """
    Generate a quiz from the provided context.
    Returns parsed quiz dict or None on failure.
    """
    if not context.strip():
        return None
    
    # Determine MCQ/short split
    num_mcq = max(1, int(num_questions * 0.6))
    num_short = num_questions - num_mcq
    
    prompt = f"{QUIZ_SYSTEM_PROMPT}\n\n{QUIZ_USER_TEMPLATE.format(
        context=context,
        num_questions=num_questions,
        num_mcq=num_mcq,
        num_short=num_short,
    )}"
    
    response = generate_completion(
        prompt,
        model=model,
        base_url=base_url,
        temperature=0.3,
        max_tokens=1024,
    )
    
    return parse_quiz_response(response)


SHORT_ANSWER_EVAL_PROMPT = """You are a strict evaluator. Compare each user answer to the expected answer.

For each question, output EXACTLY ONE LINE: CORRECT or INCORRECT.

Rules:
- CORRECT only if user answer captures the SAME core concept
- INCORRECT if wrong, incomplete, or different concept
- Do NOT be generous - be precise
- Minor wording differences OK, wrong concepts NOT OK

Examples:
Expected: "Flow control prevents the sender from overwhelming the receiver."
User: "Flow control stops the sender from sending too much data." → CORRECT
User: "Flow control makes data go faster." → INCORRECT
User: "Flow control uses a sliding window." → INCORRECT

Expected: "DNS translates domain names into IP addresses."
User: "DNS converts domain names to IP addresses." → CORRECT
User: "DNS makes websites load faster." → INCORRECT
User: "DNS is a database." → INCORRECT

Questions:
{questions}

Output (one per line):
CORRECT
INCORRECT
..."""


def evaluate_short_answers(
    questions: list[dict],
    user_answers: list[str],
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> list[bool]:
    """
    Evaluate multiple short answers in a single LLM call.
    Returns list of booleans (True = correct).
    """
    if not questions or not user_answers:
        return []
    
    # Build evaluation prompt
    q_lines = []
    for i, (q, ua) in enumerate(zip(questions, user_answers)):
        q_lines.append(
            f"Q{i+1}: {q['question']}\n"
            f"Expected: {q['expected_answer']}\n"
            f"User: {ua}\n"
        )
    
    prompt = SHORT_ANSWER_EVAL_PROMPT.format(questions="\n".join(q_lines))
    
    try:
        response = generate_completion(
            prompt,
            model=model,
            base_url=base_url,
            temperature=0.1,
            max_tokens=256,
        )
        
        results = []
        for line in response.strip().split("\n"):
            line = line.strip().upper()
            if line.startswith("CORRECT"):
                results.append(True)
            elif line.startswith("INCORRECT"):
                results.append(False)
        
        # If we got fewer results than questions, try evaluating each separately
        if len(results) < len(questions):
            return _evaluate_short_answers_individually(questions, user_answers, model, base_url)
        
        # Pad or truncate to match expected length
        while len(results) < len(questions):
            results.append(False)
        return results[:len(questions)]
    except Exception:
        # Fallback: mark all incorrect on evaluation failure
        return [False] * len(questions)


def _evaluate_short_answers_individually(
    questions: list[dict],
    user_answers: list[str],
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> list[bool]:
    """Fallback: evaluate each short answer in a separate call."""
    results = []
    for q, ua in zip(questions, user_answers):
        single_prompt = f"""You are a strict evaluator. Compare the user answer to the expected answer.

Question: {q['question']}
Expected: {q['expected_answer']}
User: {ua}

Output exactly one word: CORRECT or INCORRECT"""
        
        try:
            response = generate_completion(
                single_prompt,
                model=model,
                base_url=base_url,
                temperature=0.1,
                max_tokens=16,
            )
            line = response.strip().upper()
            if line.startswith("CORRECT"):
                results.append(True)
            else:
                results.append(False)
        except Exception:
            results.append(False)
    
    return results


EXAM_SYSTEM_PROMPT = """You are a study assistant. Create a prioritized study plan using ONLY the provided study material.

Rules:
- Analyze only the supplied material
- Identify concepts, terms, mechanisms
- Prioritize by signals in material: repetition, depth, definitions, formulas, dependencies, emphasis
- Do NOT predict exam questions - prioritize based on material evidence only
- Categorize into 3 priority levels
- Estimate study time per concept (1 sentence reason)
- Adapt to available time
- Return valid JSON only - no extra commentary"""


EXAM_USER_TEMPLATE = """Study material:
{context}

The user has {minutes} minutes to study. Create a prioritized study plan.

Return ONLY this JSON structure:
{{
  "priority_1": [{{"topic": "Concept", "reason": "Why high priority (from material)", "minutes": 8}}],
  "priority_2": [{{"topic": "Concept", "reason": "Why medium priority", "minutes": 4}}],
  "priority_3": [{{"topic": "Concept", "reason": "Why lower priority", "minutes": 2}}]
}}

Rules:
- Priority 1 (Must Know): Repeated, detailed, foundational concepts
- Priority 2 (Good to Know): Supporting concepts, definitions, mechanisms
- Priority 3 (Skim): Background, less detailed, tangential
- Total time ~{minutes} minutes
- Each item: topic (string), reason (string), minutes (integer)
- If material is thin, generate fewer items
- Output ONLY the JSON object
"""


def parse_exam_response(response: str) -> dict | None:
    """
    Parse and validate exam mode JSON response from the model.
    Returns parsed dict or None if invalid.
    """
    if not response.strip():
        return None
    
    # Extract JSON from response
    json_str = _extract_json_from_response(response)
    
    # Try to parse with multiple strategies
    data = _try_parse_json(json_str)
    
    if data is None:
        return None
    
    # Validate structure
    if not isinstance(data, dict):
        return None
    
    validated = {"priority_1": [], "priority_2": [], "priority_3": []}
    
    for priority_key in ["priority_1", "priority_2", "priority_3"]:
        items = data.get(priority_key, [])
        if not isinstance(items, list):
            continue
        
        for item in items:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("topic", "reason", "minutes")):
                continue
            topic = str(item["topic"]).strip()
            reason = str(item["reason"]).strip()
            minutes = item["minutes"]
            if not topic or not reason:
                continue
            try:
                minutes = int(minutes)
                if minutes <= 0:
                    continue
            except (ValueError, TypeError):
                continue
            validated[priority_key].append({
                "topic": topic,
                "reason": reason,
                "minutes": minutes,
            })
    
    # Must have at least something in priority_1
    if not validated["priority_1"]:
        return None
    
    return validated


def generate_exam_plan(
    context: str,
    minutes: int = 30,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> dict | None:
    """
    Generate a prioritized exam study plan from the provided context.
    Returns parsed plan dict or None on failure.
    """
    if not context.strip():
        return None
    
    prompt = f"{EXAM_SYSTEM_PROMPT}\n\n{EXAM_USER_TEMPLATE.format(context=context, minutes=minutes)}"
    
    response = generate_completion(
        prompt,
        model=model,
        base_url=base_url,
        temperature=0.3,
        max_tokens=768,
    )
    
    return parse_exam_response(response)