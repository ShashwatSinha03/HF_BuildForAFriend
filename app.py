"""StudyLens - Local-first AI study assistant."""

import streamlit as st
from document import extract_text, get_file_type, retrieve_relevant_chunks, get_representative_chunks
from llm import verify_ollama_setup, DEFAULT_MODEL, OllamaConnectionError, generate_explanation, generate_revision, generate_quiz, evaluate_short_answers, generate_exam_plan


st.set_page_config(
    page_title="StudyLens",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def init_session_state():
    """Initialize session state for document storage."""
    if "document" not in st.session_state:
        st.session_state.document = {
            "content": "",
            "filename": "",
            "file_type": "",
            "size": 0,
        }
    if "ollama_ready" not in st.session_state:
        st.session_state.ollama_ready = False
    if "ollama_message" not in st.session_state:
        st.session_state.ollama_message = ""
    if "explain_answer" not in st.session_state:
        st.session_state.explain_answer = ""
    if "explain_question" not in st.session_state:
        st.session_state.explain_question = ""
    if "revision_sheet" not in st.session_state:
        st.session_state.revision_sheet = ""
    if "quiz" not in st.session_state:
        st.session_state.quiz = {
            "questions": [],
            "current": 0,
            "answers": [],
            "short_answers": [],
            "started": False,
            "completed": False,
            "score": 0,
            "weak_areas": [],
            "num_questions": 5,
        }
    if "exam" not in st.session_state:
        st.session_state.exam = {
            "plan": None,
            "minutes": 30,
            "generated": False,
        }


def check_ollama():
    """Check Ollama connection and model availability."""
    ready, message = verify_ollama_setup()
    st.session_state.ollama_ready = ready
    st.session_state.ollama_message = message


def render_header():
    """Render the application header."""
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #666;
        }
        .preview-box {
            background: #fafafa;
            border: 1px solid #eaeaea;
            border-radius: 8px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.85rem;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        .section-divider {
            margin: 1.5rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">StudyLens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Turn your study material into something you can actually study.</div>',
        unsafe_allow_html=True,
    )


def render_ollama_status():
    """Render Ollama connection status."""
    if st.session_state.ollama_message:
        if st.session_state.ollama_ready:
            st.success(f"✅ {st.session_state.ollama_message}")
        else:
            st.error(f"❌ {st.session_state.ollama_message}")
            st.info(
                f"**Default model:** `{DEFAULT_MODEL}`\n\n"
                "**To fix:** Run `ollama pull gemma3:1b` then restart Ollama with `ollama serve`"
            )


def render_uploader():
    """Render file uploader and handle upload."""
    uploaded_file = st.file_uploader(
        "Upload a study document",
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_type = get_file_type(uploaded_file.name)

        if file_type is None:
            st.error("Unsupported file type. Please upload a PDF or TXT file.")
            return

        with st.spinner("Extracting text..."):
            file_bytes = uploaded_file.read()
            try:
                content = extract_text(file_bytes, file_type)
            except Exception:
                st.error("Could not read this file. It may be corrupted or password-protected.")
                return

        if not content.strip():
            st.warning("This document appears to be empty or contains no readable text.")
            return

        # Store in session state
        st.session_state.document = {
            "content": content,
            "filename": uploaded_file.name,
            "file_type": file_type.upper(),
            "size": len(file_bytes),
        }
        # Clear previous explain/revise/quiz/exam answers when new document is uploaded
        st.session_state.explain_answer = ""
        st.session_state.explain_question = ""
        st.session_state.revision_sheet = ""
        st.session_state.quiz = {
            "questions": [],
            "current": 0,
            "answers": [],
            "short_answers": [],
            "started": False,
            "completed": False,
            "score": 0,
            "weak_areas": [],
            "num_questions": 5,
        }
        st.session_state.exam = {
            "plan": None,
            "minutes": 30,
            "generated": False,
        }
        st.success(f"✅ Loaded: {uploaded_file.name}")


def render_document_info():
    """Render document information and preview."""
    doc = st.session_state.document
    if not doc["content"]:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{doc['filename']}</div>
                <div class="metric-label">Filename</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{doc['file_type']}</div>
                <div class="metric-label">File Type</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        text_len = len(doc["content"])
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{text_len:,}</div>
                <div class="metric-label">Characters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Preview
    st.markdown("### Preview")
    preview = doc["content"][:1500]
    if len(doc["content"]) > 1500:
        preview += "\n\n... (truncated)"

    st.markdown(
        f'<div class="preview-box">{preview}</div>',
        unsafe_allow_html=True,
    )


def render_explain():
    """Render the Explain interface."""
    doc = st.session_state.document
    if not doc["content"]:
        return

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🧠 Understand")

    # Example questions
    st.caption("Example: \"What is TCP congestion control?\" · \"Explain this like I'm a beginner\" · \"Compare TCP and UDP\"")

    question = st.text_input(
        "Ask something about your material:",
        value=st.session_state.explain_question,
        placeholder="What is...?",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        explain_clicked = st.button("Explain", type="primary", use_container_width=True)

    if explain_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        if not st.session_state.ollama_ready:
            st.error("Ollama is not available. Please check the connection status above.")
            return

        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant context
                chunks = retrieve_relevant_chunks(doc["content"], question)
                if not chunks:
                    st.session_state.explain_answer = (
                        "No relevant content found in the document for this question. "
                        "Try rephrasing or asking about a different topic."
                    )
                else:
                    context = "\n\n---\n\n".join(chunks)
                    answer = generate_explanation(question, context)
                    if not answer.strip():
                        st.session_state.explain_answer = (
                            "The model returned an empty response. Please try again."
                        )
                    else:
                        st.session_state.explain_answer = answer
                st.session_state.explain_question = question
            except OllamaConnectionError:
                st.session_state.explain_answer = "Could not connect to Ollama. Please check that Ollama is running."
            except Exception:
                st.session_state.explain_answer = "Something went wrong. Please try again."

    # Display answer
    if st.session_state.explain_answer:
        st.markdown("#### Answer")
        st.markdown(st.session_state.explain_answer)


def render_revise():
    """Render the Revise interface."""
    doc = st.session_state.document
    if not doc["content"]:
        return

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📚 Revise")

    col1, col2 = st.columns([1, 5])
    with col1:
        revise_clicked = st.button(
            "Generate Revision Sheet",
            type="primary",
            use_container_width=True,
        )
    with col2:
        regen_clicked = st.button(
            "Regenerate",
            use_container_width=True,
            disabled=not st.session_state.revision_sheet,
        )

    if revise_clicked or regen_clicked:
        if not st.session_state.ollama_ready:
            st.error("Ollama is not available. Please check the connection status above.")
            return

        with st.spinner("Creating your revision sheet..."):
            try:
                # Get representative chunks from across the document
                chunks = get_representative_chunks(doc["content"])
                if not chunks:
                    st.session_state.revision_sheet = (
                        "Unable to create revision sheet: no content available."
                    )
                else:
                    context = "\n\n---\n\n".join(chunks)
                    sheet = generate_revision(context)
                    if not sheet.strip():
                        st.session_state.revision_sheet = (
                            "The model returned an empty response. Please try again."
                        )
                    else:
                        st.session_state.revision_sheet = sheet
            except OllamaConnectionError:
                st.session_state.revision_sheet = "Could not connect to Ollama. Please check that Ollama is running."
            except Exception:
                st.session_state.revision_sheet = "Something went wrong. Please try again."

    # Display revision sheet
    if st.session_state.revision_sheet:
        st.markdown(st.session_state.revision_sheet)


def render_exam():
    """Render the Exam Mode interface."""
    doc = st.session_state.document
    if not doc["content"]:
        return

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### ⏱️ Exam Mode")

    exam = st.session_state.exam

    # Not generated yet - show configuration
    if not exam["generated"]:
        st.caption("Prioritizes concepts based on your uploaded material. Not an exam predictor.")

        col1, col2 = st.columns([1, 4])
        with col1:
            minutes = st.selectbox(
                "Available time:",
                [15, 30, 60],
                index=1,
                key="exam_minutes",
            )
            exam["minutes"] = minutes

        if st.button("Start Exam Mode", type="primary", use_container_width=True):
            if not st.session_state.ollama_ready:
                st.error("Ollama is not available. Please check the connection status above.")
                return

            with st.spinner("Analyzing material and creating study plan..."):
                try:
                    chunks = get_representative_chunks(doc["content"])
                    if not chunks:
                        st.error("Unable to create study plan: no content available.")
                        return
                    
                    context = "\n\n---\n\n".join(chunks)
                    plan = generate_exam_plan(context, minutes=minutes)
                    
                    if not plan:
                        st.error("Could not generate a valid study plan from this material. Try again.")
                        return
                    
                    exam["plan"] = plan
                    exam["generated"] = True
                    st.rerun()
                except OllamaConnectionError:
                    st.error("Could not connect to Ollama. Please check that Ollama is running.")
                except Exception:
                    st.error("Something went wrong. Please try again.")
        return

    # Display generated plan
    plan = exam["plan"]
    if not plan:
        st.error("No study plan available.")
        return

    minutes = exam["minutes"]
    
    # Calculate totals
    p1_items = plan.get("priority_1", [])
    p2_items = plan.get("priority_2", [])
    p3_items = plan.get("priority_3", [])
    
    total_p1 = sum(item["minutes"] for item in p1_items)
    total_p2 = sum(item["minutes"] for item in p2_items)
    total_p3 = sum(item["minutes"] for item in p3_items)
    grand_total = total_p1 + total_p2 + total_p3

    st.markdown(f"### ⏱️ {minutes}-MINUTE STUDY PLAN")
    st.caption("Based on your uploaded material. Prioritization reflects content emphasis, not exam prediction.")
    
    st.markdown("---")

    # Priority 1 - Must Know
    if p1_items:
        st.markdown("#### 🔥 MUST KNOW")
        for item in p1_items:
            st.markdown(f"**{item['topic']}**  \n`~{item['minutes']} min`  \n*Why: {item['reason']}*")
            st.markdown("")

    # Priority 2 - Good to Know
    if p2_items:
        st.markdown("#### 📘 GOOD TO KNOW")
        for item in p2_items:
            st.markdown(f"**{item['topic']}**  \n`~{item['minutes']} min`  \n*Why: {item['reason']}*")
            st.markdown("")

    # Priority 3 - Skim
    if p3_items:
        st.markdown("#### ⏭️ SKIM IF NEEDED")
        for item in p3_items:
            st.markdown(f"**{item['topic']}**  \n`~{item['minutes']} min`  \n*Why: {item['reason']}*")
            st.markdown("")

    st.markdown("---")
    st.markdown(f"**Total recommended time:** {grand_total} min")

    # Action buttons - connect to existing features
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("New Plan", use_container_width=True):
            exam["plan"] = None
            exam["generated"] = False
            st.rerun()
    with col2:
        if st.button("Quick Revision", use_container_width=True):
            # Reuse existing Revise functionality - we can set a flag or just let user click Revise
            st.info("Scroll to the **Revise** section above and click 'Generate Revision Sheet'")
    with col3:
        if st.button("Quick Quiz", use_container_width=True):
            st.info("Scroll to the **Quiz Me** section above and click 'Start Quiz'")


def render_quiz():
    """Render the Quiz Me interface."""
    doc = st.session_state.document
    if not doc["content"]:
        return

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Quiz Me")

    quiz = st.session_state.quiz

    # Quiz not started - show configuration
    if not quiz["started"]:
        col1, col2 = st.columns([1, 4])
        with col1:
            num_q = st.selectbox(
                "Questions:",
                [5, 10],
                index=0,
                key="quiz_num_questions",
            )
            quiz["num_questions"] = num_q

        if st.button("Start Quiz", type="primary", use_container_width=True):
            if not st.session_state.ollama_ready:
                st.error("Ollama is not available. Please check the connection status above.")
                return

            with st.spinner("Generating quiz..."):
                try:
                    chunks = get_representative_chunks(doc["content"])
                    if not chunks:
                        st.error("Unable to generate quiz: no content available.")
                        return
                    
                    context = "\n\n---\n\n".join(chunks)
                    quiz_data = generate_quiz(context, num_questions=num_q)
                    
                    if not quiz_data or not quiz_data.get("questions"):
                        st.error("Could not generate a valid quiz from this material. Try again.")
                        return
                    
                    # Initialize quiz state
                    questions = quiz_data["questions"]
                    quiz["questions"] = questions
                    quiz["current"] = 0
                    quiz["answers"] = [-1] * len(questions)
                    quiz["short_answers"] = [""] * len(questions)
                    quiz["started"] = True
                    quiz["completed"] = False
                    quiz["score"] = 0
                    quiz["weak_areas"] = []
                    st.rerun()
                except OllamaConnectionError:
                    st.error("Could not connect to Ollama. Please check that Ollama is running.")
                except Exception:
                    st.error("Something went wrong. Please try again.")
        return

    # Quiz completed - show results
    if quiz["completed"]:
        render_quiz_results()
        return

    # Quiz in progress - show current question
    questions = quiz["questions"]
    current = quiz["current"]
    total = len(questions)
    
    if current >= total:
        quiz["completed"] = True
        st.rerun()
        return

    st.markdown(f"**Question {current + 1} / {total}**")
    st.progress((current) / total)
    
    q = questions[current]
    
    if q["type"] == "mcq":
        render_mcq_question(q, current)
    else:
        render_short_answer_question(q, current)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if current > 0:
            if st.button("← Previous", use_container_width=True):
                quiz["current"] -= 1
                st.rerun()
    with col3:
        if current < total - 1:
            if st.button("Next →", use_container_width=True):
                quiz["current"] += 1
                st.rerun()
        else:
            # Last question - show submit
            if st.button("Submit Quiz", type="primary", use_container_width=True):
                submit_quiz()


def render_mcq_question(q: dict, index: int):
    """Render a multiple choice question."""
    st.markdown(f"**{q['question']}**")
    
    options = q["options"]
    selected = st.session_state.quiz["answers"][index]
    
    choice = st.radio(
        "Select an answer:",
        options,
        index=selected if selected >= 0 else None,
        key=f"mcq_{index}",
        label_visibility="collapsed",
    )
    
    if choice:
        st.session_state.quiz["answers"][index] = options.index(choice)


def render_short_answer_question(q: dict, index: int):
    """Render a short answer question."""
    st.markdown(f"**{q['question']}**")
    
    answer = st.text_area(
        "Your answer:",
        value=st.session_state.quiz["short_answers"][index],
        key=f"short_{index}",
        label_visibility="collapsed",
        height=100,
    )
    
    st.session_state.quiz["short_answers"][index] = answer


def submit_quiz():
    """Process quiz submission and calculate score."""
    quiz = st.session_state.quiz
    questions = quiz["questions"]
    answers = quiz["answers"]
    short_answers = quiz["short_answers"]
    
    score = 0
    weak_areas = []
    short_q_indices = []
    short_q_list = []
    user_short_answers = []
    
    # Evaluate MCQs
    for i, q in enumerate(questions):
        if q["type"] == "mcq":
            if answers[i] == q["answer"]:
                score += 1
            else:
                weak_areas.append(q["topic"])
        else:
            # Collect short answer questions for batch evaluation
            short_q_indices.append(i)
            short_q_list.append(q)
            user_short_answers.append(short_answers[i])
    
    # Evaluate short answers in batch
    if short_q_list:
        try:
            results = evaluate_short_answers(short_q_list, user_short_answers)
            for idx, (q_idx, is_correct) in enumerate(zip(short_q_indices, results)):
                if is_correct:
                    score += 1
                else:
                    weak_areas.append(short_q_list[idx]["topic"])
        except Exception:
            # If evaluation fails, mark all short answers as incorrect
            for q in short_q_list:
                weak_areas.append(q["topic"])
    
    quiz["score"] = score
    quiz["weak_areas"] = list(dict.fromkeys(weak_areas))  # Deduplicate while preserving order
    quiz["completed"] = True
    st.rerun()


def render_quiz_results():
    """Render quiz results screen."""
    quiz = st.session_state.quiz
    questions = quiz["questions"]
    answers = quiz["answers"]
    short_answers = quiz["short_answers"]
    score = quiz["score"]
    total = len(questions)
    weak_areas = quiz["weak_areas"]
    
    st.markdown("### 🎯 Quiz Complete")
    
    # Score display
    pct = int((score / total) * 100) if total > 0 else 0
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: #f0f4f8; border-radius: 12px; margin: 1rem 0;">
        <div style="font-size: 3rem; font-weight: 700; color: #2c3e50;">{score} / {total}</div>
        <div style="font-size: 1.5rem; color: #666;">{pct}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Weak areas
    if weak_areas:
        st.markdown("#### Weak Areas")
        st.markdown(", ".join(f"`{area}`" for area in weak_areas))
    else:
        st.markdown("#### No obvious weak areas 🎉")
    
    st.divider()
    
    # Review each question
    st.markdown("#### Review")
    for i, q in enumerate(questions):
        is_correct = False
        user_ans = ""
        
        if q["type"] == "mcq":
            is_correct = (answers[i] == q["answer"])
            user_ans = q["options"][answers[i]] if answers[i] >= 0 else "Not answered"
            correct_ans = q["options"][q["answer"]]
        else:
            # For short answers, we'd need to re-evaluate or store results
            # For now, show the expected answer
            user_ans = short_answers[i] or "Not answered"
            # We can't easily know correctness here without re-evaluating
            # Just show the expected answer for review
            is_correct = None
        
        icon = "✓" if is_correct is True else ("✗" if is_correct is False else "○")
        st.markdown(f"**{icon} Question {i+1}: {q['question']}**")
        st.caption(f"Topic: {q['topic']}")
        
        if q["type"] == "mcq":
            if is_correct:
                st.success(f"Your answer: {user_ans}")
            else:
                st.error(f"Your answer: {user_ans}")
                st.info(f"Correct: {correct_ans}")
        else:
            st.write(f"Your answer: {user_ans}")
            st.info(f"Expected: {q['expected_answer']}")
        
        st.caption(f"Explanation: {q['explanation']}")
        st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retake Quiz", use_container_width=True):
            reset_quiz()
            st.rerun()
    with col2:
        if st.button("New Quiz", use_container_width=True):
            reset_quiz()
            st.rerun()


def reset_quiz():
    """Reset quiz state for a new attempt."""
    st.session_state.quiz = {
        "questions": [],
        "current": 0,
        "answers": [],
        "short_answers": [],
        "started": False,
        "completed": False,
        "score": 0,
        "weak_areas": [],
        "num_questions": st.session_state.quiz.get("num_questions", 5),
    }


def main():
    init_session_state()

    # Check Ollama on first load
    if not st.session_state.ollama_message:
        check_ollama()

    render_header()
    render_ollama_status()

    st.divider()

    # File upload section
    st.markdown("### 📄 Upload Document")
    render_uploader()

    if st.session_state.document["content"]:
        st.divider()
        render_document_info()
        render_explain()
        render_revise()
        render_quiz()
        render_exam()


if __name__ == "__main__":
    main()