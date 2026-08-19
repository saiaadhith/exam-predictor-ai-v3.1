import io
import re
import json
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import pdfplumber

# ============================================================
# OPTIONAL LIBRARIES
# ============================================================

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ANN_OK = True
except Exception:
    ANN_OK = False

try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

try:
    from supabase import create_client
    SUPABASE_OK = True
except Exception:
    SUPABASE_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    PDF_OK = True
except Exception:
    PDF_OK = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ExamPredict AI V6",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 8% 5%,
            rgba(0,220,255,0.18),
            transparent 24%
        ),
        radial-gradient(
            circle at 92% 10%,
            rgba(155,80,255,0.22),
            transparent 27%
        ),
        linear-gradient(
            135deg,
            #070914,
            #11172d 48%,
            #080d1d
        );

    color: #f7f8ff;
}

.block-container {
    max-width: 1250px;
    padding-top: 1rem;
}

.hero,
.card,
.metric {

    border: 1px solid rgba(255,255,255,0.12);

    background:
        rgba(255,255,255,0.06);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.32);

    border-radius: 24px;
}

.hero {

    padding: 40px;

    margin-bottom: 24px;
}

.card {

    padding: 20px;
}

.metric {

    padding: 20px;

    min-height: 110px;
}

.metric-label {

    color: #aeb7d4;

    font-size: 0.75rem;

    font-weight: 800;
}

.metric-value {

    font-size: 1.45rem;

    font-weight: 900;

    margin-top: 8px;
}

.badge {

    display: inline-block;

    padding: 7px 12px;

    margin:
        0
        6px
        7px
        0;

    border-radius: 999px;

    background:
        rgba(255,255,255,0.07);

    border:
        1px solid
        rgba(255,255,255,0.12);

    font-size: 0.74rem;

    font-weight: 800;
}

.hero-title {

    font-size:
        clamp(
            2.5rem,
            6vw,
            4.3rem
        );

    font-weight: 900;

    letter-spacing: -2px;

    margin:
        14px
        0
        8px;
}

.hero-subtitle {

    color: #cbd2e9;

    font-size: 1.08rem;

    line-height: 1.65;

    max-width: 900px;
}

.disclaimer {

    padding: 18px;

    border-radius: 16px;

    background:
        rgba(255,190,70,0.09);

    border:
        1px solid
        rgba(255,190,70,0.30);

    color: #f3dfaa;

    margin-top: 24px;
}

.stButton > button {

    border-radius: 14px;

    min-height: 46px;

    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# AUTHENTICATION
# ============================================================

def has_google_auth():

    try:

        auth = st.secrets["auth"]

        return bool(
            auth.get("client_id")
            and
            auth.get("client_secret")
            and
            auth.get("server_metadata_url")
        )

    except Exception:

        return False


if has_google_auth():

    if not st.user.is_logged_in:

        st.html("""
        <div class="hero" style="text-align:center">

            <div style="font-size:4rem">
                🎯
            </div>

            <div class="hero-title">
                ExamPredict AI V6
            </div>

            <p class="hero-subtitle" style="margin:auto">
                Sign in with Google to save your analysis history.
            </p>

        </div>
        """)

        if st.button(
            "Continue with Google",
            type="primary",
            use_container_width=True
        ):

            st.login()

        st.stop()

    USER_NAME = getattr(
        st.user,
        "name",
        "Student"
    )

    USER_EMAIL = getattr(
        st.user,
        "email",
        ""
    )

else:

    USER_NAME = "Demo Student"

    USER_EMAIL = "demo-user"


# ============================================================
# SECRET HELPER
# ============================================================

def secret(
    section,
    key,
    default=None
):

    try:

        return st.secrets[
            section
        ][key]

    except Exception:

        return default


# ============================================================
# FILE READING
# ============================================================

def read_file(file):

    if file.name.lower().endswith(".pdf"):

        with pdfplumber.open(file) as pdf:

            pages = []

            for page in pdf.pages:

                pages.append(
                    page.extract_text() or ""
                )

            return "\n".join(pages)

    return file.getvalue().decode(
        "utf-8",
        errors="ignore"
    )


def read_files(files):

    result = []

    for file in files or []:

        try:

            result.append(
                (
                    file.name,
                    read_file(file)
                )
            )

        except Exception as e:

            result.append(
                (
                    file.name,
                    f"Could not read file: {e}"
                )
            )

    return result


# ============================================================
# QUESTION EXTRACTION
# ============================================================

def extract_questions(text):

    parts = re.split(

        r"(?im)"
        r"(?=^\s*"
        r"(?:q(?:uestion)?\s*)?"
        r"\d{1,3}"
        r"\s*[\.\):\-])",

        text or ""
    )

    cleaned = []

    for part in parts:

        part = re.sub(
            r"\s+",
            " ",
            part
        ).strip()

        if len(part) > 25:

            cleaned.append(part)

    return cleaned


# ============================================================
# WORD / TOPIC EXTRACTION
# ============================================================

STOP_WORDS = set("""
the and for are was were with from that this which into their there have has had
what when where why how can could should would will answer explain give find calculate
describe define state name following question questions marks section paper chapter lesson
based use using used about than then also each your their these those them they its exam
examination
""".split())


def extract_topics(
    text,
    limit=50
):

    words = re.findall(
        r"[A-Za-z][A-Za-z\-]{3,}",
        (text or "").lower()
    )

    counter = Counter(
        word
        for word in words
        if word not in STOP_WORDS
    )

    return [
        word.title()
        for word, count
        in counter.most_common(limit)
        if count >= 2
    ]


# ============================================================
# QUESTION TYPE
# ============================================================

def question_type(question):

    text = question.lower()

    if (
        "assertion" in text
        or
        "reason:" in text
    ):

        return "Assertion–Reason"

    if (
        "case study" in text
        or
        "case-based" in text
    ):

        return "Case Study"

    if any(
        x in text
        for x in [
            "calculate",
            "solve",
            "numerical",
            "find the value"
        ]
    ):

        return "Numerical/Application"

    if any(
        x in text
        for x in [
            "diagram",
            "draw",
            "label"
        ]
    ):

        return "Diagram-based"

    if (
        "which of the following" in text
        or
        re.search(
            r"\ba\.\s",
            text
        )
    ):

        return "MCQ"

    if any(
        x in text
        for x in [
            "compare",
            "differentiate",
            "analyze",
            "analyse"
        ]
    ):

        return "HOTS/Analysis"

    if any(
        x in text
        for x in [
            "explain",
            "describe",
            "why"
        ]
    ):

        return "Conceptual"

    return "Short Answer"


# ============================================================
# HISTORICAL + ANN ENGINE
# ============================================================

def analyze_patterns(
    material,
    papers
):

    topic_seed = extract_topics(
        material,
        60
    )

    all_questions = []

    for name, text in papers:

        qs = extract_questions(text)

        all_questions.extend(qs)

        topic_seed.extend(
            extract_topics(
                text,
                20
            )
        )

    topics = [
        topic
        for topic, count
        in Counter(topic_seed).most_common(45)
    ]

    if not topics:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "engine":
                    "No topic evidence found",

                "backtest":
                    "Unavailable"
            }
        )

    history = []

    for paper_index, (
        name,
        text
    ) in enumerate(papers):

        qs = extract_questions(text)

        lower = text.lower()

        for topic in topics:

            hits = len(
                re.findall(
                    r"\b"
                    +
                    re.escape(
                        topic.lower()
                    )
                    +
                    r"\b",
                    lower
                )
            )

            question_hits = sum(
                topic.lower()
                in question.lower()
                for question in qs
            )

            history.append({

                "paper":
                    name,

                "paper_index":
                    paper_index,

                "topic":
                    topic,

                "present":
                    int(
                        hits > 0
                        or
                        question_hits > 0
                    )
            })

    history_df = pd.DataFrame(
        history
    )

    rows = []

    for topic in topics:

        topic_history = (
            history_df[
                history_df["topic"]
                == topic
            ]
            .sort_values(
                "paper_index"
            )["present"]
            .tolist()
        )

        frequency = (
            sum(topic_history)
            /
            max(
                len(topic_history),
                1
            )
            *
            100
        )

        recent = (
            sum(
                topic_history[-3:]
            )
            /
            max(
                min(
                    3,
                    len(topic_history)
                ),
                1
            )
            *
            100
        )

        if len(topic_history) >= 2:

            trend = (
                topic_history[-1]
                -
                topic_history[0]
            )

        else:

            trend = 0

        material_hits = len(
            re.findall(
                r"\b"
                +
                re.escape(
                    topic.lower()
                )
                +
                r"\b",
                material.lower()
            )
        )

        rows.append({

            "Topic":
                topic,

            "Historical Frequency":
                round(
                    frequency,
                    1
                ),

            "Recent Frequency":
                round(
                    recent,
                    1
                ),

            "Trend":
                trend,

            "Material Evidence":
                material_hits
        })

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # ANN
    # --------------------------------------------------------

    if (
        ANN_OK
        and
        len(df) >= 8
    ):

        X = df[
            [
                "Historical Frequency",
                "Recent Frequency",
                "Trend",
                "Material Evidence"
            ]
        ].astype(float).values

        y = (
            0.45 * X[:, 0]
            +
            0.35 * X[:, 1]
            +
            5 * X[:, 2]
            +
            2 * X[:, 3]
        )

        try:

            model = make_pipeline(

                StandardScaler(),

                MLPRegressor(

                    hidden_layer_sizes=(
                        16,
                        8
                    ),

                    max_iter=1200,

                    random_state=42
                )
            )

            model.fit(
                X,
                y
            )

            prediction = model.predict(
                X
            )

            low = float(
                prediction.min()
            )

            high = float(
                prediction.max()
            )

            if high == low:

                df["ANN Score"] = 50

            else:

                df["ANN Score"] = (
                    (
                        prediction
                        -
                        low
                    )
                    /
                    (
                        high
                        -
                        low
                    )
                    *
                    60
                    +
                    40
                )

            df[
                "Prediction Strength"
            ] = (

                0.45
                *
                df["ANN Score"]

                +

                0.35
                *
                df["Recent Frequency"]

                +

                0.20
                *
                df["Historical Frequency"]

            ).clip(
                0,
                100
            ).round(1)

            engine_name = (
                "ANN pattern layer + "
                "historical frequency + "
                "recency"
            )

        except Exception:

            df[
                "Prediction Strength"
            ] = (

                0.55
                *
                df["Recent Frequency"]

                +

                0.45
                *
                df["Historical Frequency"]

            ).round(1)

            engine_name = (
                "Statistical fallback"
            )

    else:

        df[
            "Prediction Strength"
        ] = (

            0.55
            *
            df["Recent Frequency"]

            +

            0.45
            *
            df["Historical Frequency"]

        ).round(1)

        engine_name = (
            "Statistical pattern engine"
        )

    df = (
        df
        .sort_values(
            "Prediction Strength",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    backtest = (
        "Need at least "
        "3 previous papers"
    )

    if len(papers) >= 3:

        train = history_df[
            history_df["paper_index"]
            <
            len(papers) - 1
        ]

        test = history_df[
            history_df["paper_index"]
            ==
            len(papers) - 1
        ]

        top_topics = set(

            train
            .groupby("topic")[
                "present"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(10)
            .index
        )

        actual_topics = set(
            test.loc[
                test["present"] == 1,
                "topic"
            ]
        )

        coverage = round(

            100
            *
            len(
                top_topics
                &
                actual_topics
            )
            /
            max(
                len(actual_topics),
                1
            ),

            1
        )

        backtest = (
            "Last-paper top-10 "
            f"topic coverage: {coverage}%"
        )

    # --------------------------------------------------------
    # QUESTION ANALYSIS
    # --------------------------------------------------------

    question_rows = []

    for name, text in papers:

        for question in extract_questions(
            text
        ):

            question_rows.append({

                "Paper":
                    name,

                "Question Type":
                    question_type(
                        question
                    ),

                "Question":
                    question[:500]
            })

    question_df = pd.DataFrame(
        question_rows
    )

    return (
        df,
        question_df,
        {
            "engine":
                engine_name,

            "backtest":
                backtest
        }
    )


# ============================================================
# OPENAI
# ============================================================

def run_ai_analysis(
    material,
    notes,
    papers,
    profile,
    use_web
):

    if not OPENAI_OK:

        return (
            None,
            "OpenAI package unavailable"
        )

    try:

        client = OpenAI(
            api_key=
            st.secrets[
                "openai"
            ][
                "api_key"
            ]
        )

    except Exception:

        return (
            None,
            "OpenAI key not configured"
        )

    model = secret(
        "openai",
        "model",
        "gpt-5.6-luna"
    )

    previous_papers = "\n\n".join(

        f"### {name}\n{text[:12000]}"

        for name, text
        in papers[:8]
    )

    prompt = f"""

You are the AI reasoning layer
of an exam-pattern analysis application.

IMPORTANT:

You cannot know the actual future
exam paper.

Do NOT claim certainty.

Use uploaded evidence first.

PROFILE:

{json.dumps(profile, indent=2)}

Return JSON only.

Required structure:

{{
    "prediction_summary": [
        {{
            "topic": "...",
            "score": 0,
            "reason": "..."
        }}
    ],

    "question_type_analysis": [
        {{
            "type": "...",
            "frequency": 0,
            "interpretation": "..."
        }}
    ],

    "blueprint": [
        {{
            "marks": 1,
            "approximate_count": 0
        }}
    ],

    "study_priority": [
        "..."
    ],

    "pattern_summary": "...",

    "target_strategy": "...",

    "limitations": "...",

    "papers": [

        {{
            "title":
                "Predicted Practice Paper 1",

            "total_marks":
                {profile["paper_marks"]},

            "sections": [

                {{
                    "name":
                        "Section A",

                    "instructions":
                        "...",

                    "questions": [

                        {{
                            "number":
                                1,

                            "question":
                                "...",

                            "marks":
                                1
                        }}
                    ]
                }}
            ]
        }}
    ]
}}

Generate exactly THREE
different practice papers.

Every paper MUST total exactly:

{profile["paper_marks"]}

Do not copy uploaded questions
verbatim.

Use the style and difficulty
patterns of the uploaded papers.

STUDY MATERIAL:

{material[:28000]}

SHORT NOTES:

{notes[:18000]}

PREVIOUS PAPERS:

{previous_papers}

"""

    try:

        tools = []

        if use_web:

            tools = [
                {
                    "type":
                        "web_search"
                }
            ]

        response = client.responses.create(

            model=model,

            input=prompt,

            tools=tools
        )

        return (
            json.loads(
                response.output_text
            ),

            "OpenAI reasoning"
            +
            (
                " + web search"
                if use_web
                else ""
            )
        )

    except Exception as e:

        return (
            None,
            f"OpenAI failed: {e}"
        )


# ============================================================
# FALLBACK PAPER GENERATOR
# ============================================================

def generate_fallback_papers(
    topic_df,
    total_marks
):

    if not topic_df.empty:

        topics = (
            topic_df[
                "Topic"
            ]
            .head(18)
            .tolist()
        )

    else:

        topics = [
            "Core syllabus concepts"
        ]

    allocations = {

        40:
            [10, 10, 10, 10],

        50:
            [10, 10, 15, 15],

        80:
            [20, 20, 20, 20],

        100:
            [20, 20, 30, 30]
    }

    section_marks = allocations[
        total_marks
    ]

    section_names = [

        "MCQ / Objective",

        "Short Answer",

        "Application / Numerical",

        "Long Answer / HOTS"
    ]

    papers = []

    for paper_number in range(
        1,
        4
    ):

        sections = []

        question_number = 1

        for section_index, section_total in enumerate(
            section_marks
        ):

            remaining = section_total

            section_questions = []

            for j in range(4):

                if j == 3:

                    marks = remaining

                else:

                    marks = section_total // 4

                remaining -= marks

                topic = topics[
                    (
                        question_number
                        +
                        paper_number
                        +
                        j
                        -
                        2
                    )
                    %
                    len(topics)
                ]

                templates = [

                    f"Which statement best explains {topic}?",

                    f"Explain the main concept involved in {topic}.",

                    f"Apply the principles of {topic} to a new situation.",

                    f"Analyze {topic} and justify your answer with reasoning."
                ]

                section_questions.append({

                    "number":
                        question_number,

                    "question":
                        templates[
                            section_index
                        ],

                    "marks":
                        int(marks)
                })

                question_number += 1

            sections.append({

                "name":
                    f"Section {chr(65 + section_index)} — "
                    f"{section_names[section_index]}",

                "instructions":
                    f"This section carries "
                    f"{section_total} marks.",

                "questions":
                    section_questions
            })

        papers.append({

            "title":
                f"Predicted Practice Paper {paper_number}",

            "total_marks":
                total_marks,

            "sections":
                sections
        })

    return papers


# ============================================================
# EXCEL EXPORT
# ============================================================

def create_excel(
    result,
    topic_df,
    question_df,
    profile,
    files
):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        pd.DataFrame(
            [profile]
        ).to_excel(
            writer,
            "Overview",
            index=False
        )

        if not topic_df.empty:

            topic_df.to_excel(
                writer,
                "Topic Analysis",
                index=False
            )

        if not question_df.empty:

            question_df.to_excel(
                writer,
                "Question Analysis",
                index=False
            )

        pd.DataFrame(
            files
        ).to_excel(
            writer,
            "Uploaded Files",
            index=False
        )

        prediction_df = pd.DataFrame(
            result.get(
                "prediction_summary",
                []
            )
        )

        if not prediction_df.empty:

            prediction_df.to_excel(
                writer,
                "Predictions",
                index=False
            )

        paper_rows = []

        for paper in result.get(
            "papers",
            []
        )[:3]:

            for section in paper.get(
                "sections",
                []
            ):

                for question in section.get(
                    "questions",
                    []
                ):

                    paper_rows.append({

                        "Paper":
                            paper.get(
                                "title"
                            ),

                        "Section":
                            section.get(
                                "name"
                            ),

                        "Question":
                            question.get(
                                "question"
                            ),

                        "Marks":
                            question.get(
                                "marks"
                            )
                    })

        if paper_rows:

            pd.DataFrame(
                paper_rows
            ).to_excel(
                writer,
                "Predicted Papers",
                index=False
            )

    return output.getvalue()


# ============================================================
# PDF EXPORT
# ============================================================

def create_pdf(
    result,
    profile
):

    if not PDF_OK:

        return None

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = [

        Paragraph(
            "ExamPredict AI V6",
            styles["Title"]
        ),

        Paragraph(

            f"{profile['subject']} | "
            f"Class {profile['grade']} | "
            f"{profile['paper_marks']} marks | "
            f"Target {profile['target_score']}/100",

            styles["Normal"]
        ),

        Spacer(
            1,
            12
        )
    ]

    story.append(

        Paragraph(
            "Prediction Summary",
            styles["Heading2"]
        )
    )

    for item in result.get(
        "prediction_summary",
        []
    )[:15]:

        story.append(

            Paragraph(

                f"<b>"
                f"{item.get('topic','')}"
                f"</b> — "
                f"{item.get('score','')}% — "
                f"{item.get('reason','')}",

                styles["BodyText"]
            )
        )

        story.append(
            Spacer(
                1,
                4
            )
        )

    for paper in result.get(
        "papers",
        []
    )[:3]:

        story.append(

            Paragraph(
                paper.get(
                    "title",
                    ""
                ),
                styles["Heading2"]
            )
        )

        for section in paper.get(
            "sections",
            []
        ):

            story.append(

                Paragraph(
                    section.get(
                        "name",
                        ""
                    ),
                    styles["Heading3"]
                )
            )

            for question in section.get(
                "questions",
                []
            ):

                story.append(

                    Paragraph(

                        f"{question.get('number')}. "
                        f"{question.get('question')} "
                        f"[{question.get('marks')} marks]",

                        styles["BodyText"]
                    )
                )

    document.build(
        story
    )

    return output.getvalue()


# ============================================================
# SUPABASE
# ============================================================

def save_to_database(
    profile,
    file_names,
    result
):

    if not SUPABASE_OK:

        return

    try:

        supabase = create_client(

            st.secrets[
                "supabase"
            ][
                "url"
            ],

            st.secrets[
                "supabase"
            ][
                "key"
            ]
        )

        supabase.table(
            "exam_analyses"
        ).insert({

            "user_email":
                USER_EMAIL,

            "user_name":
                USER_NAME,

            **profile,

            "files":
                file_names,

            "prediction_summary":
                result.get(
                    "prediction_summary",
                    []
                ),

            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat()

        }).execute()

    except Exception as e:

        st.session_state[
            "db_error"
        ] = str(e)


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <span class="badge">
        V6
    </span>

    <span class="badge">
        ANN PATTERN ENGINE
    </span>

    <span class="badge">
        AI REASONING
    </span>

    <span class="badge">
        BACKTEST
    </span>

    <span class="badge">
        3 PRACTICE PAPERS
    </span>

    <div class="hero-title">
        ExamPredict AI
    </div>

    <div class="hero-subtitle">

        Analyze your syllabus, notes and previous
        question papers to identify recurring
        patterns, topic priorities, question types
        and practice-paper blueprints.

    </div>

</div>
""")


# ============================================================
# EXAM PROFILE
# ============================================================

st.markdown(
    "## 🎯 Exam Profile"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    subject = st.selectbox(

        "Subject",

        [

            "Mathematics",
            "Science",
            "Social Science",
            "English",
            "Hindi",
            "Computer Science",
            "Physics",
            "Chemistry",
            "Biology",
            "Economics",
            "Political Science",
            "Geography",
            "History",
            "Other"

        ]
    )

with c2:

    grade = st.selectbox(

        "Class / Grade",

        [
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "Other"
        ]
    )

with c3:

    paper_marks = st.selectbox(

        "Question paper marks",

        [
            40,
            50,
            80,
            100
        ]
    )

with c4:

    difficulty = st.selectbox(

        "Difficulty",

        [

            "Match past papers",
            "Easy",
            "Moderate",
            "Hard",
            "HOTS-heavy"

        ]
    )


c1, c2 = st.columns(2)

with c1:

    target_score = st.slider(

        "🎯 What mark are you aiming for?",

        1,
        100,
        90
    )

with c2:

    exam_name = st.text_input(

        "Exam name",

        "Upcoming Examination"
    )


use_web = st.checkbox(

    "🌐 Use optional web research through OpenAI",

    value=False
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    "## 📚 Your Exam Data"
)

a, b = st.columns(2)

with a:

    material_files = st.file_uploader(

        "📖 Study material / syllabus",

        type=[
            "pdf",
            "txt"
        ],

        accept_multiple_files=True
    )

with b:

    paper_files = st.file_uploader(

        "📝 Previous question papers",

        type=[
            "pdf",
            "txt"
        ],

        accept_multiple_files=True
    )


notes_files = st.file_uploader(

    "📌 Upload short-note documents",

    type=[
        "pdf",
        "txt"
    ],

    accept_multiple_files=True
)


notes_text = st.text_area(

    "Or paste your short notes",

    height=150,

    placeholder=
        "Paste lesson-wise short notes here..."
)


# ============================================================
# RUN BUTTON
# ============================================================

if st.button(

    "🚀 ANALYZE EXAM & GENERATE 3 PAPERS",

    type="primary",

    use_container_width=True

):

    material_items = read_files(
        material_files
    )

    paper_items = read_files(
        paper_files
    )

    notes_items = read_files(
        notes_files
    )

    material = "\n\n".join(

        text
        for name, text
        in material_items
    )

    notes = (

        notes_text
        +
        "\n\n"
        +
        "\n\n".join(

            text
            for name, text
            in notes_items
        )
    )

    if (
        not material
        and
        not notes
        and
        not paper_items
    ):

        st.error(

            "Upload at least one "
            "study/notes file or previous paper."
        )

        st.stop()


    profile = {

        "subject":
            subject,

        "grade":
            grade,

        "paper_marks":
            paper_marks,

        "difficulty":
            difficulty,

        "target_score":
            target_score,

        "exam_name":
            exam_name
    }


    # --------------------------------------------------------
    # LOCAL ANALYSIS
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing historical patterns..."
    ):

        topic_df, question_df, engine_info = analyze_patterns(

            material
            +
            "\n"
            +
            notes,

            paper_items
        )


    # --------------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------------

    ai_result, ai_status = run_ai_analysis(

        material,

        notes,

        paper_items,

        profile,

        use_web
    )


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if ai_result:

        result = ai_result

    else:

        result = {

            "prediction_summary": [

                {

                    "topic":
                        row["Topic"],

                    "score":
                        float(
                            row[
                                "Prediction Strength"
                            ]
                        ),

                    "reason":

                        f"Historical frequency "
                        f"{row['Historical Frequency']}%, "
                        f"recent frequency "
                        f"{row['Recent Frequency']}%."

                }

                for _, row
                in topic_df.head(
                    15
                ).iterrows()
            ],

            "question_type_analysis":
                [],

            "blueprint":
                [],

            "study_priority":

                (
                    topic_df[
                        "Topic"
                    ]
                    .head(10)
                    .tolist()

                    if not topic_df.empty

                    else []
                ),

            "pattern_summary":

                "Local historical pattern analysis "
                "was used because AI was unavailable "
                "or disabled.",

            "target_strategy":

                f"Prioritize high-ranked topics "
                f"and practice toward "
                f"{target_score}/100.",

            "limitations":

                "This is an estimate and not "
                "knowledge of the future exam.",

            "papers":

                generate_fallback_papers(
                    topic_df,
                    paper_marks
                )
        }


    result["papers"] = result.get(
        "papers",
        []
    )[:3]


    # --------------------------------------------------------
    # FILE LIST
    # --------------------------------------------------------

    files = (

        [
            {
                "Type":
                    "Study Material",

                "Filename":
                    name
            }

            for name, text
            in material_items
        ]

        +

        [
            {
                "Type":
                    "Short Notes",

                "Filename":
                    name
            }

            for name, text
            in notes_items
        ]

        +

        [
            {
                "Type":
                    "Previous Paper",

                "Filename":
                    name
            }

            for name, text
            in paper_items
        ]
    )


    # --------------------------------------------------------
    # SAVE SESSION
    # --------------------------------------------------------

    st.session_state.result = result

    st.session_state.topic_df = (
        topic_df
    )

    st.session_state.question_df = (
        question_df
    )

    st.session_state.profile = (
        profile
    )

    st.session_state.engine_info = (
        engine_info
    )

    st.session_state.ai_status = (
        ai_status
    )

    st.session_state.files = (
        files
    )


    # --------------------------------------------------------
    # EXPORTS
    # --------------------------------------------------------

    st.session_state.xlsx = create_excel(

        result,

        topic_df,

        question_df,

        profile,

        files
    )


    st.session_state.pdf = create_pdf(

        result,

        profile
    )


    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    save_to_database(

        profile,

        [
            x["Filename"]
            for x in files
        ],

        result
    )


    st.success(
        "Analysis complete — "
        "3 practice papers generated."
    )


# ============================================================
# RESULTS
# ============================================================

if "result" in st.session_state:

    result = (
        st.session_state.result
    )

    topic_df = (
        st.session_state.topic_df
    )

    question_df = (
        st.session_state.question_df
    )

    profile = (
        st.session_state.profile
    )


    st.markdown("---")

    st.markdown(
        "## 📊 Exam Intelligence Dashboard"
    )


    top_topic = (

        topic_df.iloc[0]["Topic"]

        if not topic_df.empty

        else "—"
    )


    top_score = (

        topic_df.iloc[0][
            "Prediction Strength"
        ]

        if not topic_df.empty

        else 0
    )


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.html(

            f"""
            <div class="metric">

                <div class="metric-label">
                    TOP PREDICTION
                </div>

                <div class="metric-value">
                    {top_topic}
                </div>

            </div>
            """
        )


    with m2:

        st.html(

            f"""
            <div class="metric">

                <div class="metric-label">
                    PREDICTION STRENGTH
                </div>

                <div class="metric-value">
                    {top_score}%
                </div>

            </div>
            """
        )


    with m3:

        st.html(

            f"""
            <div class="metric">

                <div class="metric-label">
                    TARGET
                </div>

                <div class="metric-value">
                    {target_score}/100
                </div>

            </div>
            """
        )


    with m4:

        st.html(

            """
            <div class="metric">

                <div class="metric-label">
                    PRACTICE PAPERS
                </div>

                <div class="metric-value">
                    3
                </div>

            </div>
            """
        )


    st.markdown(
        "### 🔬 Engine Status"
    )

    st.write(
        st.session_state.engine_info[
            "engine"
        ]
    )

    st.caption(
        st.session_state.engine_info[
            "backtest"
        ]
    )

    st.caption(
        st.session_state.ai_status
    )


    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3, tab4 = st.tabs(

        [
            "🔥 Predictions",
            "📚 Question Patterns",
            "📄 Practice Papers",
            "📦 Export"
        ]
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    with tab1:

        st.markdown(
            "### 🔥 Top predicted topics"
        )


        if not topic_df.empty:

            st.dataframe(

                topic_df.head(25),

                use_container_width=True,

                hide_index=True
            )


        for item in result.get(
            "prediction_summary",
            []
        )[:15]:

            st.markdown(

                f"""
                **{item.get('topic','')}
                —
                {item.get('score','')}%**

                {item.get('reason','')}
                """
            )


        st.markdown(
            "### 🎯 Study priority"
        )


        for i, topic in enumerate(

            result.get(
                "study_priority",
                []
            )[:12],

            1
        ):

            st.write(
                f"{i}. {topic}"
            )


        st.markdown(
            "### 🧠 AI pattern summary"
        )

        st.write(
            result.get(
                "pattern_summary",
                ""
            )
        )


        st.markdown(
            "### 🎯 Target strategy"
        )

        st.write(
            result.get(
                "target_strategy",
                ""
            )
        )


    # ========================================================
    # QUESTION PATTERNS
    # ========================================================

    with tab2:

        if not question_df.empty:

            st.markdown(
                "### Question-type distribution"
            )

            counts = (

                question_df[
                    "Question Type"
                ]

                .value_counts()

                .rename_axis(
                    "Type"
                )

                .reset_index(
                    name="Count"
                )
            )

            st.bar_chart(
                counts.set_index(
                    "Type"
                )
            )


            st.dataframe(

                question_df.head(
                    100
                ),

                use_container_width=True,

                hide_index=True
            )

        else:

            st.info(
                "Upload previous papers "
                "to unlock question-pattern analysis."
            )


    # ========================================================
    # PRACTICE PAPERS
    # ========================================================

    with tab3:

        st.markdown(

            f"""
            ### 📄 Three
            {profile['paper_marks']}-mark
            practice papers
            """
        )


        for paper in result.get(
            "papers",
            []
        )[:3]:

            st.markdown(

                f"""
                ## {paper.get(
                    'title',
                    'Practice Paper'
                )}
                """
            )

            st.caption(

                f"Total: "
                f"{paper.get(
                    'total_marks',
                    profile['paper_marks']
                )} marks"
            )


            for section in paper.get(
                "sections",
                []
            ):

                st.markdown(

                    f"""
                    ### {section.get(
                        'name',
                        'Section'
                    )}
                    """
                )

                st.write(

                    section.get(
                        "instructions",
                        ""
                    )
                )


                for question in section.get(
                    "questions",
                    []
                ):

                    st.markdown(

                        f"""
                        **{question.get(
                            'number'
                        )}.**

                        {question.get(
                            'question'
                        )}

                        **[
                        {question.get(
                            'marks'
                        )} marks]**
                        """
                    )


            st.divider()


    # ========================================================
    # EXPORT
    # ========================================================

    with tab4:

        st.markdown(
            "### 📦 Download your analysis"
        )


        st.download_button(

            "📊 Download Excel",

            data=
                st.session_state.xlsx,

            file_name=
                "ExamPredict_AI_V6.xlsx",

            mime=
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

            use_container_width=True
        )


        if st.session_state.pdf:

            st.download_button(

                "📄 Download PDF",

                data=
                    st.session_state.pdf,

                file_name=
                    "ExamPredict_AI_V6.pdf",

                mime=
                    "application/pdf",

                use_container_width=True
            )

        else:

            st.info(
                "Install reportlab "
                "to enable PDF export."
            )


        st.markdown(
            "### 📁 Files analyzed"
        )


        st.dataframe(

            pd.DataFrame(
                st.session_state.files
            ),

            use_container_width=True,

            hide_index=True
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.html(

        """
        <div class="disclaimer">

        <b>IMPORTANT DISCLAIMER</b>

        <br><br>

        ExamPredict AI cannot know the actual
        future examination paper.

        Prediction strength is NOT a guarantee
        that a question will appear.

        The system estimates patterns from
        uploaded study material, notes and
        historical question papers.

        Always study the complete syllabus.

        </div>
        """
    )
