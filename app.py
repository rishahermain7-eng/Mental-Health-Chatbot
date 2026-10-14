import json
import os

import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# MindEaseAI
# Fast version - profile, memory, AI response and emotion score
# ============================================================

st.set_page_config(
    page_title="MindEaseAI",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "profile" not in st.session_state:
    st.session_state.profile = {}

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# GEMINI SETUP
# ============================================================

def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


API_KEY = get_api_key()

if not API_KEY:
    st.error("Gemini API key is missing.")
    st.info(
        "For local use, set GEMINI_API_KEY in your terminal. "
        "For Streamlit Cloud, add it under Settings → Secrets."
    )
    st.code('$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"')
    st.stop()


client = genai.Client(api_key=API_KEY)

# IMPORTANT:
# Flash-Lite is designed for low-latency/high-throughput workloads.
# Minimal thinking is supported and is intended for fast chat-like tasks.
MODEL = "gemini-3.5-flash-lite"


# ============================================================
# STRUCTURED RESPONSE
# ============================================================

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "emotion": {
            "type": "STRING",
            "description": "Best overall emotion in the user's complete message.",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence from 0 to 1.",
        },
        "response": {
            "type": "STRING",
            "description": (
                "A natural supportive response that addresses all important "
                "concerns in the user's complete message."
            ),
        },
    },
    "required": ["emotion", "confidence", "response"],
}


# ============================================================
# PROFILE MEMORY
# ============================================================

def profile_context():
    p = st.session_state.profile

    if not p:
        return "No saved profile."

    return (
        f"Name: {p.get('name') or 'Not provided'}\n"
        f"Age: {p.get('age') or 'Not provided'}\n"
        f"Stressors: {p.get('stressors') or 'Not provided'}\n"
        f"Medications: {p.get('medications') or 'None saved'}"
    )


def recent_history():
    if not st.session_state.messages:
        return "No previous conversation."

    lines = []

    # Only the latest 4 messages are sent to reduce request size.
    for item in st.session_state.messages[-4:]:
        role = "User" if item["role"] == "user" else "MindEaseAI"
        lines.append(f"{role}: {item['content']}")

    return "\n".join(lines)


# ============================================================
# AI
# ============================================================

def ask_mindease(user_text):
    """
    One fast AI call does all three:
    - understands the full message
    - detects emotion + confidence
    - generates the response

    No hard-coded response mapping is used.
    """

    prompt = f"""
You are MindEaseAI, a supportive AI companion.

Understand the COMPLETE user message before replying.
Address EVERY important concern in it, not just one keyword.

Use the saved profile when relevant.

Rules:
- Be warm, natural, concise and specific.
- If the user mentions several things, address all of them.
- Do not give generic replies based on one keyword.
- Emotion is an estimate, not a diagnosis.
- Never diagnose.
- Never tell the user to start, stop, increase, decrease or change medication.
- For physical symptoms, give general supportive guidance and suggest
  professional medical care when symptoms are severe, worsening or persistent.
- If asked about saved medications, recall the exact saved names.
- Never invent profile information.
- Keep the response short enough for a quick chat.

SAVED PROFILE:
{profile_context()}

RECENT CONVERSATION:
{recent_history()}

USER:
{user_text}

Return ONLY JSON:
emotion, confidence, response
"""

    result = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,

            # Short answer = faster output.
            # Thinking level is kept at minimal for the fastest model.
            max_output_tokens=300,

            thinking_config=types.ThinkingConfig(
                thinking_level="minimal"
            ),
        ),
    )

    if not result.text:
        raise RuntimeError("Gemini returned an empty response.")

    data = json.loads(result.text)

    emotion = str(data.get("emotion", "neutral")).strip()
    answer = str(data.get("response", "")).strip()

    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

    return emotion, confidence, answer


# ============================================================
# ERROR HANDLING
# ============================================================

def friendly_error(error):
    text = str(error)
    upper = text.upper()

    if "404" in upper or "NOT_FOUND" in upper:
        return (
            "The Gemini model is not available for this API configuration. "
            "Try updating google-genai and check your Gemini API access."
        )

    if "401" in upper or "403" in upper or "PERMISSION" in upper:
        return (
            "Gemini rejected the API key. Check the API key and make sure "
            "the Gemini API is enabled for your Google project."
        )

    if "429" in upper or "RESOURCE_EXHAUSTED" in upper:
        return (
            "Gemini temporarily reached a usage limit. "
            "Please wait a little and try again."
        )

    if "503" in upper or "UNAVAILABLE" in upper:
        return (
            "Gemini is temporarily busy. Please try again in a few seconds."
        )

    return "MindEaseAI could not generate a response. Please try again."


# ============================================================
# HEADER
# ============================================================

st.title("MindEaseAI")
st.caption("A supportive AI companion for emotional well-being")

st.info(
    "MindEaseAI is a student project. It provides supportive information "
    "and is not a substitute for professional medical or mental-health care."
)


# ============================================================
# PROFILE
# ============================================================

with st.expander(
    "Set Up Your Profile",
    expanded=not bool(st.session_state.profile),
):

    p = st.session_state.profile

    name = st.text_input(
        "Name",
        value=p.get("name", ""),
        placeholder="Your name",
        key="profile_name",
    )

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=int(p.get("age", 21)),
        step=1,
        key="profile_age",
    )

    stressors = st.text_area(
        "Current stressors",
        value=p.get("stressors", ""),
        placeholder="Example: exams, work, relationships...",
        key="profile_stressors",
    )

    medications = st.text_area(
        "Medications you want MindEaseAI to remember",
        value=p.get("medications", ""),
        placeholder="Example: medicine names",
        key="profile_medications",
    )

    if st.button("Save Profile", type="primary", key="save_profile"):

        st.session_state.profile = {
            "name": name.strip(),
            "age": int(age),
            "stressors": stressors.strip(),
            "medications": medications.strip(),
        }

        st.success("Profile saved successfully.")
        st.rerun()


# Show saved information
if st.session_state.profile:

    with st.expander("View Saved Profile"):

        p = st.session_state.profile

        st.write(f"**Name:** {p.get('name') or 'Not provided'}")
        st.write(f"**Age:** {p.get('age') or 'Not provided'}")
        st.write(
            f"**Current stressors:** "
            f"{p.get('stressors') or 'Not provided'}"
        )
        st.write(
            f"**Saved medications:** "
            f"{p.get('medications') or 'None saved'}"
        )


# ============================================================
# CONVERSATION
# ============================================================

st.header("Conversation")

st.write(
    "Tell me what you're experiencing. You can talk about emotions, "
    "stress, everyday problems, or physical symptoms."
)


for item in st.session_state.messages:

    if item["role"] == "user":

        with st.chat_message("user"):
            st.markdown(item["content"])

    else:

        with st.chat_message("assistant"):
            st.markdown(item["content"])

            if item.get("emotion"):
                st.caption(
                    f"Possible emotion: {item['emotion'].title()} | "
                    f"AI confidence: {item['score']:.0%}"
                )


# ============================================================
# INPUT
# ============================================================

st.subheader("Talk to MindEaseAI")

with st.form("chat_form", clear_on_submit=True):

    user_text = st.text_area(
        "Type your message",
        placeholder=(
            "Example: I have been feeling dizzy and stressed lately..."
        ),
        height=120,
    )

    send = st.form_submit_button(
        "Send",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# SEND
# ============================================================

if send:

    user_text = user_text.strip()

    if not user_text:

        st.warning("Please type a message first.")

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):

            with st.spinner("MindEaseAI is thinking..."):

                try:

                    emotion, confidence, answer = ask_mindease(
                        user_text
                    )

                    st.markdown(answer)

                    st.caption(
                        f"Possible emotion: {emotion.title()} | "
                        f"AI confidence: {confidence:.0%}"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "emotion": emotion,
                            "score": confidence,
                        }
                    )

                except Exception as error:

                    st.error(friendly_error(error))

                    with st.expander("Technical details"):
                        st.code(str(error))


# ============================================================
# CLEAR
# ============================================================

if st.session_state.messages:

    if st.button(
        "Clear Conversation",
        key="clear_conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


st.divider()

st.caption(
    "Emotion detection can be inaccurate and should not be treated "
    "as a medical diagnosis."
)
