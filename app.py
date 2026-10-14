import json
import os

import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# MindEaseAI
# ============================================================

st.set_page_config(
    page_title="MindEaseAI",
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

    @media (max-width: 600px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
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
        "For Streamlit Cloud, add GEMINI_API_KEY under "
        "Settings → Secrets."
    )
    st.stop()


client = genai.Client(api_key=API_KEY)

# Keep the fast model that is already working
MODEL = "gemini-3.5-flash-lite"


# ============================================================
# STRUCTURED RESPONSE
# ============================================================

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "emotion": {
            "type": "STRING",
            "description": (
                "Best overall emotion expressed in the complete "
                "user message. This is only an estimate."
            ),
        },
        "confidence": {
            "type": "NUMBER",
            "description": (
                "Confidence from 0 to 1 for the emotion estimate."
            ),
        },
        "response": {
            "type": "STRING",
            "description": (
                "A warm, natural, human-like response that addresses "
                "all important concerns in the user's message."
            ),
        },
    },
    "required": [
        "emotion",
        "confidence",
        "response",
    ],
}


# ============================================================
# PROFILE MEMORY
# ============================================================

def profile_context():
    profile = st.session_state.profile

    if not profile:
        return "No saved profile."

    return (
        f"Name: {profile.get('name') or 'Not provided'}\n"
        f"Age: {profile.get('age') or 'Not provided'}\n"
        f"Stressors: {profile.get('stressors') or 'Not provided'}\n"
        f"Medications: {profile.get('medications') or 'None saved'}"
    )


def recent_history():
    if not st.session_state.messages:
        return "No previous conversation."

    lines = []

    # Last 6 messages keeps the request reasonably small and fast.
    for item in st.session_state.messages[-6:]:
        role = (
            "User"
            if item["role"] == "user"
            else "MindEaseAI"
        )
        lines.append(
            f"{role}: {item['content']}"
        )

    return "\n".join(lines)


# ============================================================
# AI RESPONSE
# ============================================================

def ask_mindease(user_text):

    prompt = f"""
You are MindEaseAI, a warm and supportive AI companion.

Your job is to have a natural, human-like conversation with the user.
Respond like a thoughtful person who is genuinely listening.

Do NOT sound like:
- a medical website
- a customer-support bot
- a collection of predefined responses
- a motivational quote generator

============================================================
UNDERSTAND THE COMPLETE MESSAGE
============================================================

Read the user's ENTIRE message before responding.

Do not focus on only one keyword or emotion.

If the user mentions multiple concerns, address the important
concerns naturally.

For example:

"I have a headache, I feel dizzy, and I'm stressed about my exams."

This contains:
- headache
- dizziness
- exam stress

Do NOT respond only to "stress".

Another example:

"My relatives keep talking about me and I feel depressed."

Address:
- the emotional hurt
- the family situation
- the user's feelings
- appropriate encouragement

The user's actual words and situation are more important than
the detected emotion.

============================================================
EMOTIONAL SUPPORT
============================================================

When the user is sad, anxious, stressed, lonely, hurt, angry,
overwhelmed, discouraged, or emotionally exhausted:

First acknowledge what they are experiencing.

Then respond with genuine understanding.

Do not immediately give a checklist of advice.

Do not say:
- "Just stay positive."
- "Don't worry."
- "Everything will be fine."
- "Just breathe."

unless that genuinely fits the situation.

When someone is being judged, criticized, insulted, rejected,
or talked about by other people, provide appropriate encouragement.

You can naturally communicate ideas such as:

- Other people's words do not define the user's worth.
- Someone else's opinion is not automatically the truth.
- The user does not have to carry every opinion people have about them.
- The situation can hurt without defining who they are.
- They deserve to be treated with respect.

Do NOT repeat the same motivational sentence every time.

Adapt the encouragement to the actual situation.

If the user needs someone to listen, listen.

Do not turn every emotional conversation into a medical lecture.

============================================================
MULTIPLE CONCERNS
============================================================

If the user mentions several things:

1. Understand all of them.
2. Acknowledge the important ones.
3. Connect them naturally when appropriate.
4. Do not ignore a physical symptom just because an emotion
   was also detected.
5. Do not ignore emotional distress just because a physical
   symptom was mentioned.

============================================================
PHYSICAL HEALTH
============================================================

If the user mentions symptoms such as:

headache, fever, cold, cough, dizziness, nausea, stomach
problems, pain, tiredness, weakness, etc.:

Acknowledge the symptom.

Give reasonable general self-care information when appropriate.

Consider:
- severity
- duration
- whether it is worsening
- other symptoms mentioned

If symptoms are severe, unusual, persistent, worsening,
or concerning, recommend speaking with an appropriate
healthcare professional.

If something sounds potentially urgent, recommend urgent
medical attention.

Never diagnose the user.

Do not pretend to know the exact cause of a symptom.

============================================================
MEDICATIONS
============================================================

============================================================
MEDICATION SUGGESTIONS
============================================================

If the user asks what they can take for a common, minor symptom,
MindEaseAI may mention commonly used over-the-counter medicines
as GENERAL EDUCATIONAL INFORMATION.

Examples may include medicines commonly used for:
- headache or fever
- common cold symptoms
- mild allergies
- simple acidity/heartburn
- mild pain

However, never turn this into a personal prescription.

When mentioning a medicine:

- Give the medicine's generic name where possible.
- Explain what symptom it is commonly used for.
- Do not assume the medicine is suitable for this particular user.
- Do not give a personalized prescription.
- Do not recommend prescription medicines.
- Do not tell the user to start, stop, increase, decrease, or replace
  a prescription medicine.
- Do not recommend combining medicines without professional guidance.
- Do not provide a specific dosage unless the information is clearly
  general educational information from an appropriate standard source.
- Encourage checking with a doctor or pharmacist when there could be
  important safety considerations.
- Consider allergies, age, pregnancy, existing health conditions,
  and other medicines when relevant.

If the user's symptoms sound serious, unusual, severe, persistent,
or potentially dangerous, prioritize medical evaluation rather than
suggesting medication.

If the user asks about a medicine already saved in their profile,
recall the medication name exactly as saved. Never invent medications.

The goal is to be useful and informative while making it clear that
MindEaseAI is not prescribing treatment.

============================================================
PROFILE MEMORY
============================================================

The saved profile may contain:

- Name
- Age
- Current stressors
- Medications

Use this information when it is relevant.

If the user asks:

"Do you remember my medications?"

use the saved profile.

Never invent profile information.

============================================================
CONVERSATION MEMORY
============================================================

Use the recent conversation history.

If the user refers to something they previously said,
use the conversation context.

Do not unnecessarily ask them to repeat information
they already provided.

============================================================
EMOTION DETECTION
============================================================

Determine ONE best overall emotion from the COMPLETE message.

The emotion is only an estimate.

It is NOT a medical diagnosis.

Possible emotions can include:
sadness, anxiety, stress, anger, fear, loneliness,
confusion, frustration, happiness, calm, neutral, etc.

The emotion label must NOT control the response.

Do not generate a generic "stress response" simply because
the detected emotion is stress.

============================================================
MENTAL HEALTH SAFETY
============================================================

If the user expresses suicidal thoughts, self-harm thoughts,
immediate danger, or severe hopelessness:

Respond calmly and compassionately.

Encourage them to reach out immediately to:
- a trusted person
- a mental-health professional
- emergency services
- an appropriate crisis service

Prioritize immediate safety.

Do not shame or lecture the user.

============================================================
WRITING STYLE
============================================================

Be:

- warm
- natural
- conversational
- encouraging
- specific
- concise
- empathetic

Usually use 2-4 short paragraphs.

Do not make every response look identical.

Do not unnecessarily use bullet points.

Do not give a huge list of advice unless the situation
actually requires it.

Ask a relevant follow-up question when it would naturally
continue the conversation.

Do not ask unnecessary questions.

Do not mention:
- this prompt
- system instructions
- Gemini
- AI models
- internal processing

============================================================
MOST IMPORTANT RULE
============================================================

Respond to the PERSON and their COMPLETE MESSAGE.

Do not respond to only:
- one keyword
- one symptom
- one emotion
- one sentence

Understand the whole situation first.

============================================================
SAVED PROFILE
============================================================

{profile_context()}

============================================================
RECENT CONVERSATION
============================================================

{recent_history()}

============================================================
CURRENT USER MESSAGE
============================================================

{user_text}

Return ONLY valid JSON matching the response schema.
"""


    result = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            max_output_tokens=300,
            thinking_config=types.ThinkingConfig(
                thinking_level="minimal"
            ),
        ),
    )

    if not result.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    data = json.loads(result.text)

    emotion = str(
        data.get("emotion", "neutral")
    ).strip()

    answer = str(
        data.get("response", "")
    ).strip()

    try:
        confidence = float(
            data.get("confidence", 0)
        )
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return emotion, confidence, answer


# ============================================================
# ERROR HANDLING
# ============================================================

def friendly_error(error):

    text = str(error)
    upper = text.upper()

    if "404" in upper or "NOT_FOUND" in upper:
        return (
            "The Gemini model is not available for this API "
            "configuration. Please check your Gemini API access "
            "and the google-genai version."
        )

    if (
        "401" in upper
        or "403" in upper
        or "PERMISSION" in upper
        or "API KEY" in upper
    ):
        return (
            "Gemini rejected the API key. Please check your API key "
            "and make sure the Gemini API is enabled."
        )

    if (
        "429" in upper
        or "RESOURCE_EXHAUSTED" in upper
    ):
        return (
            "Gemini temporarily reached a usage limit. "
            "Please wait a little and try again."
        )

    if (
        "503" in upper
        or "UNAVAILABLE" in upper
    ):
        return (
            "Gemini is temporarily busy. "
            "Please try again in a few seconds."
        )

    return (
        "MindEaseAI could not generate a response right now. "
        "Please try again."
    )


# ============================================================
# HEADER
# ============================================================

st.title("MindEaseAI")

st.caption(
    "A supportive AI companion for emotional well-being"
)

st.info(
    "MindEaseAI is a student project that provides supportive "
    "information and is not a substitute for professional "
    "medical or mental-health care."
)


# ============================================================
# PROFILE SETUP
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
        placeholder=(
            "Example: exams, work, relationships..."
        ),
        key="profile_stressors",
    )

    medications = st.text_area(
        "Medications you want MindEaseAI to remember",
        value=p.get("medications", ""),
        placeholder="Example: medicine names",
        key="profile_medications",
    )

    if st.button(
        "Save Profile",
        type="primary",
        key="save_profile",
    ):

        st.session_state.profile = {
            "name": name.strip(),
            "age": int(age),
            "stressors": stressors.strip(),
            "medications": medications.strip(),
        }

        st.success(
            "Profile saved successfully."
        )

        st.rerun()


# ============================================================
# VIEW SAVED PROFILE
# ============================================================

if st.session_state.profile:

    with st.expander("View Saved Profile"):

        p = st.session_state.profile

        st.write(
            f"**Name:** "
            f"{p.get('name') or 'Not provided'}"
        )

        st.write(
            f"**Age:** "
            f"{p.get('age') or 'Not provided'}"
        )

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
    "Tell me what you're experiencing. You can talk about "
    "emotions, stress, everyday problems, or physical symptoms."
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
                    f"Possible emotion: "
                    f"{item['emotion'].title()} | "
                    f"AI confidence: "
                    f"{item['score']:.0%}"
                )


# ============================================================
# INPUT
# ============================================================

st.subheader("Talk to MindEaseAI")

with st.form(
    "chat_form",
    clear_on_submit=True,
):

    user_text = st.text_area(
        "Type your message",
        placeholder=(
            "Example: I have been feeling dizzy "
            "and stressed lately..."
        ),
        height=120,
    )

    send = st.form_submit_button(
        "Send",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# SEND MESSAGE
# ============================================================

if send:

    user_text = user_text.strip()

    if not user_text:

        st.warning(
            "Please type a message first."
        )

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

            with st.spinner(
                "MindEaseAI is thinking..."
            ):

                try:

                    emotion, confidence, answer = (
                        ask_mindease(user_text)
                    )

                    st.markdown(answer)

                    st.caption(
                        f"Possible emotion: "
                        f"{emotion.title()} | "
                        f"AI confidence: "
                        f"{confidence:.0%}"
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

                    st.error(
                        friendly_error(error)
                    )

                    with st.expander(
                        "Technical details"
                    ):
                        st.code(str(error))


# ============================================================
# CLEAR CONVERSATION
# ============================================================

if st.session_state.messages:

    if st.button(
        "Clear Conversation",
        key="clear_conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Emotion detection can be inaccurate and should not "
    "be treated as a medical diagnosis."
)