import re

import requests
import streamlit as st


BASE_URL = "http://localhost:8000"


def extract_destination(question: str) -> str | None:
    """Extract a likely destination following common travel prepositions."""
    match = re.search(
        r"\b(?:in|to|around|near|visit(?:ing)?)\s+"
        r"([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})",
        question,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    destination_words = re.split(r"\s+", match.group(1).strip())
    stop_words = {
        "and",
        "at",
        "by",
        "during",
        "for",
        "in",
        "on",
        "this",
        "to",
        "with",
    }
    destination_words = destination_words[: next(
        (
            index
            for index, word in enumerate(destination_words)
            if word.casefold() in stop_words
        ),
        len(destination_words),
    )]
    destination = " ".join(destination_words)
    return destination.strip().title() or None


st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🌍 AI Trip Planner")
st.header("How can I help you plan a trip?")

with st.sidebar:
    st.title("AI Trip Planner")
    if st.button("My Corrections", use_container_width=True):
        st.switch_page("pages/my_corrections.py")
    st.caption("Corrections are checked against sources before they can affect answers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_notice" in st.session_state:
    st.success(st.session_state.pop("feedback_notice"))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("message_id"):
            if st.button(
                "Suggest a correction",
                key=f"correct_{message['message_id']}",
            ):
                st.session_state.correction_message_id = message["message_id"]

if st.session_state.get("correction_message_id"):
    selected_message_id = st.session_state.correction_message_id
    selected_message = next(
        (
            message
            for message in st.session_state.messages
            if message.get("message_id") == selected_message_id
        ),
        None,
    )
    if selected_message is not None:
        with st.expander("Suggest a correction", expanded=True):
            with st.form(key=f"feedback_{selected_message_id}", clear_on_submit=True):
                st.write("What is incorrect?")
                feedback_text = st.text_area(
                    "Describe what's wrong",
                    key=f"feedback_text_{selected_message_id}",
                )
                correction = st.text_area(
                    "What should it say instead? (optional)",
                    key=f"correction_{selected_message_id}",
                )
                explanation = st.text_area(
                    "Additional context (optional)",
                    key=f"explanation_{selected_message_id}",
                )
                submitted = st.form_submit_button("Submit correction")

            if submitted:
                if not feedback_text.strip():
                    st.error("Please describe the correction before submitting.")
                else:
                    try:
                        response = requests.post(
                            f"{BASE_URL}/feedback",
                            json={
                                "question": selected_message.get("question", ""),
                                "answer": selected_message["content"],
                                "message_id": selected_message_id,
                                "feedback_text": feedback_text.strip(),
                                "suggested_correction": correction.strip() or None,
                                "destination": extract_destination(
                                    selected_message.get("question", "")
                                ),
                                "explanation": explanation.strip() or None,
                            },
                            timeout=10,
                        )
                        response.raise_for_status()
                        result = response.json()
                        st.session_state.last_feedback_id = result["feedback_id"]
                        st.session_state.feedback_notice = result["message"]
                        st.session_state.pop("correction_message_id", None)
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(f"Failed to submit correction: {exc}")

question = st.chat_input("Where would you like to go?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Planning your trip..."):
            response = requests.post(
                f"{BASE_URL}/query",
                json={"question": question},
                timeout=120,
            )
        response.raise_for_status()
        result = response.json()
        answer = result.get("answer", "No answer returned.")
        message_id = result.get("message_id")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "message_id": message_id,
                "question": question,
            }
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
            if message_id and st.button(
                "Suggest a correction",
                key=f"correct_{message_id}",
            ):
                st.session_state.correction_message_id = message_id
    except requests.RequestException as exc:
        st.error(f"Could not reach the trip planner: {exc}")
