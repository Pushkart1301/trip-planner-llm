import requests
import streamlit as st


BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Correction Details", layout="wide")
st.title("Correction Details")

feedback_id = st.query_params.get("id")
if not feedback_id:
    st.error("No correction was selected.")
    st.stop()

try:
    response = requests.get(f"{BASE_URL}/feedback/{feedback_id}", timeout=10)
    response.raise_for_status()
    feedback = response.json()
except requests.RequestException as exc:
    st.error(f"Could not load correction details: {exc}")
    st.stop()

st.write(f"**Status:** {feedback['status']}")
st.write(f"**Submitted:** {feedback['created_at'][:10]}")
st.write("**Correction:**")
st.write(feedback["feedback_text"])
if feedback.get("suggested_correction"):
    st.write("**Suggested replacement:**")
    st.write(feedback["suggested_correction"])
if feedback.get("explanation"):
    st.write("**Additional context:**")
    st.write(feedback["explanation"])
if feedback.get("original_question"):
    st.write("**Original question:**")
    st.write(feedback["original_question"])
if feedback.get("original_answer"):
    st.write("**Original answer:**")
    st.markdown(feedback["original_answer"])
if feedback.get("policy_decision"):
    st.write(f"**Policy decision:** {feedback['policy_decision']}")
if feedback.get("claim"):
    st.write("**Extracted claim:**")
    st.json(feedback["claim"])

if st.button("Back to My Corrections"):
    st.switch_page("pages/my_corrections.py")
