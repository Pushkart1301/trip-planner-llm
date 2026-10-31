import requests
import streamlit as st


BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="My Corrections", layout="wide")
st.title("My Corrections")

try:
    response = requests.get(
        f"{BASE_URL}/feedback",
        params={"user_id": "anonymous"},
        timeout=10,
    )
    response.raise_for_status()
    feedback_items = response.json().get("feedback", [])
except requests.RequestException as exc:
    st.error(f"Could not load corrections: {exc}")
    st.stop()

tabs = st.tabs(["All", "Pending", "Verified", "Not Verified"])
tab_statuses = {
    "All": None,
    "Pending": {"PENDING", "VERIFYING"},
    "Verified": {"VERIFIED"},
    "Not Verified": {"REJECTED", "UNCERTAIN", "NOT_FACTUAL"},
}
status_icons = {
    "PENDING": "⏳",
    "VERIFYING": "🔍",
    "VERIFIED": "✅",
    "REJECTED": "❌",
    "UNCERTAIN": "❓",
    "NOT_FACTUAL": "🚫",
}

for tab, (label, statuses) in zip(tabs, tab_statuses.items()):
    with tab:
        visible_items = [
            item
            for item in feedback_items
            if statuses is None or item["status"] in statuses
        ]
        if not visible_items:
            st.info("No corrections in this category yet.")
            continue

        for item in visible_items:
            with st.container(border=True):
                columns = st.columns([4, 1])
                with columns[0]:
                    st.write(item["feedback_text"])
                    st.caption(f"Submitted: {item['created_at'][:10]}")
                with columns[1]:
                    icon = status_icons.get(item["status"], "•")
                    st.write(f"{icon} {item['status']}")

                if st.button("Details", key=f"details_{item['feedback_id']}"):
                    st.query_params["id"] = item["feedback_id"]
                    st.switch_page("pages/correction_detail.py")
