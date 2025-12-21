import streamlit as st
import random
import time
import pandas as pd
import numpy as np
import pickle

# ---------------- Load ML model and scaler ----------------
model = pickle.load(open("dementia_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- Stroop Config ----------------
COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW"]
COLOR_HEX = {
    "RED": "#d62828",
    "GREEN": "#2a9d8f",
    "BLUE": "#0077b6",
    "YELLOW": "#f4d35e"
}
NUM_QUESTIONS = 20
ISI = 0.3  # inter-trial interval

# ---------------- Helper Functions ----------------
def make_trial():
    word = random.choice(COLOR_NAMES)
    ink = random.choice(COLOR_NAMES)
    return {"word": word, "ink": ink}

def show_stimulus(trial):
    st.markdown(
        f"<div style='text-align:center; margin-top:50px;'>"
        f"<span style='font-size:100px; font-weight:700; color:{COLOR_HEX[trial['ink']]};'>{trial['word']}</span>"
        f"</div>", unsafe_allow_html=True
    )

def record_response(trial, response, rt):
    correct = int(response.upper() == trial["ink"][0])
    st.session_state.results.append({
        "word": trial["word"],
        "ink": trial["ink"],
        "response": response.upper(),
        "correct": correct,
        "rt": rt
    })

# ---------------- Session State Init ----------------
if "stage" not in st.session_state:
    st.session_state.stage = "instructions"
if "trials" not in st.session_state:
    st.session_state.trials = [make_trial() for _ in range(NUM_QUESTIONS)]
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "results" not in st.session_state:
    st.session_state.results = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "user_age" not in st.session_state:
    st.session_state.user_age = None

st.set_page_config(page_title="Cognitive Decline Detection Stroop Test", layout="centered")

st.title("🧠 Cognitive Decline Detection Using Stroop Test")
st.write("Select the **ink color** of the word as fast and accurately as possible.")

# ---------------- Instructions ----------------
if st.session_state.stage == "instructions":
    st.header("Instructions")
    st.markdown("""
        - You will see a color word displayed in colored ink.
        - Select the **color of the ink**, not the text.
        - Try to respond quickly and accurately.
    """)

    # ---------------- Age input ----------------
    age_input = st.number_input(
        "Enter your age (years):", min_value=10, max_value=120, step=1
    )
    st.session_state.user_age = age_input

    if st.button("Start Test") and st.session_state.user_age is not None:
        st.session_state.stage = "test"
        st.session_state.current_idx = 0
        st.session_state.results = []
        st.session_state.start_time = None
        st.rerun()

# ---------------- Test ----------------
elif st.session_state.stage == "test":
    idx = st.session_state.current_idx
    if idx >= NUM_QUESTIONS:
        st.session_state.stage = "results"
        st.rerun()
    else:
        trial = st.session_state.trials[idx]
        st.write(f"Test {idx+1} / {NUM_QUESTIONS}")
        show_stimulus(trial)

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        cols = st.columns(4)
        clicked = None
        for i, color in enumerate(COLOR_NAMES):
            if cols[i].button(color):
                clicked = color[0]

        if clicked:
            rt = time.time() - st.session_state.start_time
            record_response(trial, clicked, rt)
            st.session_state.current_idx += 1
            st.session_state.start_time = None
            time.sleep(ISI)
            st.rerun()

# ---------------- Results ----------------
elif st.session_state.stage == "results":
    st.header("Results")
    df = pd.DataFrame(st.session_state.results)
    
    if df.empty:
        st.info("No data collected.")
    else:
        avg_rt = df["rt"].mean()
        correct = df["correct"].sum()
        wrong = len(df) - correct
        stroop_score = correct*4 - wrong*2

        st.write("### Stroop Test Summary")
        st.metric("Correct Answers", f"{correct} / {NUM_QUESTIONS}")
        st.metric("Wrong Answers", wrong)
        st.metric("Average Reaction Time (s)", f"{avg_rt:.3f}")
        

        # ML prediction using age collected at the start
        user_age = st.session_state.user_age
        user_data = np.array([[user_age, avg_rt, correct, wrong, stroop_score]])
        user_scaled = scaler.transform(user_data)
        pred = model.predict(user_scaled)[0]

        st.write("---")
        st.write("## Cognitive Decline Risk Result")
        if pred == 1:
            st.error("🔴 Red Zone – High risk of Cognitive Decline")
        else:
            st.success("🟢 Green Zone - Low risk of Cognitive Decline")
        st.caption("⚠ This result is based on a screening test and is not a medical diagnosis.")

        # Trial-level table & download
        st.write("### Test Data")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", data=csv, file_name="stroop_results.csv", mime="text/csv")

    if st.button("Restart Test"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()





