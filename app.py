import streamlit as st
import random
import time
import numpy as np
import pickle

# --- Load trained ML model and scaler ---
model = pickle.load(open("dementia_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# --- Stroop Test Setup ---
colors = ["RED", "BLUE", "GREEN", "YELLOW", "PURPLE"]
actual_colors = ["red", "blue", "green", "yellow", "purple"]

def run_stroop_test(num_questions=5):
    st.write("### Stroop Test")
    
    # Initialize session state variables
    if "question_index" not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.correct = 0
        st.session_state.wrong = 0
        st.session_state.reaction_times = []
        st.session_state.current_word = random.choice(colors)
        st.session_state.current_color = random.choice(actual_colors)
        st.session_state.start_time = time.time()
        st.session_state.finished = False
    
    if st.session_state.finished:
        avg_rt = np.mean(st.session_state.reaction_times)
        stroop_score = st.session_state.correct*4 - st.session_state.wrong*2
        return avg_rt, st.session_state.correct, st.session_state.wrong, stroop_score

    st.write(f"**Question {st.session_state.question_index+1}/{num_questions}**")
    st.write(f"Word: **{st.session_state.current_word}**")

    user_answer = st.radio("Choose the COLOR of the word:", actual_colors, key="answer")
    submit = st.button("Submit Answer")

    if submit:
        reaction_time = time.time() - st.session_state.start_time
        st.session_state.reaction_times.append(reaction_time)

        if user_answer == st.session_state.current_color:
            st.session_state.correct += 1
        else:
            st.session_state.wrong += 1

        st.session_state.question_index += 1

        if st.session_state.question_index < num_questions:
            st.session_state.current_word = random.choice(colors)
            st.session_state.current_color = random.choice(actual_colors)
            st.session_state.start_time = time.time()
        else:
            st.session_state.finished = True
            avg_rt = np.mean(st.session_state.reaction_times)
            stroop_score = st.session_state.correct*4 - st.session_state.wrong*2
            return avg_rt, st.session_state.correct, st.session_state.wrong, stroop_score

    return None

# --- Streamlit App UI ---
st.set_page_config(page_title="Dementia Detection", page_icon="🧠", layout="centered")

st.title("🧠 Dementia Detection Using Stroop Test")
st.write("""
This website uses a trained Machine Learning model + Stroop Test performance 
to estimate early dementia risk.
""")

age = st.number_input("Enter your age", 40, 90)

result = run_stroop_test(num_questions=5)

if result:
    avg_rt, correct, wrong, stroop_score = result

    st.write("### Your Stroop Test Summary")
    st.write(f"**Average Reaction Time:** {avg_rt:.2f} sec")
    st.write(f"**Correct Answers:** {correct}")
    st.write(f"**Wrong Answers:** {wrong}")
    st.write(f"**Stroop Score:** {stroop_score}")

    # ML prediction
    user_data = np.array([[age, avg_rt, correct, wrong, stroop_score]])
    user_data_scaled = scaler.transform(user_data)
    pred = model.predict(user_data_scaled)[0]

    st.write("---")
    st.write("## 🧠 Dementia Risk Result")

    if pred == 1:
        st.error("⚠ High probability of Cognitive Impairment / Dementia")
    else:
        st.success("✅ Low probability of Dementia")

    st.write("**Disclaimer:** This is for educational purposes and not a medical diagnosis.")
