import streamlit as st
import pandas as pd
import joblib
import os

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "student_performance_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("AI-Based Student Performance Prediction System")

reading_score = st.number_input("Reading Score", 0, 100, 70)
writing_score = st.number_input("Writing Score", 0, 100, 70)

gender = st.selectbox("Gender", ["female", "male"])
lunch = st.selectbox("Lunch", ["free/reduced", "standard"])
test_prep = st.selectbox("Test Preparation", ["completed", "none"])

gender_male = 1 if gender == "male" else 0
lunch_standard = 1 if lunch == "standard" else 0
test_prep_none = 1 if test_prep == "none" else 0

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("🚀 Predict Math Score")

with col2:
    reset_button = st.button("🔄 Reset")

if predict_button:

    input_data = pd.DataFrame({
        "reading score": [reading_score],
        "writing score": [writing_score],
        "gender_male": [gender_male],
        "lunch_standard": [lunch_standard],
        "test preparation course_none": [test_prep_none]
    })

    prediction = model.predict(input_data)

    predicted_score = prediction[0]

    st.success(f"Predicted Math Score: {predicted_score:.2f}")

    # ADD GRAPH CODE HERE
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    subjects = ["Reading", "Writing", "Predicted Math"]
    scores = [reading_score, writing_score, predicted_score]

    ax.bar(subjects, scores)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Marks")
    ax.set_title("Student Performance Comparison")

    st.pyplot(fig)

    # Overall Performance Analysis
    overall_score = (predicted_score + reading_score + writing_score) / 3

    if overall_score >= 80:
       performance = "🌟 Excellent"
    elif overall_score >= 60:
       performance = "👍 Good"
    elif overall_score >= 40:
       performance = "📘 Average"
    else:
       performance = "📚 Needs Improvement"

    st.subheader(f"Overall Performance: {performance}")
    st.write(f"Overall Average Score: {overall_score:.2f}")

    # Student Risk Detection
    if overall_score < 40:
        st.error("🔴 Risk Level: HIGH")
    elif overall_score < 70:
        st.warning("🟡 Risk Level: MEDIUM")
    else:
         st.success("🟢 Risk Level: LOW")

    st.subheader("📅 AI Personalized Study Plan")

    if overall_score < 40:
      st.write("📌 Monday: Revise Mathematics Basics")
      st.write("📌 Tuesday: Solve 20 Practice Questions")
      st.write("📌 Wednesday: Reading Comprehension Practice")
      st.write("📌 Thursday: Writing Exercises")
      st.write("📌 Friday: Mock Test")

    elif overall_score < 70:
      st.write("📌 Monday: Solve Previous Year Questions")
      st.write("📌 Tuesday: Practice Weak Topics")
      st.write("📌 Wednesday: Revision Session")
      st.write("📌 Thursday: Timed Practice Test")
      st.write("📌 Friday: Performance Review")

    else:
      st.write("📌 Monday: Advanced Problems")
      st.write("📌 Tuesday: Competitive Questions")
      st.write("📌 Wednesday: Mock Exam")
      st.write("📌 Thursday: Review Mistakes")
      st.write("📌 Friday: Concept Reinforcement") 

    st.subheader("📊 Performance Dashboard")

    st.write("Reading Score")
    st.progress(int(reading_score))

    st.write("Writing Score")
    st.progress(int(writing_score))

    st.write("Predicted Math Score")
    st.progress(int(predicted_score))


    # AI Study Advisor
    if predicted_score < 40:
        st.error("📚 AI Study Advisor")
        st.write("• Study at least 2 extra hours daily.")
        st.write("• Focus on Mathematics basics and formulas.")
        st.write("• Solve 10 practice questions every day.")

    elif predicted_score < 70:
        st.warning("🎯 AI Study Advisor")
        st.write("• Revise weak chapters regularly.")
        st.write("• Practice previous year questions.")
        st.write("• Take weekly mock tests.")

    else:
        st.success("🏆 AI Study Advisor")
        st.write("• Excellent performance!")
        st.write("• Continue your current study strategy.")
        st.write("• Try advanced-level problems.")

if reset_button:
    st.rerun()