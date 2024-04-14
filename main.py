import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open("Model\model.pkl", "rb"))

# Define the professions
professions = [
    "Accountant",
    "Artist",
    "Banker",
    "Business Owner",
    "Construction Engineer",
    "Designer",
    "Doctor",
    "Game Developer",
    "Government Officer",
    "Lawyer",
    "Real Estate Developer",
    "Scientist",
    "Software Engineer",
    "Stock Investor",
    "Teacher",
    "Unknown",
    "Writer"
]

# Define the recommendation function
def recommendation(gender, part_time_job, absence_days, extracurricular_activities,
       weekly_self_study_hour, math_score, history_score, physics_score, chemistry_score, 
       biology_score, english_score, geography_score, total_score, average_score):
    
    gender_encoded = 1 if gender.lower() =='male' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0
    
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hour, math_score, history_score, physics_score, chemistry_score,
                               biology_score, english_score, geography_score, total_score, average_score]])
    
    probabilities = model.predict_proba(feature_array)
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(professions[idx], probabilities[0][idx]) for idx in top_classes_idx]
    return top_classes_names_probs

# Streamlit app
def main():
    st.title("Carieer Recommendation System")
    
    gender = st.radio("Gender", options=["Male", "Female"])
    part_time_job = st.checkbox("Part-Time Job")
    absence_days = st.number_input("Absence Days", min_value=0, max_value=30, step=1)
    extracurricular_activities = st.checkbox("Extracurricular Activities")
    weekly_self_study_hours = st.number_input("Weekly Self-Study Hours", min_value=0, max_value=100, step=1)
    math_score = st.number_input("Math Score", min_value=0, max_value=100, step=1)
    history_score = st.number_input("History Score", min_value=0, max_value=100, step=1)
    physics_score = st.number_input("Physics Score", min_value=0, max_value=100, step=1)
    chemistry_score = st.number_input("Chemistry Score", min_value=0, max_value=100, step=1)
    biology_score = st.number_input("Biology Score", min_value=0, max_value=100, step=1)
    english_score = st.number_input("English Score", min_value=0, max_value=100, step=1)
    geography_score = st.number_input("Geography Score", min_value=0, max_value=100, step=1)
    total_score = st.number_input("Total Score", min_value=0, max_value=1000, step=1)
    average_score = st.number_input("Average Score", min_value=0.0, max_value=100.0, step=0.01)

    if st.button("Get Recommendations"):
        recommendations = recommendation(gender, part_time_job, absence_days, extracurricular_activities,
                                          weekly_self_study_hours, math_score, history_score, physics_score,
                                          chemistry_score, biology_score, english_score, geography_score,
                                          total_score, average_score)
        
        st.subheader("Top Recommendations")
        for profession, probability in recommendations:
            st.write(f"- {profession}: {probability * 100:.2f}%")

if __name__ == "__main__":
    main()
