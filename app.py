from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
import pickle


app = Flask(__name__, template_folder="templates")


model = pickle.load(open("Model\model.pkl","rb"))

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



def recommendation(gender, part_time_job, absence_days, extracurricular_activities,
       weekly_self_study_hour, math_score,history_score, physics_score, chemistry_score, 
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


@app.route("/")
def home():
    return render_template("home.html")
@app.route("/recommend")
def recommend():
    return render_template("recommend.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        part_time_job = request.form['part_time_job'] == 'true'
        absence_days = int(request.form['absence_days'])
        extracurricular_activities = request.form['extracurricular_activities'] == 'true'
        weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
        math_score = int(request.form['math_score'])
        history_score = int(request.form['history_score'])
        physics_score = int(request.form['physics_score'])
        chemistry_score = int(request.form['chemistry_score'])
        biology_score = int(request.form['biology_score'])
        english_score = int(request.form['english_score'])
        geography_score = int(request.form['geography_score'])
        total_score = float(request.form['total_score'])
        average_score = float(request.form['average_score'])

        recommendations = recommendation(gender, part_time_job, absence_days, extracurricular_activities,
                                          weekly_self_study_hours, math_score, history_score, physics_score,
                                          chemistry_score, biology_score, english_score, geography_score,
                                          total_score, average_score)

        return render_template('results.html', recommendations=recommendations)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)



