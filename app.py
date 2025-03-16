from flask import Flask, render_template, request
from model.model import train_model, predict_career

app = Flask(__name__)

model, scaler = train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_data = {
        'math_score': int(request.form['math_score']),
        'programming_skill': int(request.form['programming_skill']),
        'design_skill': int(request.form['design_skill']),
        'communication_skill': int(request.form['communication_skill']),
        'analytical_skill': int(request.form['analytical_skill'])
    }
    
    age = int(request.form['age'])
    years_to_workforce = 24 - age if age < 24 else 0
    
    financial_status = request.form['financial_status']
    
    # Make prediction
    career_predictions = predict_career(user_data, model, scaler)
    
    job_market = {}
    for career, _ in career_predictions:
        if career == "Software Development" or career == "Data Science":
            growth = "High growth expected (15-20% in the next 5 years)"
        elif career == "Web Development" or career == "UI/UX Design":
            growth = "Moderate growth expected (10-15% in the next 5 years)"
        else:
            growth = "Stable growth expected (5-10% in the next 5 years)"
        job_market[career] = growth
    
    education_rec = {}
    for career, _ in career_predictions:
        if career == "Data Science":
            education_rec[career] = "Master's degree recommended"
        elif career == "Software Development":
            if financial_status == "high":
                education_rec[career] = "Bachelor's degree with specialized certifications"
            else:
                education_rec[career] = "Bachelor's degree with self-learning"
        else:
            education_rec[career] = "Bachelor's degree is sufficient"
    
    return render_template('result.html', 
                          predictions=career_predictions,
                          job_market=job_market,
                          education_rec=education_rec,
                          years_to_workforce=years_to_workforce)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
