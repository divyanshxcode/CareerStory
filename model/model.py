import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

career_paths = [
    "Software Development",
    "Data Science",
    "Web Development",
    "UI/UX Design",
    "Digital Marketing",
    "Business Analysis"
]

# Create a dummy training dataset 
def create_dummy_dataset():
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'math_score': np.random.randint(1, 11, n_samples),
        'programming_skill': np.random.randint(1, 11, n_samples),
        'design_skill': np.random.randint(1, 11, n_samples),
        'communication_skill': np.random.randint(1, 11, n_samples),
        'analytical_skill': np.random.randint(1, 11, n_samples),
        'career': np.random.choice(career_paths, n_samples)
    }
    
    return pd.DataFrame(data)

# Train a simple model
def train_model():
    df = create_dummy_dataset()
    
    X = df.drop('career', axis=1)
    y = df['career']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# Predict career based on user input
def predict_career(user_data, model, scaler):
    user_df = pd.DataFrame([user_data])

    user_scaled = scaler.transform(user_df)
    
    prediction = model.predict(user_scaled)
    probabilities = model.predict_proba(user_scaled)
    
    # Get top 3 career recommendations
    proba_dict = {career_paths[i]: probabilities[0][i] for i in range(len(career_paths))}
    top_careers = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return top_careers
