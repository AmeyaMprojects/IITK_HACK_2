import logging
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# Global variables for the trained model and scaler
model = None
scaler = None

# Popularity metric function
def popularity_metric(friends_count: int, followers_count: int):
    return np.round(np.log(1 + friends_count) * np.log(1 + followers_count), 3)

# Function to process and train the models on uploaded CSV data
def train_model(df):
    global model, scaler

    logger.info("Starting model training...")

    # Preprocessing steps
    df['account_type'] = df['account_type'].replace({'human': 1, 'bot': 0})
    df['default_profile'] = df['default_profile'].astype(int)
    df['default_profile_image'] = df['default_profile_image'].astype(int)
    df['geo_enabled'] = df['geo_enabled'].astype(int)
    df['verified'] = df['verified'].astype(int)
    df = df.drop(columns=['location', 'profile_background_image_url', 'profile_image_url', 'screen_name', 'lang', 'id', 'Unnamed: 0', 'created_at', 'description'])

    # Compute popularity metric
    df["popularity"] = df.apply(lambda row: popularity_metric(row["friends_count"], row["followers_count"]), axis=1)

    # Prepare features and target
    target = df['account_type']
    features = df.drop(columns=['account_type'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    model_results = {}

    # Train and evaluate each model
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

        model_results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }

   # Create a combined plot with all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(model_results.keys())

    # Plotting the metrics
    metrics = ['precision', 'recall', 'f1', 'auc_roc']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Color palette
    bar_width = 0.2  # Width of the bars
    index = range(len(model_names))

    for i, metric in enumerate(metrics):
        ax.bar(
            [x + bar_width * i for x in index],  # Offset each metric slightly for better visualization
            [model_results[name][metric] for name in model_names],
            bar_width,
            label=metric.capitalize(),
            color=colors[i]
        )

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison: Precision, Recall, F1 Score, and AUC-ROC')
    ax.set_xticks([x + bar_width * 1.5 for x in index])
    ax.set_xticklabels(model_names)
    ax.legend()

    # Add grid lines for better visibility
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adding numbers to the bars for better visibility
    for i, metric in enumerate(metrics):
        for j, model_name in enumerate(model_names):
            ax.text(
                j + bar_width * i,  # Positioning of text
                model_results[model_name][metric] + 0.02,  # Slightly above the bar
                f"{model_results[model_name][metric]:.2f}",  # Format number
                ha='center', va='bottom', color='black', fontsize=10
            )

    # Adjust layout to avoid label overlap
    plt.tight_layout()

    # Save the plot to a BytesIO object and encode as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    return model_results, img_base64

@app.route('/train', methods=['POST'])
def train():
    logger.info("Received request to train the model.")
    if 'file' not in request.files:
        logger.error("No file uploaded.")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected.")
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read the uploaded CSV file
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        logger.info(f"CSV file uploaded: {file.filename}")

        # Train the models and get evaluation results
        model_results, plot_img = train_model(df)

        # Return the model evaluation results and the plot image
        return jsonify({
            'model_results': model_results,
            'roc_auc_plot': plot_img
        })

    except Exception as e:
        logger.error(f"Error during training process: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the model and scaler are trained
        if model is None or scaler is None:
            logger.error("Model or scaler is not trained. Please train the model first.")
            return jsonify({'error': 'Model or scaler is not trained. Please train the model first.'}), 400

        data = request.json
        logger.info(f"Received data for prediction: {data}")

        # Convert '0'/'1' strings to integers
        data['default_profile'] = int(data['default_profile'])
        data['default_profile_image'] = int(data['default_profile_image'])
        data['verified'] = int(data['verified'])
        data['geo_enabled'] = int(data['geo_enabled'])

        # Calculate popularity metric (logarithmic feature)
        popularity = np.round(np.log(1 + int(data['friends_count'])) * np.log(1 + int(data['followers_count'])), 3)

        # Prepare the input data for prediction (e.g., preprocess the data)
        input_data = np.array([[
            int(data['default_profile']),
            int(data['default_profile_image']),
            int(data['favourites_count']),
            int(data['followers_count']),
            int(data['friends_count']),
            int(data['geo_enabled']),
            int(data['statuses_count']),
            int(data['verified']),
            float(data['average_tweets_per_day']),
            int(data['account_age_days']),
            popularity  # Add the popularity metric
        ]])

        # Log the input data for debugging
        logger.info(f"Input data for prediction: {input_data}")

        # Standardize input (same as during model training)
        input_data_scaled = scaler.transform(input_data)  # Use the scaler from training

        # Predict using the trained model
        prediction = model.predict(input_data_scaled)

        # Convert the prediction (which is a numpy int) to a regular Python int
        prediction = int(prediction[0])

        # Log prediction result
        logger.info(f"Prediction result: {prediction}")

        return jsonify({'prediction': prediction})  # 1 = Bot, 0 = Not a Bot

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True)