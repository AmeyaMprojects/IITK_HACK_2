import logging
import tweepy
from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# Load the pretrained model
MODEL_PATH = "best_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info("Pretrained model loaded successfully.")
else:
    logger.error("Model file not found! Ensure it exists in the directory.")
    model = None

# Twitter API credentials (replace with your own)
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAF%2FBzAEAAAAAN%2FXfx98dMbPckR9EpBWmllfQR8g%3DNMKVeabrfux7WSMpmQsGRoifMOr81zoiubGMTMVSkQbkI3tVzt"

# Initialize Tweepy client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Predict bot or human and fetch tweet activity
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if model is None:
            return jsonify({'error': 'Model is not available'}), 400

        data = request.json
        username = data.get("twitter_handle")
        if not username:
            return jsonify({'error': 'Twitter handle is required'}), 400

        # Fetch user data from Twitter API
        user = client.get_user(username=username, user_fields=["public_metrics", "verified", "created_at"])
        user_info = user.data
        metrics = user_info.public_metrics

        # Extract features
        verified = int(user_info.verified)
        followers_count = int(metrics["followers_count"])
        friends_count = int(metrics["following_count"])
        statuses_count = int(metrics["tweet_count"])
        account_age_days = (pd.Timestamp.utcnow() - pd.Timestamp(user_info.created_at)).days
        average_tweets_per_day = round(statuses_count / account_age_days, 2) if account_age_days > 0 else 0
        popularity = np.round(np.log(1 + friends_count) * np.log(1 + followers_count), 3)

        # Prepare input data
        input_data = np.array([[
            followers_count, friends_count, statuses_count, verified,
            average_tweets_per_day, account_age_days, popularity
        ]])

        # Get prediction probability
        probabilities = model.predict_proba(input_data)[0]  # Get probability scores
        bot_prob = round(probabilities[0] * 100, 2)  # Probability of being a bot
        human_prob = round(probabilities[1] * 100, 2)  # Probability of being human
        prediction = int(model.predict(input_data)[0])  # Final prediction

        # Get tweet activity
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=10)
        start_time = start_date.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        end_time = end_date.isoformat(timespec='milliseconds').replace('+00:00', 'Z')

        tweets = client.get_users_tweets(
            id=user_info.id,
            start_time=start_time,
            end_time=end_time,
            max_results=100,
            tweet_fields=["created_at"]
        )

        tweet_counts = defaultdict(int)
        if tweets.data:
            for tweet in tweets.data:
                if tweet.created_at:
                    tweet_date = tweet.created_at.date()
                    tweet_counts[tweet_date] += 1

        tweet_activity = []
        for i in range(10):
            date = (end_date - timedelta(days=i)).date()
            tweet_activity.append({"date": str(date), "count": tweet_counts[date]})

        return jsonify({
            'twitter_handle': username,
            'prediction': prediction,  # 0 = Bot, 1 = Human
            'bot_probability': bot_prob,
            'human_probability': human_prob,
            'tweet_activity': tweet_activity
        })
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True)
