# Bot Detection Model

This project is a web application for training and predicting bot detection using various machine learning models. The frontend is built with React, and the backend is built with Flask. The application allows users to upload a CSV file for training the model and then use the trained model to predict whether a user is a bot or not.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Errors](#errors)

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js and npm
- pip for Python package management

### Backend Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/AmeyaMprojects/IITK_HACK_2
    
    ```
    go to the backend folder
    ```sh
    cd IITK_HACK_2/backend
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Flask application:
    ```sh
    python app.py
    ```

### Frontend Setup

1. Go to the root directory of the project IN A NEW TERNIMAL:

2. Install the required npm packages:
    ```sh
    npm install
    ```

3. Start the React application:
    ```sh
    npm run dev
    ```

## Usage

1. Open your browser and navigate to [http://localhost:80](http://localhost:80).
2. Upload a CSV file containing the training data. The CSV file should contain the following columns: [default_profile](http://_vscodecontentref_/1), [default_profile_image](http://_vscodecontentref_/2), [favourites_count](http://_vscodecontentref_/3), [followers_count](http://_vscodecontentref_/4), [friends_count](http://_vscodecontentref_/5), [screen_name](http://_vscodecontentref_/6), [statuses_count](http://_vscodecontentref_/7), [verified](http://_vscodecontentref_/8), [geo_enabled](http://_vscodecontentref_/9), [average_tweets_per_day](http://_vscodecontentref_/10), [account_age_days](http://_vscodecontentref_/11). The CSV file that we used is given in the zip file.
3. Click on the "Train Model" button to train the model.
4. After the model is trained, enter user information to predict whether the user is a bot or not.

## API Endpoints

### /train (POST)

- **Description**: Train the model with the uploaded CSV file.
- **Request**: Multipart form data with a CSV file.
- **Response**: JSON object containing the training results and performance metrics.

### /predict (POST)

- **Description**: Predict whether a user is a bot or not using the trained model.
- **Request**: JSON object containing user information.
- **Response**: JSON object containing the prediction result.

## Errors

- If facing any errors related to network or CORS, please check if the backend server is running.

## Demo

For a demonstration of the application, please watch the [demo video](https://drive.google.com/file/d/135WSNdyu9c8c-GlQcdFGNI4Ok_FwofpG/view?usp=sharing).