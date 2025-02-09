import React, { useState } from "react";
import axios from "axios";

const backendURL = "https://iitk-hack-2.onrender.com";

const App = () => {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [rocAucPlot, setRocAucPlot] = useState(null);
  const [formData, setFormData] = useState({
    default_profile: "",
    default_profile_image: "",
    favourites_count: "",
    followers_count: "",
    friends_count: "",
    statuses_count: "",
    verified: "",
    geo_enabled: "",
    average_tweets_per_day: "",
    account_age_days: "",
  });

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleTrain = async () => {
    if (!file) {
      setError("Please upload a CSV file.");
      return;
    }
  
    setLoading(true);
    setError(null);
    setResults(null);
    setRocAucPlot(null);
  
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      const response = await axios.post(`${backendURL}/train`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 180000,  // Increase timeout to 3 minutes
      });
  
      console.log("Response:", response);  
      if (response.data) {
        setResults(response.data.model_results);
        setRocAucPlot(response.data.roc_auc_plot);
      } else {
        setError("No data received from the server.");
      }
    } catch (err) {
      console.error("Error:", err);
      setError(err.response?.data?.error || "Unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };
  
  

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmitForm = async (e) => {
    e.preventDefault();
    const formattedData = {
      ...formData,
      default_profile: parseInt(formData.default_profile, 10),
      default_profile_image: parseInt(formData.default_profile_image, 10),
      favourites_count: parseInt(formData.favourites_count, 10),
      followers_count: parseInt(formData.followers_count, 10),
      friends_count: parseInt(formData.friends_count, 10),
      statuses_count: parseInt(formData.statuses_count, 10),
      verified: parseInt(formData.verified, 10),
      geo_enabled: parseInt(formData.geo_enabled, 10),
      average_tweets_per_day: parseFloat(formData.average_tweets_per_day),
      account_age_days: parseInt(formData.account_age_days, 10),
    };

    try {
      const response = await axios.post(`${backendURL}/predict`, formattedData, {
        headers: { "Content-Type": "application/json" },
        timeout: 180000,      
      });
      setResults({ ...results, prediction: response.data.prediction });
    } catch (err) {
      setError("Error during prediction. Please try again.");
    }
  };

  return (
    <div className="container">
      <h1>Train and Predict Bot Detection</h1>

      {/* File upload input */}
      <div className="file-upload-container">
        <input
          type="file"
          id="file-upload"
          accept=".csv"
          onChange={handleFileChange}
          className="file-upload"
        />
        <label htmlFor="file-upload" className="file-label">
          Upload CSV
        </label>
        {file && <span className="file-name">{file.name}</span>}
      </div>

      {/* Train button */}
      <button onClick={handleTrain} disabled={loading}>
        {loading ? "Training..." : "Train Model"}
      </button>

      {/* Display error message if any */}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {/* Display ROC-AUC plot */}
      {rocAucPlot && (
        <div>
          <h3>Model Comparison: Precision, Recall, F1, and AUC-ROC</h3>
          <img
            src={`data:image/png;base64,${rocAucPlot}`}
            alt="Model Comparison Plot"
          />
        </div>
      )}

      {/* Display results after model is trained */}
      {results && (
        <div>
          {/* Display the form after model is trained */}
          <h3>Enter User Information for Prediction:</h3>
          <form onSubmit={handleSubmitForm}>
            <div>
              <label>Default Profile</label>
              <select
                name="default_profile"
                value={formData.default_profile}
                onChange={handleFormChange}
                required
              >
                <option value="">Select</option>
                <option value="1">True</option>
                <option value="0">False</option>
              </select>
            </div>
            <div>
              <label>Default Profile Image</label>
              <select
                name="default_profile_image"
                value={formData.default_profile_image}
                onChange={handleFormChange}
                required
              >
                <option value="">Select</option>
                <option value="1">True</option>
                <option value="0">False</option>
              </select>
            </div>
            <div>
              <label>Favourites Count</label>
              <input
                type="number"
                name="favourites_count"
                value={formData.favourites_count}
                onChange={handleFormChange}
                required
                min="0"
                step="0.1"
              />
            </div>
            <div>
              <label>Followers Count</label>
              <input
                type="number"
                name="followers_count"
                value={formData.followers_count}
                onChange={handleFormChange}
                required
                min="0"
                step="0.1"
              />
            </div>
            <div>
              <label>Friends Count</label>
              <input
                type="number"
                name="friends_count"
                value={formData.friends_count}
                onChange={handleFormChange}
                required
                min="0"
                step="0.1"
              />
            </div>
            <div>
              <label>Geo Enabled</label>
              <select
                name="geo_enabled"
                value={formData.geo_enabled}
                onChange={handleFormChange}
                required
              >
                <option value="">Select</option>
                <option value="1">True</option>
                <option value="0">False</option>
              </select>
            </div>
            <div>
              <label>Statuses Count</label>
              <input
                type="number"
                name="statuses_count"
                value={formData.statuses_count}
                onChange={handleFormChange}
                required
                min="0"
              />
            </div>
            <div>
              <label>Verified</label>
              <select
                name="verified"
                value={formData.verified}
                onChange={handleFormChange}
                required
              >
                <option value="">Select</option>
                <option value="1">True</option>
                <option value="0">False</option>
              </select>
            </div>
            <div>
              <label>Average Tweets per Day</label>
              <input
                type="number"
                name="average_tweets_per_day"
                value={formData.average_tweets_per_day}
                onChange={handleFormChange}
                required
                min="0"
                step="0.1"
              />
            </div>
            <div>
              <label>Account Age (in days)</label>
              <input
                type="number"
                name="account_age_days"
                value={formData.account_age_days}
                onChange={handleFormChange}
                required
                min="0"
              />
            </div>
            <div>
              <label>Screen Name</label>
              <input
                type="text"
                name="screen_name"
                value={formData.screen_name}
                onChange={handleFormChange}
                required
              />
            </div>

            <button type="submit">Predict</button>
          </form>

          {/* Display prediction result below the form */}
          {results.prediction !== undefined && (
            <div>
              <h2>Prediction Result:</h2>
              <p>{results.prediction === 1 ? "Not Bot" : "Bot"}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default App;
