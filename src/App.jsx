import React, { useState } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import TextScramble from '@skits/react-text-scramble';

const App = () => {
  const [username, setUsername] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [customKey, setCustomKey] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [botProbability, setBotProbability] = useState(null);
  const [humanProbability, setHumanProbability] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tweetData, setTweetData] = useState([]);

  const handleUsernameChange = (e) => setUsername(e.target.value);
  const handleApiKeyChange = (e) => setApiKey(e.target.value);
  const handleCustomKeyChange = (e) => setCustomKey(e.target.value);
  
  const addCustomKey = () => {
    if (customKey.trim()) {
      setApiKey(customKey.trim());
      setCustomKey("");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username) {
      setError("Please enter a Twitter handle.");
      return;
    }
    setLoading(true);
    setError(null);
    setPrediction(null);
    setTweetData([]);

    try {
      const response = await axios.post("https://iitk-hack-2.onrender.com/analyze", {
        twitter_handle: username,
        bearer_token: apiKey,
      });
      setPrediction(response.data.prediction);
      setBotProbability(response.data.bot_probability);
      setHumanProbability(response.data.human_probability);
      setTweetData(response.data.tweet_activity);
    } catch (err) {
      setError("Error fetching data or predicting. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">
        <TextScramble
        text="Twitter Bot Detector"
        revealDelay={0.5}
        revealText="True"
        />
      </h1>
      <form onSubmit={handleSubmit}>
        <label>Enter Twitter Username:</label>
        <input type="text" value={username} onChange={handleUsernameChange} placeholder="e.g. jack" required />
        
        <label>Enter API Key:</label>
        <input type="text" value={apiKey} onChange={handleApiKeyChange} placeholder="Enter your API key" />

        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Check Bot Status"}
        </button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}
      {prediction !== null && (
        <div>
          <h2>Prediction Result</h2>
          <p>{botProbability > humanProbability ? `${botProbability}% Bot` : `${humanProbability}% Human`}</p>
        </div>
      )}

      {tweetData.length > 0 && (
        <div>
          <h2>Tweet Activity (Last 10 Days)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={tweetData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.3)"/>
              <XAxis dataKey="date" stroke="#ffffff"/>
              <YAxis stroke="#ffffff"/>
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#00f0ff" strokeWidth={2} dot={{fill:"#00f0ff"}} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default App;
