// filepath: /C:/Users/ameya/Desktop/projects/stack/IITK_HACK_2/src/App.jsx
import React, { useState } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import TextScramble from '@skits/react-text-scramble';
import './index.css';

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
    <div className="container mx-auto p-8 bg-white bg-opacity-50 rounded-lg shadow-lg text-center">
      <h1 className="font-bold text-6xl mb-8">
        <TextScramble
          text="Twitter Bot Detector"
          revealDelay={0.5}
          revealText="True"
        />
      </h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <label className="block text-lg font-medium">Enter Twitter Username:</label>
        <input
          type="text"
          value={username}
          onChange={handleUsernameChange}
          placeholder="e.g. jon doe"
          required
          className="w-full max-w-md mx-auto p-3 rounded bg-white bg-opacity-20 text-black text-center"
        />
        
        <label className="block text-lg font-medium">Enter API Key:</label>
        <input
          type="text"
          value={apiKey}
          onChange={handleApiKeyChange}
          placeholder="Enter your API key"
          className="w-full max-w-md mx-auto p-3 rounded bg-white bg-opacity-20 text-black text-center"
        />

        <button
          type="submit"
          disabled={loading}
          className="px-6 py-3 bg-blue-500 text-white font-semibold rounded-full shadow-lg hover:bg-blue-600 transition duration-300"
        >
          {loading ? "Analyzing..." : "Check Bot Status"}
        </button>
      </form>

      {error && <p className="text-red-500 mt-4">{error}</p>}
      {prediction !== null && (
        <div className="mt-8">
          <h2 className="text-2xl font-semibold">Prediction Result</h2>
          <p className="text-xl">{botProbability > humanProbability ? `${botProbability}% Bot` : `${humanProbability}% Human`}</p>
        </div>
      )}

      {tweetData.length > 0 && (
        <div className="mt-8">
          <h2 className="text-2xl font-semibold">Tweet Activity (Last 10 Days)</h2>
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