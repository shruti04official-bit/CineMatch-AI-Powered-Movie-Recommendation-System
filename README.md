🎬 CineMatch – AI-Powered Movie Recommendation System
AI-Powered Content-Based Movie Recommendation System
<p align="center"> <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" /> <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit" /> <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn" /> <img src="https://img.shields.io/badge/Deployment-Render-purple?style=for-the-badge" /> </p>
🌐 Live Application

🔗 Try the App Here:
https://cinematch-ai-powered-movie.onrender.com/

📖 Overview

CineMatch is a machine learning–powered movie recommendation system that suggests similar movies based on genre similarity.

The system uses TF-IDF vectorization to transform movie genres into numerical feature representations and applies cosine similarity to identify the most relevant recommendations.

It is built with Python and deployed as an interactive web application using Streamlit on Render.

🎯 Problem Statement

Users often struggle to discover movies aligned with their preferences.
Traditional recommendation systems either rely heavily on user history or require large-scale infrastructure.

This project demonstrates how a lightweight, scalable content-based recommendation engine can deliver accurate suggestions using only movie metadata.

🧠 System Architecture

User Input (Selected Movie)
↓
Genre Extraction
↓
TF-IDF Vectorization
↓
Cosine Similarity Computation
↓
Ranked Similar Movies
↓
Top-N Recommendations Displayed

⚙️ How It Works
1️⃣ Data Processing

Loads movies.csv and ratings.csv

Removes duplicate titles

Handles missing values

Optimizes dataset size for cloud deployment

2️⃣ Feature Engineering

Extracts genre information

Applies TF-IDF transformation

Builds a feature matrix for similarity comparison

3️⃣ Similarity Computation

Computes cosine similarity dynamically

Ranks movies by similarity score

Returns configurable Top-N results

4️⃣ Optimization Techniques

@st.cache_data for dataset caching

@st.cache_resource for model caching

Reduced memory footprint for free-tier hosting

🚀 Key Features

✔ Content-Based Filtering
✔ Real-Time Similarity Computation
✔ User-Based Filtering (excludes watched movies)
✔ Cold-Start Handling
✔ Interactive Sidebar Controls
✔ Cloud Deployment Ready
✔ Clean Dark-Theme UI

🛠 Tech Stack
Layer	Technology
Programming Language	Python 3
Frontend Framework	Streamlit
Data Handling	Pandas
Machine Learning	Scikit-Learn
Vectorization	TF-IDF
Similarity Metric	Cosine Similarity
Deployment	Render
📂 Project Structure
CineMatch-AI-Powered-Movie-Recommendation-System/
│
├── app.py                # Main Streamlit application
├── movies.csv            # Movie dataset
├── ratings.csv           # User ratings dataset
├── requirements.txt      # Project dependencies
└── README.md
💻 Local Setup
Clone Repository
git clone https://github.com/shruti04official-bit/CineMatch-AI-Powered-Movie-Recommendation-System.git
cd CineMatch-AI-Powered-Movie-Recommendation-System
Install Dependencies
pip install -r requirements.txt
Run Application
streamlit run app.py
🌍 Deployment Configuration (Render)

Build Command

pip install -r requirements.txt

Start Command

streamlit run app.py --server.port 10000 --server.address 0.0.0.0
📈 Future Enhancements

Integration with TMDB API for movie posters

Advanced hybrid weighting strategy

Recommendation evaluation metrics (Precision@K, Recall@K)

Database-backed user personalization

Improved UI with movie card grid layout

👩‍💻 Author





