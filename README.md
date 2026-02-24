🎬 Content-Based Movie Recommendation System

A production-ready Content-Based Movie Recommendation System built using Python, Streamlit, and Scikit-Learn.
The system recommends movies by analyzing genre similarity using TF-IDF vectorization and cosine similarity.

The application is deployed on Render for public access.

🔗 Live Demo:
https://cinematch-ai-powered-movie.onrender.com/

📌 Project Overview

This project implements a content-based filtering approach to recommend movies based on their genre similarity.

Instead of relying on other users’ ratings, the system analyzes movie metadata (genres) to identify and suggest similar movies.

The system is optimized for cloud deployment and includes caching mechanisms for performance efficiency.

🧠 How It Works
1️⃣ Data Preprocessing

Loads movie and rating datasets using Pandas

Handles missing values

Removes duplicate titles

Reduces dataset size for efficient deployment

2️⃣ Feature Engineering

Extracts movie genres

Applies TF-IDF Vectorization to convert text data into numerical features

3️⃣ Similarity Computation

Computes Cosine Similarity

Identifies top similar movies dynamically

Returns configurable Top-N recommendations

🚀 Features

Interactive Streamlit interface

Sidebar movie selection

Dynamic similarity computation

Cold-start user handling

Cached dataset & model loading

Optimized for free-tier cloud hosting

🛠️ Tech Stack
Category	Technology
Language	Python 3
Frontend	Streamlit
Data Processing	Pandas
NLP	TF-IDF (Scikit-Learn)
Similarity Metric	Cosine Similarity
Deployment	Render
📂 Project Structure
Content-Based-Movie-Recommender/
│
├── app.py               # Main Streamlit application
├── movies.csv           # Movies dataset
├── ratings.csv          # User ratings dataset
├── requirements.txt     # Project dependencies
└── README.md
⚙️ Installation & Local Setup
1️⃣ Clone Repository
git clone https://github.com/shruti04official-bit/CineMatch-AI-Powered-Movie-Recommendation-System.git
cd CineMatch-AI-Powered-Movie-Recommendation-System
2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Application
streamlit run app.py
🌐 Deployment

The application is deployed on Render using:

Build Command

pip install -r requirements.txt

Start Command

streamlit run app.py --server.port 10000 --server.address 0.0.0.0
📈 Future Enhancements

Add collaborative filtering

Integrate TMDB API for movie posters

Improve ranking logic

Deploy with database-backed user tracking

Add advanced recommendation evaluation metrics

👩‍💻 Author

Shruti
Computer Science Undergraduate
Machine Learning & AI Enthusiast

