Content-Based Movie Recommendation System
1. Introduction

This project implements a Content-Based Movie Recommendation System using Python and Streamlit.
The system recommends movies based on genre similarity by applying TF-IDF vectorization and cosine similarity.

The application is deployed on Render for public access.

Live Application:
https://cinematch-ai-powered-movie.onrender.com/

2. Objective

The objective of this project is to:

Build a scalable content-based recommendation engine

Convert textual genre data into numerical feature vectors

Compute similarity between movies

Deploy the system on a cloud platform

3. Methodology
3.1 Data Processing

Movies and ratings datasets are loaded using Pandas

Duplicate titles are removed

Missing values are handled

Dataset size is optimized for deployment performance

3.2 Feature Engineering

Movie genres are extracted

TF-IDF vectorization is applied to transform text into numerical vectors

3.3 Similarity Computation

Cosine similarity is calculated between movie vectors

Top-N similar movies are retrieved dynamically

3.4 Performance Optimization

Dataset caching using @st.cache_data

TF-IDF matrix caching using @st.cache_resource

Reduced dataset size for cloud hosting efficiency

4. System Architecture

Input (Selected Movie)
↓
Genre Extraction
↓
TF-IDF Vectorization
↓
Cosine Similarity Computation
↓
Top-N Similar Movies
↓
Recommendation Display (Streamlit UI)

5. Features

Interactive web interface using Streamlit

Movie selection from dropdown

Configurable number of recommendations

Efficient similarity computation

Cloud deployment ready

6. Technology Stack

Python 3

Streamlit

Pandas

Scikit-Learn (TF-IDF, Cosine Similarity)

Render (Deployment Platform)

7. Project Structure
Movie-Recommendation-System/
│
├── app.py
├── movies.csv
├── ratings.csv
├── requirements.txt
└── README.md
8. Installation and Setup
Step 1: Clone Repository
git clone https://github.com/shruti04official-bit/CineMatch-AI-Powered-Movie-Recommendation-System.git
cd CineMatch-AI-Powered-Movie-Recommendation-System
Step 2: Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate
Step 3: Install Dependencies
pip install -r requirements.txt
Step 4: Run Application
streamlit run app.py
9. Deployment

The application is deployed on Render.

Build Command:

pip install -r requirements.txt

Start Command:

streamlit run app.py --server.port 10000 --server.address 0.0.0.0
10. Future Improvements

Integration of collaborative filtering

Addition of movie posters via external APIs

Improved ranking strategy

Recommendation evaluation metrics

Database-backed user tracking

11. Author

Shruti Kumari
Computer Science Undergraduate
Machine Learning Enthusiast
