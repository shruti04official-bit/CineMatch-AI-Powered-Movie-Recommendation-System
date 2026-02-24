# 🎬 CineMatch – AI-Powered Hybrid Movie Recommendation System

CineMatch is a machine learning–based hybrid movie recommendation system that combines content-based filtering and collaborative filtering to generate personalized movie suggestions. The application is built using Python and deployed using Streamlit Cloud.

🔗 **Live Application:**  
https://cinematch-ai04.streamlit.app/

---

## 📌 Project Overview

The objective of this project is to design and deploy an intelligent movie recommendation system capable of suggesting relevant movies based on both movie metadata and user interaction patterns.

The system integrates:
- Content similarity using TF-IDF and cosine similarity
- Collaborative filtering using user rating data
- A hybrid strategy to improve recommendation accuracy

---

## 🚀 Key Features

- Hybrid recommendation engine (Content + Collaborative)
- TF-IDF vectorization for feature extraction
- Cosine similarity for movie matching
- Personalized suggestions based on user ID
- Interactive UI built with Streamlit
- Cloud deployment for public access

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📂 Project Structure

```
CineMatch-AI-Powered-Movie-Recommendation-System/
│
├── app.py
├── movies.csv
├── ratings.csv
├── train_svd.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Local Setup

Clone the repository:

```bash
git clone https://github.com/shruti04official-bit/CineMatch-AI-Powered-Movie-Recommendation-System.git
cd CineMatch-AI-Powered-Movie-Recommendation-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🌐 Deployment

The application is deployed on Streamlit Community Cloud by:

1. Pushing the project to GitHub
2. Connecting the repository to Streamlit Cloud
3. Configuring `app.py` as the main entry point
4. Managing dependencies via `requirements.txt`
5. Deploying and rebooting the app

---

## 📊 Future Improvements

- Integration with TMDB API for movie posters
- Advanced hybrid weighting techniques
- Real-time personalization improvements
- UI enhancement with grid layout and posters
- Model evaluation metrics and performance analysis

---

## 👩‍💻 Author

Shruti Kumari 
Machine Learning Enthusiast | AI Developer  
GitHub: https://github.com/shruti04official-bit
