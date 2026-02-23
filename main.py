import pandas as pd

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
# Remove duplicates
print("Before removing duplicates:", df.shape)

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

print(df.head())

print("\nShape of dataset:", df.shape)

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values first (if not already done)
df["Director"] = df["Director"].fillna("Unknown")
df["Actor 1"] = df["Actor 1"].fillna("Unknown")
df["Actor 2"] = df["Actor 2"].fillna("Unknown")
df["Actor 3"] = df["Actor 3"].fillna("Unknown")
df["Genre"] = df["Genre"].fillna("Unknown")


# Combine important features
df["tags"] = (
    df["Genre"].astype(str) + " " +
    df["Director"] + " " +
    df["Actor 1"] + " " +
    df["Actor 2"] + " " +
    df["Actor 3"]
)

print(df[["Name", "tags"]].head())
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

# Convert tags into vectors
vectors = cv.fit_transform(df["tags"]).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

print("Similarity matrix shape:", similarity.shape)
def recommend(movie):
    movie_index = df[df["Name"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(df.iloc[i[0]].Name)
recommend("#Yaaram")

# Remove rows where Name is empty
df = df[df["Name"].str.strip() != ""]

print("After removing empty names:", df.shape)

# Columns important for recommendation
important_cols = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

# Fill missing values with "Unknown"
for col in important_cols:
    df[col] = df[col].fillna("Unknown")
# Function to clean text
def clean_text(text):
    return text.lower().replace(",", "").strip()

# Apply cleaning to important columns
for col in important_cols:
    df[col] = df[col].apply(clean_text)
# Clean movie names for better search
df["Name"] = df["Name"].str.lower().str.strip()

# Combine Genre, Director, Actors into single tags column
df["tags"] = (
    df["Genre"] + " " +
    df["Director"] + " " +
    df["Actor 1"] + " " +
    df["Actor 2"] + " " +
    df["Actor 3"]
)

# Remove extra spaces
df["tags"] = df["tags"].str.replace("  ", " ")
print("Sample cleaned data:")
print(df[["Name", "tags"]].head())

print("Shape of dataset:", df.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Initialize vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Fit and transform tags column
tfidf_matrix = tfidf.fit_transform(df["tags"])

print("TF-IDF matrix shape:", tfidf_matrix.shape)
# Compute similarity
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Similarity matrix shape:", similarity.shape)
def recommend(movie_name):
    movie_name = movie_name.lower().strip()
    if movie_name not in df["Name"].values:
        print("Movie not found!")
        return
    idx = df[df["Name"] == movie_name].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 similar movies
    recommended = [df.iloc[i[0]]["Name"] for i in sim_scores]
    from rapidfuzz import process

def recommend(movie_name):
    movie_name_clean = movie_name.lower().strip()
    
    # Find the closest match in dataset
    match = process.extractOne(movie_name_clean, df["Name"].tolist(), score_cutoff=70)
    
    if not match:
        print("Movie not found!")
        return
    
    matched_name = match[0]
    idx = df[df["Name"] == matched_name].index[0]
    
    # Compute similarity scores
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 similar movies
    
    recommended = [df.iloc[i[0]]["Name"] for i in sim_scores]
    
    print(f"Your movie: {matched_name.title()}")
    print("Recommended movies:")
    for rec in recommended:
        print(rec.title())
