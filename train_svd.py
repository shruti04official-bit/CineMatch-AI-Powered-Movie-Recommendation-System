import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD

# Load MovieLens ratings (download ml-latest-small)
ratings = pd.read_csv("ratings.csv")

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)

# Save trained model
joblib.dump(model, "svd_model.pkl")

print("SVD model trained and saved successfully.")

