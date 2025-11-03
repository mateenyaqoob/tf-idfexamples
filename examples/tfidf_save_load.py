from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load


texts = ["TF-IDF saves models", "Reuse vectorizers efficiently"]


vec = TfidfVectorizer()
X = vec.fit_transform(texts)


# Save vectorizer
dump(vec, 'vectorizer.joblib')


# Load it back
vec_loaded = load('vectorizer.joblib')
X2 = vec_loaded.transform(["Save and load TF-IDF models"])


print("Transformed vector shape:", X2.shape)
