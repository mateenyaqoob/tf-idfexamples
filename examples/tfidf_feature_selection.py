import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


texts = [
"Artificial intelligence and machine learning",
"Deep learning with neural networks",
"AI applications in healthcare"
]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)


# Display top features by average TF-IDF weight
avg_tfidf = X.mean(axis=0).A1
terms = vectorizer.get_feature_names_out()
ranking = sorted(zip(terms, avg_tfidf), key=lambda x: x[1], reverse=True)


print("Top TF-IDF Terms:")
for term, score in ranking[:10]:
print(f"{term:20s}: {score:.4f}")
