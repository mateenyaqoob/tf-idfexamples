from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


corpus = [
"Machine learning enables computers to learn",
"Deep learning is part of machine learning",
"Natural language processing is fun"
]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)


sim_matrix = cosine_similarity(X, X)
print("\nCosine Similarity Matrix:\n", sim_matrix)
