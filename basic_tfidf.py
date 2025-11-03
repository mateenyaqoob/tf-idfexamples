from sklearn.feature_extraction.text import TfidfVectorizer


documents = [
"Data science is fun",
"Python makes data analysis easy",
"TF-IDF is a feature extraction method"
]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)


print("Feature Names:", vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:\n", X.toarray())
