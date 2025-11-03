import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Sample dataset
data = pd.DataFrame({
'text': [
'I love this movie',
'This film was terrible',
'Amazing acting and story',
'Worst movie ever',
'Highly recommend this film'
],
'label': [1, 0, 1, 0, 1]
})


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


preds = model.predict(X_test)
print(classification_report(y_test, preds))
