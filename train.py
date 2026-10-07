import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading dataset...")
df = pd.read_csv('IMDB Dataset.csv')

# Use 20,000 samples for fast training
df = df.sample(20000, random_state=42)

X = df['review']
y = df['sentiment']

print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vec = vectorizer.fit_transform(X)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

print("Saving model and vectorizer...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Done! Files saved successfully.")