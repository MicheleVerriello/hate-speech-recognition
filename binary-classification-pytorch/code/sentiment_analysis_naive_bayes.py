from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from dataset_preprocessing import get_dataset

# Load datasets
df = get_dataset('train')

# Split the data into features and labels
X = df['sentence']
y = df['label']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_tfidf = vectorizer.fit_transform(X)

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_tfidf, y)
print('Training finished...')

# Save the trained model and vectorizer to files
model_filename = '../models/naive_bayes/naive_bayes_model.joblib'
joblib.dump(nb_classifier, model_filename)
vectorizer_filename = '../models/naive_bayes/tfidf_vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_filename)
print(f"Model and vectorizer saved to {model_filename} and {vectorizer_filename}.")
