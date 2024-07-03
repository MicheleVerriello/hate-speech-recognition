import joblib
from dataset_preprocessing import get_dataset
from generating_results import generate_results


# Load the saved model and vectorizer
model_filename = '../models/naive_bayes/naive_bayes_model.joblib'
vectorizer_filename = '../models/naive_bayes/tfidf_vectorizer.joblib'
nb_classifier = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)
print("Model and vectorizer loaded successfully.")

# Sample new data for prediction
dataset = get_dataset('test')
dataset['result_naive_bayes'] = None

for index, row in dataset.iterrows():
    # Transform the new data using the loaded vectorizer
    print(row['sentence'])
    sentence_vector = vectorizer.transform([row['sentence']])
    # Make predictions using the loaded model
    prediction = nb_classifier.predict(sentence_vector)[0]
    print(prediction)
    dataset.loc[index, 'result_naive_bayes'] = prediction
    print('expected: ' + row['label'] + ' predicted: ' + prediction)

generate_results(dataset['label'].tolist(), dataset['result_naive_bayes'].tolist(), ['Non-Sexist', 'Sexist'])
