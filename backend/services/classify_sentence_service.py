import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Function to classify a sentence
def classify_sentence_bert(input_sentence):
    try:
        device = torch.device("mps" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'static', 'models', 'transformers', 'bert_model_for_binary_classification.pt')
        model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenize input sentence
        inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=1).item()
            label = ([prediction])[0]

        return "sexist" if label == 1 else "not sexist"
    except Exception as e:
        print(f"Error classifying sentence: {input_sentence}. Error: {e}")
        return "error"


def classify_sentence_naive_bayes(input_sentence):
    print("classify_sentence_naive_bayes" + input_sentence)
    # Load the saved model and vectorizer
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_filename = os.path.join(base_dir, 'static', 'models', 'naive_bayes', 'naive_bayes_model.joblib')
    vectorizer_filename = os.path.join(base_dir, 'static', 'models', 'naive_bayes', 'tfidf_vectorizer.joblib')
    nb_classifier = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)

    sentence_vector = vectorizer.transform([input_sentence])
    # Make predictions using the loaded model
    prediction = nb_classifier.predict(sentence_vector)[0]
    return prediction
