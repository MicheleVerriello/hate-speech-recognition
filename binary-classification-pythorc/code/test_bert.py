import torch
from transformers import BertTokenizer, BertForSequenceClassification
from dataset_preprocessing import get_dataset
from generating_results import generate_results
import concurrent.futures
import pandas as pd


# Function to classify a sentence
def classify_sentence(input_sentence, model, tokenizer, device):
    try:
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
        return None


if __name__ == '__main__':
    # device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    #
    # # Load model and tokenizer
    # model_path = '../models/transformers/bert_model_for_binary_classification.pt'
    # model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)
    # model.eval()
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # dataset = get_dataset('test')
    # dataset['result_bert'] = None
    #
    # error_sentences = []
    #
    #
    # def process_row(row):
    #     result = classify_sentence(row['sentence'], model, tokenizer, device)
    #     return row.name, result  # Return the index and result
    #
    #
    # # Use ThreadPoolExecutor to parallelize the processing of rows
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_row, row) for index, row in dataset.iterrows()]
    #
    #     for future in concurrent.futures.as_completed(futures):
    #         index, result = future.result()
    #         if result is not None:
    #             dataset.loc[index, 'result_bert'] = result
    #             print("good")
    #         else:
    #             error_sentences.append(dataset.loc[index, 'sentence'])
    #
    # print(f"Errors: {error_sentences}")
    #
    # print(dataset.head())
    # dataset.to_csv('../dataset/test_bert_result.csv', index=False)

    dataset = pd.read_csv('../dataset/test_bert_result.csv')

    # Generate results
    generate_results(dataset['label'].tolist(), dataset['result_bert'].tolist(), ['Non-Sexist', 'Sexist'])
