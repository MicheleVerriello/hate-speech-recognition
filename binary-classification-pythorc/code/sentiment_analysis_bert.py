import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, logging
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dataset_preprocessing import get_dataset

start_date = datetime.now()
print(f'Process started at {start_date}')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logging.set_verbosity_warning()

df = get_dataset('train')

# Initialize the tokenizer
print('Loading tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the sentences
print('Tokenize the sentences ...')
inputs = tokenizer(df['sentence'].tolist(), padding=True, truncation=True, return_tensors="pt")

# Encode the labels
print('Encode the labels, transforming sexist/non_sexist into 1 and 0 ...')
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
label_list = df['label'].to_numpy()
print(label_list)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

# Ensure labels are integers and convert to a tensor
labels = torch.tensor(label_list, dtype=torch.long)

# Create a DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Check device and use MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
labels.to(device)
inputs.to(device)

# Training loop
epochs = 5
print(f"Training for {epochs} epochs")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    err = 0
    good = 0

    for batch in dataloader:
        input_ids, attention_mask, labels = tuple(b.to(device) for b in batch)

        optimizer.zero_grad()

        try:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]

            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        except BaseException as e:
            print(e)

    avg_val_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Val loss: {avg_val_loss:.4f}")
    print(f'Epoch {epoch + 1}/{epochs} completed. Loss: {loss.item():.4f}')

# Save the model
print('Saving the model into the device ...')
path = '../models/transformers/bert_model_for_binary_classification.pt'
torch.save(model.state_dict(), path)
print(f'model saved into {path}')

end_date = datetime.now()
print(f'Process ended at {end_date}')

total_time = end_date - start_date
print(f'Total time: {total_time}')
