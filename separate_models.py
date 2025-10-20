import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Data Loading and Preprocessing
print("\nLoading and preprocessing data...")
df = pd.read_csv('datas/incidents_labelled.csv')
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

texts = df['text'].values
hazard_labels = df['hazard_category'].values
product_labels = df['product_category'].values

hazard_encoder = LabelEncoder()
product_encoder = LabelEncoder()

hazard_labels_encoded = hazard_encoder.fit_transform(hazard_labels)
product_labels_encoded = product_encoder.fit_transform(product_labels)

print(f"\nNumber of hazard categories: {len(hazard_encoder.classes_)}")
print(f"Number of product categories: {len(product_encoder.classes_)}")

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split data
print("\nSplitting data into train and validation sets...")
train_texts, val_texts, train_hazard_labels, val_hazard_labels, train_product_labels, val_product_labels = train_test_split(
    texts, hazard_labels_encoded, product_labels_encoded,
    test_size=0.2, random_state=42
)

# Initialize tokenizer and models
print("\nInitializing models...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

hazard_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(hazard_encoder.classes_)
).to(device)

product_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(product_encoder.classes_)
).to(device)

def train_model(model, train_loader, val_loader, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print("Classification Report:")
        print(classification_report(all_labels, all_preds))
        
    return train_losses, val_losses

# Create datasets and dataloaders for hazard classification
print("\nCreating datasets for hazard classification...")
train_hazard_dataset = TextDataset(train_texts, train_hazard_labels, tokenizer)
val_hazard_dataset = TextDataset(val_texts, val_hazard_labels, tokenizer)

train_hazard_loader = DataLoader(train_hazard_dataset, batch_size=8, shuffle=True)
val_hazard_loader = DataLoader(val_hazard_dataset, batch_size=8)

# Train hazard model
print("\nTraining hazard classification model...")
hazard_train_losses, hazard_val_losses = train_model(hazard_model, train_hazard_loader, val_hazard_loader)

# Create datasets and dataloaders for product classification
print("\nCreating datasets for product classification...")
train_product_dataset = TextDataset(train_texts, train_product_labels, tokenizer)
val_product_dataset = TextDataset(val_texts, val_product_labels, tokenizer)

train_product_loader = DataLoader(train_product_dataset, batch_size=8, shuffle=True)
val_product_loader = DataLoader(val_product_dataset, batch_size=8)

# Train product model
print("\nTraining product classification model...")
product_train_losses, product_val_losses = train_model(product_model, train_product_loader, val_product_loader)

# Load submission data
print("\nLoading submission data...")
submission_df = pd.read_csv('submission.csv')
submission_texts = submission_df['text'].values

# Create submission dataset
submission_dataset = TextDataset(submission_texts, [0]*len(submission_texts), tokenizer)
submission_loader = DataLoader(submission_dataset, batch_size=8)

# Prediction function
def make_predictions(model, dataloader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return predictions

# Make predictions
print("\nMaking predictions...")
hazard_predictions = make_predictions(hazard_model, submission_loader)
product_predictions = make_predictions(product_model, submission_loader)

# Decode predictions
hazard_categories = hazard_encoder.inverse_transform(hazard_predictions)
product_categories = product_encoder.inverse_transform(product_predictions)

# Create submission dataframe
submission_df['hazard-category'] = hazard_categories
submission_df['product-category'] = product_categories

# Save predictions
submission_df.to_csv('separated_predictions.csv', index=False)
print("\nPredictions saved to separated_predictions.csv")

# Plot training and validation losses
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hazard_train_losses, label='Training Loss')
plt.plot(hazard_val_losses, label='Validation Loss')
plt.title('Hazard Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(product_train_losses, label='Training Loss')
plt.plot(product_val_losses, label='Validation Loss')
plt.title('Product Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_losses.png')
print("\nTraining loss plots saved to training_losses.png") 