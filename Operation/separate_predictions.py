import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

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

class HazardProductClassifier:
    def __init__(self, model_name="bert-base-uncased"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize separate models for hazard and product classification
        self.hazard_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_hazard_categories  # You'll need to define this
        ).to(self.device)
        
        self.product_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_product_categories  # You'll need to define this
        ).to(self.device)

    def train(self, train_texts, train_hazard_labels, train_product_labels, 
              val_texts, val_hazard_labels, val_product_labels,
              batch_size=8, epochs=3):
        
        # Create datasets
        train_hazard_dataset = TextDataset(train_texts, train_hazard_labels, self.tokenizer)
        train_product_dataset = TextDataset(train_texts, train_product_labels, self.tokenizer)
        
        val_hazard_dataset = TextDataset(val_texts, val_hazard_labels, self.tokenizer)
        val_product_dataset = TextDataset(val_texts, val_product_labels, self.tokenizer)
        
        # Create dataloaders
        train_hazard_loader = DataLoader(train_hazard_dataset, batch_size=batch_size, shuffle=True)
        train_product_loader = DataLoader(train_product_dataset, batch_size=batch_size, shuffle=True)
        
        val_hazard_loader = DataLoader(val_hazard_dataset, batch_size=batch_size)
        val_product_loader = DataLoader(val_product_dataset, batch_size=batch_size)
        
        # Train hazard model
        print("Training hazard classification model...")
        self._train_model(self.hazard_model, train_hazard_loader, val_hazard_loader, epochs)
        
        # Train product model
        print("Training product classification model...")
        self._train_model(self.product_model, train_product_loader, val_product_loader, epochs)

    def _train_model(self, model, train_loader, val_loader, epochs):
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
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
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Validation
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    preds = torch.argmax(outputs.logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
            print(classification_report(all_labels, all_preds))

    def predict(self, texts):
        self.hazard_model.eval()
        self.product_model.eval()
        
        dataset = TextDataset(texts, [0]*len(texts), self.tokenizer)  # Dummy labels
        dataloader = DataLoader(dataset, batch_size=8)
        
        hazard_predictions = []
        product_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get hazard predictions
                hazard_outputs = self.hazard_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                hazard_preds = torch.argmax(hazard_outputs.logits, dim=1)
                hazard_predictions.extend(hazard_preds.cpu().numpy())
                
                # Get product predictions
                product_outputs = self.product_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                product_preds = torch.argmax(product_outputs.logits, dim=1)
                product_predictions.extend(product_preds.cpu().numpy())
        
        return hazard_predictions, product_predictions

def main():
    # Load your data
    # You'll need to implement this part based on your data structure
    pass

if __name__ == "__main__":
    main() 