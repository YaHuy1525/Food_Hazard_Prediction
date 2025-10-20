import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import re
import string
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Define text preprocessing functions
def clean_text(text):
    """
    Basic text cleaning for all formats
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove newlines
    text = text.replace('\n', ' ')
    # Normalize white space
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove PRA No. references
    text = re.sub(r'pra no\. \d+/\d+', '', text)
    
    return text

def detect_format(text):
    """
    Detect the format of the text (recall case or product alert)
    """
    text = text.lower()
    if "case number" in text or "recall class" in text:
        return "recall_case"
    elif "product description" in text or "what are the hazards" in text:
        return "product_alert"
    else:
        return "unknown"

def extract_recall_case_info(text):
    """
    Extract structured information from recall case format
    """
    info = {}
    patterns = {
        'case_number': r'case number\s*(\S+)',
        'date_opened': r'date opened\s*(\S+)',
        'date_closed': r'date closed\s*(\S+)',
        'recall_class': r'recall class\s*(\S+)',
        'product': r'product\s*:?\s*(.*?)(?=\s*problem|\s*$)',
        'problem': r'problem\s*:?\s*(.*?)(?=\s*description|\s*$)',
        'description': r'description\s*:?\s*(.*?)(?=\s*total pounds|\s*$)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info[key] = match.group(1).strip()
    
    return info

def extract_product_alert_info(text):
    """
    Extract structured information from product alert format
    """
    info = {}
    patterns = {
        'product_description': r'product description\s*(.*?)(?=\s*what are the defects|\s*$)',
        'defects': r'what are the defects\?\s*(.*?)(?=\s*what are the hazards|\s*$)',
        'hazards': r'what are the hazards\?\s*(.*?)(?=\s*what should consumers do|\s*$)',
        'product_category': r'product category\s*(.*?)(?=\s*$)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info[key] = match.group(1).strip()
    
    return info

def extract_features(text):
    """
    Extract features from text based on format detection
    """
    cleaned_text = clean_text(text)
    format_type = detect_format(cleaned_text)
    
    features = {
        'format_type': format_type,
        'text_length': len(cleaned_text),
        'has_bacteria': 1 if 'bacteria' in cleaned_text else 0,
        'has_listeria': 1 if 'listeria' in cleaned_text else 0,
        'has_contamination': 1 if 'contamination' in cleaned_text else 0,
        'has_allergen': 1 if 'allergen' in cleaned_text else 0,
        'has_foreign': 1 if 'foreign' in cleaned_text else 0,
        'has_chemical': 1 if 'chemical' in cleaned_text else 0,
    }
    
    if format_type == "recall_case":
        info = extract_recall_case_info(cleaned_text)
        features.update({
            'product': info.get('product', ''),
            'problem': info.get('problem', ''),
            'description': info.get('description', ''),
        })
    elif format_type == "product_alert":
        info = extract_product_alert_info(cleaned_text)
        features.update({
            'product_description': info.get('product_description', ''),
            'defects': info.get('defects', ''),
            'hazards': info.get('hazards', ''),
            'product_category': info.get('product_category', ''),
        })
    
    return features

# Define multi-task model
class MultiTaskBertModel(nn.Module):
    def __init__(self, hazard_classes, product_classes):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        self.hazard_classifier = nn.Linear(512, hazard_classes)
        self.product_classifier = nn.Linear(512, product_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Extract shared features
        features = self.feature_extractor(pooled_output)
        
        # Task-specific predictions
        hazard_logits = self.hazard_classifier(features)
        product_logits = self.product_classifier(features)
        
        return hazard_logits, product_logits

# Dataset class
class FoodHazardDataset(Dataset):
    def __init__(self, texts, hazard_labels, product_labels, tokenizer, max_length=512):
        self.texts = texts
        self.hazard_labels = hazard_labels
        self.product_labels = product_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract additional features
        self.features = [extract_features(text) for text in texts]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        hazard_label = self.hazard_labels[idx]
        product_label = self.product_labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get additional features
        features = self.features[idx]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'hazard_label': torch.tensor(hazard_label, dtype=torch.long),
            'product_label': torch.tensor(product_label, dtype=torch.long),
            'features': features  # Not used in current model but available for future extensions
        }

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Define loss functions
    hazard_loss_fn = nn.CrossEntropyLoss()
    product_loss_fn = nn.CrossEntropyLoss()
    
    # Track metrics
    train_losses = []
    val_losses = []
    hazard_accuracies = []
    product_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        hazard_correct = 0
        product_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hazard_labels = batch['hazard_label'].to(device)
            product_labels = batch['product_label'].to(device)
            
            # Forward pass
            hazard_logits, product_logits = model(input_ids, attention_mask)
            
            # Calculate losses
            hazard_loss = hazard_loss_fn(hazard_logits, hazard_labels)
            product_loss = product_loss_fn(product_logits, product_labels)
            
            # Combined loss (can be weighted if needed)
            loss = hazard_loss + product_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            hazard_preds = torch.argmax(hazard_logits, dim=1)
            product_preds = torch.argmax(product_logits, dim=1)
            
            hazard_correct += (hazard_preds == hazard_labels).sum().item()
            product_correct += (product_preds == product_labels).sum().item()
            total_samples += hazard_labels.size(0)
        
        # Calculate training metrics
        avg_train_loss = total_loss / len(train_loader)
        hazard_accuracy = hazard_correct / total_samples
        product_accuracy = product_correct / total_samples
        
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        hazard_correct = 0
        product_correct = 0
        total_samples = 0
        
        hazard_preds_all = []
        hazard_labels_all = []
        product_preds_all = []
        product_labels_all = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                hazard_labels = batch['hazard_label'].to(device)
                product_labels = batch['product_label'].to(device)
                
                # Forward pass
                hazard_logits, product_logits = model(input_ids, attention_mask)
                
                # Calculate losses
                hazard_loss = hazard_loss_fn(hazard_logits, hazard_labels)
                product_loss = product_loss_fn(product_logits, product_labels)
                
                # Combined loss
                loss = hazard_loss + product_loss
                
                # Track metrics
                val_loss += loss.item()
                
                # Calculate accuracy
                hazard_preds = torch.argmax(hazard_logits, dim=1)
                product_preds = torch.argmax(product_logits, dim=1)
                
                hazard_correct += (hazard_preds == hazard_labels).sum().item()
                product_correct += (product_preds == product_labels).sum().item()
                total_samples += hazard_labels.size(0)
                
                # Collect predictions for classification report
                hazard_preds_all.extend(hazard_preds.cpu().numpy())
                hazard_labels_all.extend(hazard_labels.cpu().numpy())
                product_preds_all.extend(product_preds.cpu().numpy())
                product_labels_all.extend(product_labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        hazard_accuracy = hazard_correct / total_samples
        product_accuracy = product_correct / total_samples
        
        val_losses.append(avg_val_loss)
        hazard_accuracies.append(hazard_accuracy)
        product_accuracies.append(product_accuracy)
        
        # Print epoch results
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Hazard Accuracy: {hazard_accuracy:.4f}")
        print(f"Product Accuracy: {product_accuracy:.4f}")
        
        print("\nHazard Classification Report:")
        print(classification_report(hazard_labels_all, hazard_preds_all))
        
        print("\nProduct Classification Report:")
        print(classification_report(product_labels_all, product_preds_all))
    
    # Return training history
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'hazard_accuracies': hazard_accuracies,
        'product_accuracies': product_accuracies
    }

def predict_example(model, tokenizer, text, hazard_encoder, product_encoder):
    """
    Predict hazard and product categories for a single text example
    """
    # Clean and preprocess text
    cleaned_text = clean_text(text)
    
    # Extract features for analysis
    features = extract_features(cleaned_text)
    
    # Tokenize
    encoding = tokenizer(
        cleaned_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        hazard_logits, product_logits = model(input_ids, attention_mask)
        
        hazard_probs = torch.softmax(hazard_logits, dim=1)
        product_probs = torch.softmax(product_logits, dim=1)
        
        hazard_pred = torch.argmax(hazard_probs, dim=1).item()
        product_pred = torch.argmax(product_probs, dim=1).item()
    
    # Convert to category names
    hazard_category = hazard_encoder.inverse_transform([hazard_pred])[0]
    product_category = product_encoder.inverse_transform([product_pred])[0]
    
    # Get confidence scores
    hazard_confidence = hazard_probs[0, hazard_pred].item()
    product_confidence = product_probs[0, product_pred].item()
    
    # Return predictions with confidence and extracted features
    return {
        'hazard_category': hazard_category,
        'hazard_confidence': hazard_confidence,
        'product_category': product_category,
        'product_confidence': product_confidence,
        'extracted_features': features
    }

def main():
    # Load data
    print("\nLoading and preprocessing data...")
    df = pd.read_csv('datas/incidents_labelled.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Extract texts and labels
    texts = df['text'].values
    hazard_labels = df['hazard-category'].values
    product_labels = df['product-category'].values
    
    # Encode labels
    hazard_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    
    hazard_labels_encoded = hazard_encoder.fit_transform(hazard_labels)
    product_labels_encoded = product_encoder.fit_transform(product_labels)
    
    print(f"\nNumber of hazard categories: {len(hazard_encoder.classes_)}")
    print(f"Number of product categories: {len(product_encoder.classes_)}")
    
    # Split data
    train_texts, val_texts, train_hazard_labels, val_hazard_labels, train_product_labels, val_product_labels = train_test_split(
        texts, hazard_labels_encoded, product_labels_encoded,
        test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = FoodHazardDataset(
        texts=train_texts,
        hazard_labels=train_hazard_labels,
        product_labels=train_product_labels,
        tokenizer=tokenizer
    )
    
    val_dataset = FoodHazardDataset(
        texts=val_texts,
        hazard_labels=val_hazard_labels,
        product_labels=val_product_labels,
        tokenizer=tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Initialize model
    model = MultiTaskBertModel(
        hazard_classes=len(hazard_encoder.classes_),
        product_classes=len(product_encoder.classes_)
    ).to(device)
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, train_loader, val_loader, epochs=3)
    
    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'multi_task_food_hazard_model.pt')
    tokenizer.save_pretrained('multi_task_food_hazard_tokenizer')
    
    # Save encoders
    with open('hazard_encoder.pkl', 'wb') as f:
        pickle.dump(hazard_encoder, f)
    
    with open('product_encoder.pkl', 'wb') as f:
        pickle.dump(product_encoder, f)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['hazard_accuracies'], label='Hazard Accuracy')
    plt.title('Hazard Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['product_accuracies'], label='Product Accuracy')
    plt.title('Product Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history plot saved to training_history.png")
    
    # Test with example texts
    print("\nTesting with example texts...")
    
    example1 = """
    Case Number: 024-94   
    Date Opened: 07/01/1994   
    Date Closed: 09/22/1994 
    Recall Class:  1   
    Press Release (Y/N):  Y  
    Domestic Est. Number:  05893  P   
    Name:  GERHARD'S NAPA VALLEY SAUSAGE
    Imported Product (Y/N):  N       
    Foreign Estab. Number:  N/A
    City:  NAPA    
    State:  CA   
    Country:  USA
    Product:  SMOKED CHICKEN SAUSAGE
    Problem:  BACTERIA   
    Description: LISTERIA
    Total Pounds Recalled:  2,894   
    Pounds Recovered:  2,894
    """
    
    example2 = """
    PRA No. 1998/3436 Date published 14 Jan 1998 Product description Brands: Gibbs 400g, Sumners 350g & Foodlands 350g. Code Between 8002 And 8012. What are the defects? Rancid Taste In Icing. What are the hazards? Contamination. What should consumers do? Contact The Supplier Undertaking The Recall Action. Supplier Balfour Wauchope Pty Ltd Coordinating agency Food Standards Australia New Zealand is the coordinating agency for this recall. Product category Confectionery Ã— Close
    """
    
    result1 = predict_example(model, tokenizer, example1, hazard_encoder, product_encoder)
    result2 = predict_example(model, tokenizer, example2, hazard_encoder, product_encoder)
    
    print("\nExample 1 Results:")
    print(f"Hazard Category: {result1['hazard_category']} (Confidence: {result1['hazard_confidence']:.4f})")
    print(f"Product Category: {result1['product_category']} (Confidence: {result1['product_confidence']:.4f})")
    print("Extracted Features:", result1['extracted_features'])
    
    print("\nExample 2 Results:")
    print(f"Hazard Category: {result2['hazard_category']} (Confidence: {result2['hazard_confidence']:.4f})")
    print(f"Product Category: {result2['product_category']} (Confidence: {result2['product_confidence']:.4f})")
    print("Extracted Features:", result2['extracted_features'])

if __name__ == "__main__":
    import pickle
    main()
