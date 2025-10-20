import torch
import pickle
import re
from transformers import AutoTokenizer, BertModel
import torch.nn as nn

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the model class (same as in the main script)
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

# Text preprocessing functions
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

def classify_text(text):
    """
    Classify a single text example using rule-based approach
    """
    # Clean and extract features
    cleaned_text = clean_text(text)
    features = extract_features(cleaned_text)
    
    # Rule-based classification for hazard category
    hazard_category = "unknown"
    if features['has_bacteria'] or features['has_listeria']:
        hazard_category = "biological"
    elif features['has_contamination'] and "rancid" in cleaned_text:
        hazard_category = "chemical"
    elif features['has_contamination'] and "foreign" in cleaned_text:
        hazard_category = "foreign bodies"
    elif features['has_allergen']:
        hazard_category = "allergens"
    elif "packaging" in cleaned_text or "package" in cleaned_text:
        hazard_category = "packaging defect"
    elif "taste" in cleaned_text or "smell" in cleaned_text or "odor" in cleaned_text or "rancid" in cleaned_text:
        hazard_category = "organoleptic aspects"
    
    # Rule-based classification for product category
    product_category = "unknown"
    if "product category" in cleaned_text:
        # Extract directly if mentioned
        match = re.search(r'product category\s*(\w+)', cleaned_text)
        if match:
            product_category = match.group(1)
    else:
        # Infer from content
        if "chicken" in cleaned_text or "meat" in cleaned_text or "sausage" in cleaned_text:
            product_category = "meat, egg and dairy products"
        elif "bread" in cleaned_text or "bakery" in cleaned_text:
            product_category = "cereals and bakery products"
        elif "confectionery" in cleaned_text or "icing" in cleaned_text:
            product_category = "confectionery"
        elif "fish" in cleaned_text or "seafood" in cleaned_text:
            product_category = "seafood"
        elif "fruit" in cleaned_text or "vegetable" in cleaned_text:
            product_category = "fruits and vegetables"
    
    return {
        'hazard_category': hazard_category,
        'product_category': product_category,
        'extracted_features': features
    }

def main():
    print("Food Hazard and Product Category Classifier")
    print("===========================================")
    
    # Example texts
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
    
    # Classify examples
    result1 = classify_text(example1)
    result2 = classify_text(example2)
    
    # Print results
    print("\nExample 1 (Recall Case Format):")
    print("-" * 40)
    print(f"Hazard Category: {result1['hazard_category']}")
    print(f"Product Category: {result1['product_category']}")
    print("\nExtracted Features:")
    for key, value in result1['extracted_features'].items():
        if key not in ['format_type', 'text_length']:
            print(f"  - {key}: {value}")
    
    print("\nExample 2 (Product Alert Format):")
    print("-" * 40)
    print(f"Hazard Category: {result2['hazard_category']}")
    print(f"Product Category: {result2['product_category']}")
    print("\nExtracted Features:")
    for key, value in result2['extracted_features'].items():
        if key not in ['format_type', 'text_length']:
            print(f"  - {key}: {value}")
    
    print("\nAnalysis:")
    print("-" * 40)
    print("The system uses a rule-based approach to classify food hazard incidents.")
    print("For more complex cases, a machine learning model would be more accurate.")
    print("The improved_food_hazard_classifier.py script implements a BERT-based")
    print("multi-task learning model that can be trained on labeled data.")

if __name__ == "__main__":
    main()
