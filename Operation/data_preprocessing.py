import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def removeTitleNumber(df):
    df['title'] = df['title'].str.lower().str.replace('[^a-z\s]', '', regex=True)

def remove_pra_no(text):
    pattern = r'PRA No\. \d+/\d+'
    return re.sub(pattern, '', text)

def load_and_preprocess_data():
    # Load the labelled incidents data
    df = pd.read_csv('datas/incidents_labelled.csv')
    
    # Assuming the data has columns: 'text', 'hazard_category', 'product_category'
    # If the column names are different, adjust accordingly
    texts = df['text'].values
    hazard_labels = df['hazard_category'].values
    product_labels = df['product_category'].values
    df['text'] = df['text'].apply(remove_pra_no)
    
    # Encode labels
    hazard_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    
    hazard_labels_encoded = hazard_encoder.fit_transform(hazard_labels)
    product_labels_encoded = product_encoder.fit_transform(product_labels)
    
    # Split data into train and validation sets
    train_texts, val_texts, train_hazard_labels, val_hazard_labels, train_product_labels, val_product_labels = train_test_split(
        texts, hazard_labels_encoded, product_labels_encoded,
        test_size=0.2, random_state=42
    )
    
    return {
        'train_texts': train_texts,
        'val_texts': val_texts,
        'train_hazard_labels': train_hazard_labels,
        'val_hazard_labels': val_hazard_labels,
        'train_product_labels': train_product_labels,
        'val_product_labels': val_product_labels,
        'hazard_encoder': hazard_encoder,
        'product_encoder': product_encoder
    }

def prepare_submission_data():
    # Load the submission data
    df = pd.read_csv('submission.csv')
    texts = df['text'].values
    return texts

def save_predictions(hazard_predictions, product_predictions, hazard_encoder, product_encoder):
    # Decode predictions
    hazard_categories = hazard_encoder.inverse_transform(hazard_predictions)
    product_categories = product_encoder.inverse_transform(product_predictions)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'text': prepare_submission_data(),
        'hazard-category': hazard_categories,
        'product-category': product_categories
    })
    
    # Save to file
    submission_df.to_csv('separated_predictions.csv', index=False)
    print("Predictions saved to separated_predictions.csv") 