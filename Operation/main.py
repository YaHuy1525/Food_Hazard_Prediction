from separate_predictions import HazardProductClassifier
from Operation.data_preprocessing import load_and_preprocess_data, prepare_submission_data, save_predictions

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = HazardProductClassifier(model_name="bert-base-uncased")
    
    # Train models
    print("Training models...")
    classifier.train(
        data['train_texts'],
        data['train_hazard_labels'],
        data['train_product_labels'],
        data['val_texts'],
        data['val_hazard_labels'],
        data['val_product_labels'],
        batch_size=8,
        epochs=3
    )
    
    # Prepare submission data
    print("Preparing submission data...")
    submission_texts = prepare_submission_data()
    
    # Make predictions
    print("Making predictions...")
    hazard_predictions, product_predictions = classifier.predict(submission_texts)
    
    # Save predictions
    print("Saving predictions...")
    save_predictions(
        hazard_predictions,
        product_predictions,
        data['hazard_encoder'],
        data['product_encoder']
    )
    
    print("Done!")

if __name__ == "__main__":
    main() 