import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adamax, Nadam
#from tensorflow.keras.optimizers import Adamax,SDG
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load and preprocess the data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop(['Gene_type'], axis=1)
    y = data['Gene_type'].astype(int)  # Ensure y is binary (0 or 1)
    gene_ids = X.pop('gene_id')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
    return X_scaled, y, gene_ids

# Build the neural network model
def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),  # Explicit input layer defining input shape
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    opti=Adam()
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main execution function
def main():
    X_scaled, y, gene_ids = load_and_preprocess_data('new data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = build_model(X_train.shape[1])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=70, batch_size=32, verbose=1, callbacks=[early_stopping])
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    all_prob_predictions = model.predict(X_scaled).flatten()
    roc_auc = roc_auc_score(y, all_prob_predictions)
    print(f"ROC AUC: {roc_auc:.2f}")

    model.save('ann_model1.keras')  # Saving in the new recommended format
    #print("Model saved in '.keras' format.")

    full_results_df = pd.DataFrame({
        'Gene_ID': gene_ids,
        'Actual_Label': y,
        'Predicted_Probability': all_prob_predictions
    })
    full_results_df.to_csv('full_dataset_predictions.csv', index=False)
    print("Predicted probabilities saved to 'full_dataset_predictions.csv'.")

if __name__ == "__main__":
    main()
