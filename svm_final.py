import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import joblib

def train_and_evaluate():
    # Load the data
    data = pd.read_csv('new data.csv')  # Ensure the file path is correct

    # Prepare the data
    X = data.drop(['gene_id', 'Gene_type'], axis=1)  # Features
    y = data['Gene_type'].astype(int)  # Target variable converted to integer

    # Initialize and save the scaler
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm_model = SVC(kernel='rbf', probability=True,C=1.0,gamma='auto')

    # Results storage
    probabilities_list = []
    all_y_test = []
    all_probs = []

    # Cross-validation
    for train_index, test_index in cv.split(X_scaled_full, y):
        X_train, X_test = X_scaled_full[train_index], X_scaled_full[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Model training
        svm_model.fit(X_train, y_train)
        probs = svm_model.predict_proba(X_test)[:, 1]
        all_probs.extend(probs)
        all_y_test.extend(y_test)

        # Storing results
        fold_results = pd.DataFrame({
            'Gene_ID': data.iloc[test_index]['gene_id'],
            'Actual_Label': y_test,
            'Probability_ASD_Related': probs
        })
        probabilities_list.append(fold_results)

    # Results concatenation and saving
    all_results_df = pd.concat(probabilities_list)
    all_results_df.to_csv('cross_validation_probabilities2.csv', index=False)
    print("Probabilities saved to 'cross_validation_probabilities2.csv'.")

    # Overall evaluation
    accuracy = accuracy_score(all_y_test, [1 if x > 0.5 else 0 for x in all_probs])
    fpr, tpr, _ = roc_curve(all_y_test, all_probs)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc)
    print(f"Overall Accuracy: {accuracy:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")

    # Full model training and saving
    svm_model.fit(X_scaled_full, y)
    joblib.dump(svm_model, 'svm_model_full1.pkl')
    print("Full model saved as 'svm_model_full1.pkl'.")

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
