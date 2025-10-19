import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def main(input_csv, model_out):
    """
    Loads data, trains a Logistic Regression model, and saves it.
    """
    try:
        data = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: The input file '{input_csv}' was not found.")
        sys.exit(1)

    print(f"Iris data loaded successfully with shape: {data.shape}")

    # data split for training, testing
    train, test = train_test_split(
        data,
        test_size=0.4,
        stratify=data['species'],
        random_state=42
    )

    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train['species']
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']

    # Initializing and training the Logistic Regression model
    mod_lr = LogisticRegression(solver='liblinear', random_state=1)
    mod_lr.fit(X_train, y_train)

    # Evaluating the model
    prediction = mod_lr.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    print(f"Logistic Regression accuracy: {accuracy:.3f}")

    # Saving the trained model to the specified path
    joblib.dump(mod_lr, model_out)
    print(f"Model saved to {model_out}")

if __name__ == "__main__":
    # Checking if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python train.py <input_csv> <model_out>")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    output_model_path = sys.argv[2]
    main(input_file_path, output_model_path)
