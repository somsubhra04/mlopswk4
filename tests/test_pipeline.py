
import pandas as pd
import joblib
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define paths
DATA_PATH = "data/v1.csv"
MODEL_PATH = "model.joblib"

# --- Data Validation Test ---
def test_data_validation():
    """Checks for expected columns and data types specific to the Iris dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        pytest.fail(f"Data file not found at {DATA_PATH}. Run 'dvc pull'.")

    # Check required columns
    required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

    # Check data types for features (should be numeric)
    feature_columns = required_columns[:-1]
    for col in feature_columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Feature column {col} is not numeric."

    # Check target column unique values
    species_values = df['species'].unique()
    expected_species = {'setosa', 'versicolor', 'virginica'}
    assert set(species_values).issubset(expected_species), "Target column contains unexpected values."
    
    # Check minimum size
    assert len(df) >= 100, "Dataset is too small."

# --- Model Evaluation Sanity Test ---
def test_model_evaluation_sanity():
    """Checks if the model can achieve a reasonable accuracy on a test set."""
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        pytest.fail("Model or data file not found. Run 'dvc pull'.")
        
    # Recreate the train/test split from train.py to ensure consistency
    train, test = train_test_split(
        df,
        test_size=0.4,
        stratify=df['species'],
        random_state=42
    )

    # Use the test set from the split
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X_test = test[feature_cols]
    y_test = test['species']

    # Make predictions
    y_pred = model.predict(X_test)

    # Sanity check: Iris is highly separable, a good model should be > 85%
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.85, f"Model accuracy is too low: {accuracy:.3f}"
