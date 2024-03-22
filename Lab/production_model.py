import pandas as pd
import joblib

# Load the test samples
test_samples = pd.read_csv('test_samples.csv')

# Load the model from disk
knn = joblib.load('knn_model.pkl')

# Make predictions on the test samples
predictions = knn.predict(test_samples)

# Get probabilities for each class
probabilities = knn.predict_proba(test_samples)

# Create a DataFrame with the probabilities and predictions
results = pd.DataFrame({
    'probability class 0': probabilities[:, 0],
    'probability class 1': probabilities[:, 1],
    'prediction': predictions
})

# Export the results to a CSV file
results.to_csv('prediction.csv', index=False)