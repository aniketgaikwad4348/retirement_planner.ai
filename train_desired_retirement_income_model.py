import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import time

# Function to estimate remaining time
def estimate_remaining_time(start_time, total_samples, processed_samples):
    elapsed_time = time.time() - start_time
    time_per_sample = elapsed_time / processed_samples
    remaining_samples = total_samples - processed_samples
    remaining_time = remaining_samples * time_per_sample
    return remaining_time

# Start timing the script
start_time = time.time()

# Load your cleaned dataset
print("Loading data...")
data = pd.read_csv('user_data_cleaned.csv')
print("Data loaded. Shape:", data.shape)

# Define features (X) and target (y)
X = data[['age', 'income', 'current_savings', 'InvestmentAccountBalance', 'MonthlySavings', 'retirement_age', 'expected_investment_return_rate']]
y = data['desired_annual_retirement_income']

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split. Training set size:", X_train.shape[0], "Testing set size:", X_test.shape[0])

# Initialize and train the Random Forest model
print("Initializing and training the Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model in batches to estimate time remaining
batch_size = 50000  # Define batch size
num_batches = len(X_train) // batch_size
for i in range(num_batches + 1):
    start_batch_time = time.time()
    batch_start = i * batch_size
    batch_end = min((i + 1) * batch_size, len(X_train))
    X_batch = X_train.iloc[batch_start:batch_end]
    y_batch = y_train.iloc[batch_start:batch_end]
    model.fit(X_batch, y_batch)
    elapsed_batch_time = time.time() - start_batch_time
    processed_samples = (i + 1) * batch_size
    remaining_time = estimate_remaining_time(start_time, len(X_train), processed_samples)
    print(f"Batch {i + 1}/{num_batches + 1} completed. Elapsed time for this batch: {elapsed_batch_time:.2f} seconds. Estimated remaining time: {remaining_time:.2f} seconds.")

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Save the trained model for future use
print("Saving the trained model...")
joblib.dump(model, 'desired_retirement_income_model.pkl')
print("Model saved as 'desired_retirement_income_model.pkl'.")

# End timing and print the total time taken
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
