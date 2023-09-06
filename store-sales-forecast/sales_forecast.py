import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data from the given csv files
holidays = pd.read_csv('data/holidays_events.csv')
oil = pd.read_csv('data/oil.csv')
stores = pd.read_csv('data/stores.csv')
transactions_data = pd.read_csv('data/train.csv')
transactions_counts = pd.read_csv('data/transactions.csv')

# Merging datasets for training data
merged_data = transactions_data.merge(oil, on='date', how='left')
merged_data = merged_data.merge(stores, on='store_nbr', how='left')
merged_data = merged_data.merge(transactions_counts[['date', 'store_nbr', 'transactions']], on=['date', 'store_nbr'], how='left', suffixes=('', '_y'))
merged_data = merged_data[merged_data.columns.drop(list(merged_data.filter(regex='_y')))]
merged_data = merged_data.merge(holidays, on='date', how='left')
merged_data = merged_data.fillna(0)

train_data = merged_data.copy()

categorical_columns = ['family']

# Convert categorical data to numerical data using one-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(train_data[categorical_columns])
encoded_features_df = pd.DataFrame(encoded_features.toarray())

X = encoded_features_df
y = train_data['sales']

X = X.astype('float32')
y = y.astype('float32')

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_val, y_val)
print(score)

# Load test data
test_data = pd.read_csv('data/test.csv')

# Preprocess test data
test_data = test_data.merge(oil, on='date', how='left')
test_data = test_data.merge(stores, on='store_nbr', how='left')
test_data = test_data.merge(transactions_counts[['date', 'store_nbr', 'transactions']], on=['date', 'store_nbr'], how='left', suffixes=('', '_y'))
test_data = test_data[test_data.columns.drop(list(test_data.filter(regex='_y')))]
test_data = test_data.merge(holidays, on='date', how='left')
test_data = test_data.fillna(0)

# Convert categorical data to numerical data for test data
encoded_test_features = encoder.transform(test_data[categorical_columns])
encoded_test_features_df = pd.DataFrame(encoded_test_features.toarray())

X_test = encoded_test_features_df
X_test = X_test.astype('float32')

# Make predictions on the test data
predictions = model.predict(X_test)

# Save predictions to submission.csv
submission = pd.DataFrame({'id': test_data['id'], 'sales': predictions.flatten()})
submission.to_csv('data/submission.csv', index=False)
