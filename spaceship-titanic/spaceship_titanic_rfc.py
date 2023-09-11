import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score

categorical_columns = [
    "HomePlanet",
    "Destination",
    "Deck",
    "CabinClass",
    "LastName",
    "CryoSleep",
    "VIP",
    #"GroupNumber"
]

numerical_columns = [
    "CabinNumber",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "LastNameCount",
    #"GroupSize"
]

training_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

y = training_data['Transported'].copy()
X = training_data.drop('Transported', axis=1).copy()

def process_data(df):
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNumber'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['CabinClass'] = df['Cabin'].str.split('/').str[2]
    df.drop('Cabin', axis=1, inplace=True)
    df['GroupNumber'] = df['PassengerId'].str.split('[.]').str[1]
    df['LastName'] = df['Name'].str.split('[\s]').str[1]
    df.drop('Name', axis=1, inplace=True)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 19, 59, 120], labels=['Child', 'Teen', 'Adult', 'Senior'])
    df.drop('Age', axis=1, inplace=True)
    return df

X = process_data(X)
test_data = process_data(test_data)

# combined = pd.concat([X, test_data], axis=0, ignore_index=True)
# combined['GroupSize'] = combined.groupby('GroupNumber')['GroupNumber'].transform('count')

# X = combined.iloc[:len(X)].copy()
# test_data = combined.iloc[len(X):].copy()

X.drop("GroupNumber", axis=1, inplace=True)
test_data.drop("GroupNumber", axis=1, inplace=True)

X.drop('PassengerId', axis=1, inplace=True)

X['LastNameCount'] = X.groupby('LastName')['LastName'].transform('count')
test_data['LastNameCount'] = test_data.groupby('LastName')['LastName'].transform('count')

X = pd.get_dummies(X, columns=['AgeGroup'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['AgeGroup'], drop_first=True)

X.fillna({
    'Deck': 'Unknown',
    'CabinNumber': X['CabinNumber'].mean(),
    'CabinClass': 'Unknown',
    'LastName': 'Unknown',
    'LastNameCount': -1,
    'HomePlanet': 'Unknown',
    'Destination': 'Unknown'
}, inplace=True)

test_data.fillna({
    'Deck': 'Unknown',
    'CabinNumber': X['CabinNumber'].mean(),
    'CabinClass': 'Unknown',
    'LastName': 'Unknown',
    'LastNameCount': -1,
    'HomePlanet': 'Unknown',
    'Destination': 'Unknown'
}, inplace=True)

test_X = test_data.drop('PassengerId', axis=1).copy()

for column in categorical_columns:
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded = encoder.fit_transform(X[column].values.reshape(-1, 1))
    encoded_df = pd.DataFrame(encoded.toarray(), columns=[f"{column}_{cat}" for cat in encoder.categories_[0]])
    X.drop(column, axis=1, inplace=True)
    X = pd.concat([X, encoded_df], axis=1)
    
    encoded_test = encoder.transform(test_X[column].values.reshape(-1, 1))
    encoded_test_df = pd.DataFrame(encoded_test.toarray(), columns=[f"{column}_{cat}" for cat in encoder.categories_[0]])
    test_X.drop(column, axis=1, inplace=True)
    test_X = pd.concat([test_X, encoded_test_df], axis=1)

scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
test_X[numerical_columns] = scaler.transform(test_X[numerical_columns])

encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

# Convert the one-hot encoded labels back to a single column
y_label = np.argmax(y_encoded, axis=1)

# Check for NaN values in the dataset
nan_columns = X.isnull().sum()
print(f"Number of NaN values in each column:\n{nan_columns}")

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

# 2. Check for Infinite values:

# Replace infinity values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# After replacing infinite values with NaN, we might have new NaN values, 
# so we fill them again with the mean of the column
X.fillna(X.mean(), inplace=True)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y_label, cv=kf, scoring='accuracy')

print("Average accuracy across all folds: ", np.mean(scores))
print("Standard deviation of accuracy across all folds: ", np.std(scores))

# Train on the entire dataset
rf.fit(X, y_label)

# Check for NaN values in the test dataset
nan_columns_test = test_X.isnull().sum()
print(f"Number of NaN values in each test column:\n{nan_columns_test}")

# Fill NaN values with the mean of the column in test_X
test_X.fillna(test_X.mean(), inplace=True)

# Check for Infinite values in test_X:

# Replace infinity values with NaN in test_X
test_X.replace([np.inf, -np.inf], np.nan, inplace=True)

# After replacing infinite values with NaN in test_X, we might have new NaN values, 
# so we fill them again with the mean of the column
test_X.fillna(test_X.mean(), inplace=True)

# Make predictions on the test data
predictions = rf.predict(test_X)

# Create a dataframe with the PassengerId and predictions
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': predictions.flatten()})

# Convert Transported to boolean
submission['Transported'] = submission['Transported'].astype(bool)

# Save the dataframe to a csv file
submission.to_csv('data/submission.csv', index=False)
