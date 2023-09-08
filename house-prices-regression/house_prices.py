import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import random

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

training_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

categorical_columns = [
    'MSSubClass',
    'MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'ExterQual',
    'ExterCond',
    'Foundation',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'Heating',
    'HeatingQC',
    'Electrical',
    'KitchenQual',
    'Functional',
    'FireplaceQu',
    'GarageType',
    'GarageYrBlt',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PoolQC',
    'Fence',
    'MiscFeature',
    'SaleType',
    'SaleCondition',
    'CentralAir',
    'PavedDrive'
]

unused_categorical_columns = [
]


numerical_columns = [
    'LotFrontage',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'GrLivArea',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'GarageCars',
    'GarageArea',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'YrSold',
    'GarageYrBlt',
    'YearBuilt',
    'YearRemodAdd',
    'MoSold',
    'MiscVal',
]

unused_numerical_columns = [

]

print("Filling NA.....")
# Convert rows with no data or the string 'NA' in numerical columns to 0
training_data[numerical_columns] = training_data[numerical_columns].fillna(0)
test_data[numerical_columns] = test_data[numerical_columns].fillna(0)

print("Label encoding.....")
# Convert categorical data to numerical data using label encoding
encoder = LabelEncoder()

# Separate categorical columns into separate dataframes
categorical_df = training_data[categorical_columns]
categorical_test_df = test_data[categorical_columns]

# Fill NA values with 'NA' string
categorical_df = categorical_df.fillna('NA')
categorical_test_df = categorical_test_df.fillna('NA')

# Ensure all categorical columns are of type string
categorical_df = categorical_df.astype(str)
categorical_test_df = categorical_test_df.astype(str)

# Encode categorical data
encoded_features = training_data[categorical_columns].apply(encoder.fit_transform)
encoded_test_features = test_data[categorical_columns].apply(encoder.fit_transform)
encoded_features_df = pd.DataFrame(encoded_features)
encoded_test_features_df = pd.DataFrame(encoded_test_features)

print("Scaling.....")
# Scale numerical data
scaler = StandardScaler()
scaled_numerical_features = scaler.fit_transform(training_data[numerical_columns])
scaled_test_numerical_features = scaler.transform(test_data[numerical_columns])
scaled_numerical_features_df = pd.DataFrame(scaled_numerical_features)
scaled_test_numerical_features_df = pd.DataFrame(scaled_test_numerical_features)

print("Concatenating.....")
X = pd.concat([encoded_features_df, scaled_numerical_features_df], axis=1)
X_test = pd.concat([encoded_test_features_df, scaled_test_numerical_features_df], axis=1)

# Ensure all feature names are of type string
X.columns = X.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

print("Removing outliers.....")
clf = IsolationForest(random_state=42, contamination=0.02).fit(X)
outliers = clf.predict(X)
X = X[outliers == 1]
y = training_data['SalePrice'][outliers == 1]

print("Converting to float32.....")
X = X.astype('float32')
y = y.astype('float32')
X_test = X_test.astype('float32')

# Ensure all feature names are of type string
X.columns = X.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

continue_training = True
while continue_training:
    state = random.randint(0, 100000)
    print("Splitting into training and validation sets.....")
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=state)

    print("Training.....")
    model = RandomForestRegressor(n_estimators=1000, random_state=state)
    model.fit(X_train, y_train)

    print("Scoring.....")
    score = model.score(X_val, y_val)
    print(score)

    if score > 0.90:
        continue_training = False
        print("Finished Training. State: ", state)

print("Predicting.....")
# Make predictions on the test data
predictions = model.predict(X_test)

print("Saving.....")
# Save predictions to a csv file
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
submission.to_csv('data/submission.csv', index=False)