import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score


categorical_columns = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "AgeGroup",
    "FareGroup",    
    "IsAlone",
    "HasCabin",
    "Deck",
    "TicketPrefix",
]

numerical_columns = [
    "FamilySize",
    "CabinNumber",
    "TicketNumber",
]

training_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

y = training_data['Survived'].copy()
X = training_data.drop('Survived', axis=1).copy()

test_X = test_data.drop('PassengerId', axis=1).copy()

def can_convert_to_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def preprocess_data(df):

    # Extract Title from Name, store in column and drop Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
    df.drop('Name', axis=1, inplace=True)

    # Replace Miss with Ms, Mlle with Miss, and Mme with Mrs
    df['Title'] = df['Title'].str.replace('Miss', 'Ms')
    df['Title'] = df['Title'].str.replace('Mlle', 'Miss')
    df['Title'] = df['Title'].str.replace('Mme', 'Mrs')

    # Extract Deck, which is the first letter of the Cabin string, store in column
    df['Deck'] = df['Cabin'].str[0]
    
    #If Cabin contains a space (i.e. multiple cabins), take the first one
    df['Cabin'] = df['Cabin'].str.split('[\s]').str[0]

    # Extract CabinNumber from Cabin, which is everything after the first letter of the Cabin string, store in column, and drop Cabin
    df['CabinNumber'] = pd.to_numeric(df['Cabin'].str[1:], errors='coerce')
    df.drop('Cabin', axis=1, inplace=True)    

    # Fill missing cabin numbers with 0
    df['CabinNumber'].fillna(0, inplace=True)

    # Split the ticket string
    ticket_split = df['Ticket'].str.split('[\s]')

    # Always use the last item as the ticket number
    df['TicketNumber'] = ticket_split.apply(lambda x: float(x[-1]) if can_convert_to_float(x[-1]) else np.nan)

    # Use all items except the last one as the ticket prefix, joined by space
    df['TicketPrefix'] = ticket_split.apply(lambda x: ' '.join(x[:-1]))

    # Drop the original Ticket column
    df.drop('Ticket', axis=1, inplace=True)

    # Create FamilySize column
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Create IsAlone column
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Create HasCabin column
    df['HasCabin'] = 0
    df.loc[df['CabinNumber'] > 0, 'HasCabin'] = 1

    # Create FareGroup column
    df['FareGroup'] = pd.cut(df['Fare'], bins=[0, 7.91, 14.454, 31, 513], labels=['Low', 'Mid', 'High', 'VeryHigh'])
    df.drop('Fare', axis=1, inplace=True)

    # Create AgeGroup column
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 19, 59, 120], labels=['Child', 'Teen', 'Adult', 'Senior'])
    df.drop('Age', axis=1, inplace=True)

    # Fill missing Embarked values with S
    df['Embarked'].fillna('S', inplace=True)

    # Fill missing FareGroup values with Low
    df['FareGroup'].fillna('Low', inplace=True)

    # Fill missing AgeGroup values with Adult
    df['AgeGroup'].fillna('Adult', inplace=True)

    # Fill missing Deck values with Unknown
    df['Deck'].fillna('Unknown', inplace=True)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Scale numerical columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Replace NaNs with median values 
    df.fillna(df.median(), inplace=True)

    return df

X = preprocess_data(X)
test_X = preprocess_data(test_X)

X, test_X = X.align(test_X, join='left', axis=1)
test_X.fillna(0, inplace=True)

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42)

# Create a KFold object with 5 splits
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rfc, X, y, cv=kf, scoring='accuracy')

# Print the mean of the cross-validated scores
print(scores.mean())

# Fit the model to the training data
rfc.fit(X, y)

# Create predictions for the test data
predictions = rfc.predict(test_X)

# Create a dataframe with two columns: PassengerId & Survived. Survived contains your predictions
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions.flatten()})

# Convert Survived back to int
submission['Survived'] = submission['Survived'].astype(int)

# Save the submission to a csv file
submission.to_csv('data/submission.csv', index=False)