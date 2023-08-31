import pandas
import numpy as np
from scipy import stats
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

EPOCHS = 300
BATCH_SIZE = 2048
LEARNING_RATE = 0.001
RANDOM_SEED = 42
NUM_MODELS = 5
TEST_SIZE = 0.3

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


### Load dataset 
def load_dataset():
    training_data = pandas.read_csv('./data/train.csv')
    test_data = pandas.read_csv('./data/test.csv')
    training_data, test_data = preprocess_dataset(training_data, test_data)
    training_y = training_data['Survived']
    training_X = training_data.drop(['Survived'], axis=1)
    return training_X, training_y, test_data


### Preprocess dataset
def preprocess_dataset(training_data, test_data):



    print("Number of rows in training data: {}".format(len(training_data)))


    # Drop ID column
    training_data = training_data.drop(['PassengerId'], axis=1)
    # Parse title from the name column and replace the name column with the title column
    training_data['Title'] = training_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Drop the Name column
    training_data = training_data.drop(['Name'], axis=1)
    test_data = test_data.drop(['Name'], axis=1)

    # Replace the value of Ticket with an extracted prefix
    training_data['Ticket_Prefix'] = training_data['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')) > 1 else 'None')
    test_data['Ticket_Prefix'] = test_data['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')) > 1 else 'None')

    # One-hot encode the Ticket_Prefix column
    training_data = pandas.get_dummies(training_data, columns=['Ticket_Prefix'])
    test_data = pandas.get_dummies(test_data, columns=['Ticket_Prefix'])

    # Extract the ticket number from the ticket column
    training_data['Ticket_Numeric'] = training_data['Ticket'].apply(lambda x: int(x.split(' ')[-1]) if x.split(' ')[-1].isdigit() else -1)
    test_data['Ticket_Numeric'] = test_data['Ticket'].apply(lambda x: int(x.split(' ')[-1]) if x.split(' ')[-1].isdigit() else -1)

    # Check for shared tickets
    training_data['Shared_Ticket_Count'] = training_data.groupby('Ticket')['Ticket'].transform('count')
    test_data['Shared_Ticket_Count'] = test_data.groupby('Ticket')['Ticket'].transform('count')

    # Add a ticket length column
    training_data['Ticket_Length'] = training_data['Ticket'].apply(len)
    test_data['Ticket_Length'] = test_data['Ticket'].apply(len)

    # Drop the Ticket column
    training_data = training_data.drop(['Ticket'], axis=1)
    test_data = test_data.drop(['Ticket'], axis=1)

    # Normalize the ticket number column and fill missing values 0
    mean_ticket_numeric = training_data['Ticket_Numeric'].mean()
    std_ticket_numeric = training_data['Ticket_Numeric'].std()
    training_data['Ticket_Numeric'] = (training_data['Ticket_Numeric'] - mean_ticket_numeric) / std_ticket_numeric
    test_data['Ticket_Numeric'] = (test_data['Ticket_Numeric'] - mean_ticket_numeric) / std_ticket_numeric
    training_data['Ticket_Numeric'] = training_data['Ticket_Numeric'].fillna(0)
    test_data['Ticket_Numeric'] = test_data['Ticket_Numeric'].fillna(0)

    # Normalize the shared ticket count column and fill missing values with 0
    mean_shared_ticket_count = training_data['Shared_Ticket_Count'].mean()
    std_shared_ticket_count = training_data['Shared_Ticket_Count'].std()
    training_data['Shared_Ticket_Count'] = (training_data['Shared_Ticket_Count'] - mean_shared_ticket_count) / std_shared_ticket_count
    test_data['Shared_Ticket_Count'] = (test_data['Shared_Ticket_Count'] - mean_shared_ticket_count) / std_shared_ticket_count
    training_data['Shared_Ticket_Count'] = training_data['Shared_Ticket_Count'].fillna(0)
    test_data['Shared_Ticket_Count'] = test_data['Shared_Ticket_Count'].fillna(0)

    # Normalize the ticket length column and fill missing values with 0
    mean_ticket_length = training_data['Ticket_Length'].mean()
    std_ticket_length = training_data['Ticket_Length'].std()
    training_data['Ticket_Length'] = (training_data['Ticket_Length'] - mean_ticket_length) / std_ticket_length
    test_data['Ticket_Length'] = (test_data['Ticket_Length'] - mean_ticket_length) / std_ticket_length
    training_data['Ticket_Length'] = training_data['Ticket_Length'].fillna(0)
    test_data['Ticket_Length'] = test_data['Ticket_Length'].fillna(0)

    # Replace Mlle and Ms with Miss. Replace Mme with Mrs.
    training_data['Title'] = training_data['Title'].replace(['Mlle','Ms'], 'Miss')
    training_data['Title'] = training_data['Title'].replace('Mme', 'Mrs')
    test_data['Title'] = test_data['Title'].replace(['Mlle','Ms'], 'Miss')
    test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')

    # Map titles to integers
    unique_titles = training_data['Title'].unique()
    title_mapping = {title: i for i, title in enumerate(unique_titles)}
    training_data['Title'] = training_data['Title'].map(title_mapping)
    test_data['Title'] = test_data['Title'].map(title_mapping)

    # Normalize the Title column and fill missing values with -1
    mean_title_index = training_data['Title'].mean()
    std_title_index = training_data['Title'].std()
    training_data['Title'] = (training_data['Title'] - mean_title_index) / std_title_index
    test_data['Title'] = (test_data['Title'] - mean_title_index) / std_title_index
    training_data['Title'] = training_data['Title'].fillna(-1)
    test_data['Title'] = test_data['Title'].fillna(-1)    

    # Normalize the Fare column and fill missing values with 0
    mean_fare = training_data['Fare'].mean()
    std_fare = training_data['Fare'].std()
    training_data['Fare'] = (training_data['Fare'] - mean_fare) / std_fare
    test_data['Fare'] = (test_data['Fare'] - mean_fare) / std_fare
    training_data['Fare'] = training_data['Fare'].fillna(0)
    test_data['Fare'] = test_data['Fare'].fillna(0)

    # Normalize the Age column and fill missing values with 0
    mean_age = training_data['Age'].mean()
    std_age = training_data['Age'].std()
    training_data['Age'] = (training_data['Age'] - mean_age) / std_age
    test_data['Age'] = (test_data['Age'] - mean_age) / std_age
    training_data['Age'] = training_data['Age'].fillna(0)
    test_data['Age'] = test_data['Age'].fillna(0)

    # Combine SibSp and Parch into a single column called FamilySize
    training_data['FamilySize'] = training_data['SibSp'] + training_data['Parch'] + 1
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
    
    # Create a new column called IsAlone. Set it to 1 if FamilySize is 1, and 0 otherwise
    training_data['IsAlone'] = 0
    training_data.loc[training_data['FamilySize'] == 1, 'IsAlone'] = 1
    test_data['IsAlone'] = 0
    test_data.loc[test_data['FamilySize'] == 1, 'IsAlone'] = 1

    # Drop SibSp and Parch columns
    training_data = training_data.drop(['SibSp', 'Parch'], axis=1)
    test_data = test_data.drop(['SibSp', 'Parch'], axis=1)

    # One-hot encode the embarked column
    training_data = pandas.get_dummies(training_data, columns=['Embarked'])
    test_data = pandas.get_dummies(test_data, columns=['Embarked'])

    # Convert sex to numeric values, 0 for male and 1 for female
    training_data['Sex'] = training_data['Sex'].map({"male": 0, "female": 1})
    test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})
    
    # Extract deck from cabin
    training_data['Deck'] = training_data['Cabin'].str[:1]
    test_data['Deck'] = test_data['Cabin'].str[:1]

    # Extract cabin number from cabin
    training_data['CabinNumber'] = training_data['Cabin'].str.extract('(\d+)', expand=False)
    test_data['CabinNumber'] = test_data['Cabin'].str.extract('(\d+)', expand=False)

    # Convert cabin number to numeric values
    training_data['CabinNumber'] = pandas.to_numeric(training_data['CabinNumber'])
    test_data['CabinNumber'] = pandas.to_numeric(test_data['CabinNumber'])

    # Drop Cabin column
    training_data = training_data.drop(['Cabin'], axis=1)
    test_data = test_data.drop(['Cabin'], axis=1)

    # Map decks to integers
    deck_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5,
                    "F": 6, "G": 7, "T": 8}
    training_data['Deck'] = training_data['Deck'].map(deck_mapping)
    training_data['Deck'] = training_data['Deck'].fillna(0)
    test_data['Deck'] = test_data['Deck'].map(deck_mapping)
    test_data['Deck'] = test_data['Deck'].fillna(0)

    #Normalize the Deck column and fill missing values with -1
    mean_deck = training_data['Deck'].mean()
    std_deck = training_data['Deck'].std()
    training_data['Deck'] = (training_data['Deck'] - mean_deck) / std_deck
    test_data['Deck'] = (test_data['Deck'] - mean_deck) / std_deck
    training_data['Deck'] = training_data['Deck'].fillna(-1)
    test_data['Deck'] = test_data['Deck'].fillna(-1)

    #Normalize the CabinNumber column and fill missing values with 0
    mean_cabin_number = training_data['CabinNumber'].mean()
    std_cabin_number = training_data['CabinNumber'].std()
    training_data['CabinNumber'] = (training_data['CabinNumber'] - mean_cabin_number) / std_cabin_number
    test_data['CabinNumber'] = (test_data['CabinNumber'] - mean_cabin_number) / std_cabin_number
    training_data['CabinNumber'] = training_data['CabinNumber'].fillna(0)
    test_data['CabinNumber'] = test_data['CabinNumber'].fillna(0)
    
    # One-hot encode the Pclass column
    training_data = pandas.get_dummies(training_data, columns=['Pclass'])
    test_data = pandas.get_dummies(test_data, columns=['Pclass'])

    # Align the dataframes to ensure they have the same columns
    training_data, test_data = training_data.align(test_data, axis=1, fill_value=0)

    test_data.drop(['Survived'], axis=1, inplace=True)
    training_data.drop("PassengerId", axis=1, inplace=True)


    # Remove outliers from training data
    numerical_cols = ['FamilySize', 'Age', 'Fare']
    z_scores = np.abs(stats.zscore(training_data[numerical_cols]))
    training_data = training_data[(z_scores < 3).all(axis=1)]

    return  training_data, test_data


### Build model
def build_model():
    inputs = layers.Input(shape=(52,))
    x = layers.Dense(128, activation='relu')(inputs)    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model

if __name__ == "__main__":
    X, y, test_data = load_dataset()

    

    models = []
    for i in range(NUM_MODELS):
        cont = True        
        while cont:
            model = build_model()
            model.compile(loss='binary_crossentropy',
                            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                            metrics=['accuracy',
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall')])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random.randint(0, 1000))    
            learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                                        patience=10,
                                                                        verbose=1,
                                                                        factor=0.9,
                                                                        min_lr=0.00001)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                            patience=20,  # Number of epochs with no improvement to wait before stopping
                                                            restore_best_weights=True)

            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), callbacks=[early_stopping, learning_rate_reduction])
            results = model.evaluate(X_test, y_test)
            if results[1] > 0.8:
                cont = False
        models.append(model)

    predictions = []
    for model in models:
        predictions.append(model.predict(test_data.drop(['PassengerId'], axis=1)))

    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)

    submission = pandas.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions.flatten()})

    submission['Survived'] = submission['Survived'].apply(lambda x: 1 if x > 0.5 else 0)
    submission.to_csv('./data/submission.csv', index=False)