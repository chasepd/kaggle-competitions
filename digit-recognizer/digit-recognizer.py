import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

NUM_MODELS = 15

# Import data from data/train.csv and data/test.csv
training_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

y = training_data['label']
X = training_data.drop('label', axis=1)

# Normalize X
X = X / 255.0
test_data = test_data / 255.0

# One-hot encode y
encoder = OneHotEncoder()
y = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

# Reshape X and test_data to 28x28x1 3D matrices
X = X.values.reshape(-1, 28, 28, 1)
test_data = test_data.values.reshape(-1, 28, 28, 1)

models = []

for i in range(NUM_MODELS):

    # Create a ConvNet model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Create an image data generator
    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, vertical_flip=False, horizontal_flip=False)

    # Create an early stopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=64, callbacks=[early_stopping, lr_reduction], validation_data=(X_val, y_val))

    models.append(model)

all_predictions = []

for model in models:
    # Predict the test data
    predictions = model.predict(test_data)
    all_predictions.append(predictions)

# Average the predictions from all models and convert to labels
predictions = np.argmax(np.mean(all_predictions, axis=0), axis=1)

# Create a submission file
submission = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': predictions})
submission.to_csv('data/submission.csv', index=False)