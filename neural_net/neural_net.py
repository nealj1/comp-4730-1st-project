import tensorflow as tf
import os
import pandas as pd 
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def read_data(path):
    df = pd.read_csv(path)
    return df

def encode_labels(df):
    """Encodes string labels into numerical values and returns the DataFrame and mappings."""
    encoder = LabelEncoder()
    mappings = []
    for column in df.columns:
        df[column] = encoder.fit_transform(df[column])
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
        mappings.append(mappings_dict)
    return df, mappings

def preprocess_data(df):
    """Preprocesses the data and returns train and test sets."""
    df, mappings = encode_labels(df)
    x = df.drop(columns=["class"])
    y = df['class']
    
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test, mappings


data_path = os.path.join( "/root", "comp-4730-1st-project", "Mushroom", "agaricus-lepiota.csv")
mushroom_data = read_data(data_path)

mushroom_data = mushroom_data.drop(columns=["stalk-root"]) # we end up dropping this because it has too many missing values

# now we check for other columns that will not contribute in helping us find poisonous and non poisnonous mushrooms
#from the above describe call, we can see that veil has only 1 type of value same for everything so it wont add anything
mushroom_data = mushroom_data.drop(columns=["veil-type"]) # we drop it 

x_train, x_test, y_train, y_test, mappings  = preprocess_data(mushroom_data)

model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

y_pred = model.predict(x_test)

binary_predictions = [1 if pred >= 0.5 else 0 for pred in y_pred]

# If you have mappings for the target variable, you can convert numerical predictions back to original labels
original_labels = mappings[0]  # assuming mappings is available and the target column is the last one
label_predictions = [original_labels[pred] for pred in binary_predictions]

print(label_predictions)