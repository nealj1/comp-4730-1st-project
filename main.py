# PACKAGES ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import printing as dataPrinting
import matplotlib.pyplot as plt
import seaborn as sns
#SETTINGS ---------------------------------------------------------------
pd.set_option('display.max_colwidth', None)
# Set the display option to show all rows
#pd.set_option('display.max_rows', None)

# Functions -------------------------------------------------------------
def summary(df):
    summary_df = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary_df['duplicated'] = df.duplicated().sum()
    summary_df['missing#'] = df.isna().sum()
    summary_df['missing%'] = (df.isna().sum()) / len(df)
    summary_df['uniques'] = df.nunique().values
    summary_df['count'] = df.count().values
    # Add a column 'unique_values' containing unique values for each column
    unique_values = []
    for column in df.columns:
        unique_values.append(df[column].unique())
    summary_df['unique_values'] = unique_values

    return summary_df


def columns_with_question_marks(df):
    
    columns_with_question_mark = []
    # Loop through all columns in the DataFrame
    for column in df.columns:
        # Create a boolean mask for rows with "?"
        mask = df[column] == "?"

        # Use the mask to filter the rows
        rows_with_question_mark = df[mask]

        # Check if there are any rows with "?"
        if not rows_with_question_mark.empty:
            columns_with_question_mark.append(column)
    return columns_with_question_mark

def balance_dataset(mushroom_data, rows_to_remove):
    # Find rows where 'class' is "EDIBLE" and 'stalk-root' is "?"
    edible_with_question_mark_stalk_root = mushroom_data[(mushroom_data['class'] == 'EDIBLE') & (mushroom_data['stalk-root'] == '?')]
    # Get the indices of the rows to remove
    indices_to_remove = edible_with_question_mark_stalk_root.index[:rows_to_remove]
    # Remove the specified rows from the DataFrame
    mushroom_data = mushroom_data.drop(indices_to_remove)
    # Verify that the rows have been removed
    print("Number of rows removed:", len(indices_to_remove))
    return mushroom_data

# Main --------------------------------------------------------------------
# Load the dataset
mushroom_data = pd.read_csv('Mushroom/agaricus-lepiota.csv')

# Print or display the summary DataFrame
df_summary = summary(mushroom_data)
print(df_summary)

# Dropped duplicate rows
# Get the count of duplicated rows
print(f"Number of duplicate rows before dropping: {mushroom_data.duplicated().sum()}")
mushroom_data = mushroom_data.drop_duplicates()
print(f"Number of duplicate rows after dropping: {mushroom_data.duplicated().sum()}")

# Check for ? in the data
question_mark_columns = columns_with_question_marks(mushroom_data)
for column_name in question_mark_columns:
    count_question_marks = mushroom_data[column_name].str.count("\?").sum()
    print(f"Number of '?' in '{column_name}': {count_question_marks}")

# Find rows where 'class' is "EDIBLE" and 'stalk-root' is "?"
edible_with_question_mark_stalk_root = mushroom_data[(mushroom_data['class'] == 'EDIBLE') & (mushroom_data['stalk-root'] == '?')]
edible = mushroom_data[(mushroom_data['class'] == 'EDIBLE')]
poisonous = mushroom_data[(mushroom_data['class'] == 'POISONOUS')]
dataPrinting.print_class_compare(mushroom_data, "Distribution of Mushroom Classes")

# Balancing
mushroom_data = balance_dataset(mushroom_data, edible.shape[0] - poisonous.shape[0])
print(f'mushroom count: {edible.shape[0] - poisonous.shape[0]}')
dataPrinting.print_class_compare(mushroom_data,"Distribution of Mushroom Classes Balanced" )

#Drop Columns
mushroom_data = mushroom_data.drop(columns=['stalk-root']) # column has lots of ?
mushroom_data = mushroom_data.drop(columns=['veil-type']) # is a constant column so does not add anything


#Encodes categorical features
label_encoder = LabelEncoder()
for column in mushroom_data.columns:
    mushroom_data[column] = label_encoder.fit_transform(mushroom_data[column])

dataPrinting.print_heatmap(mushroom_data)


# Split the data into train and test sets
X = mushroom_data.drop('class', axis=1)
y = mushroom_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of models
models = [
    RandomForestClassifier(random_state=42),
    LogisticRegression(random_state=42),
    SVC(random_state=42),
    DecisionTreeClassifier(random_state=42),
    GaussianNB()
]

# Split the data into train and test sets
X = mushroom_data.drop('class', axis=1)
y = mushroom_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lists to store model names and their corresponding accuracies
model_names = []
accuracies = []

# Iterate through the list of models and train/evaluate each one
for model in models:
    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    # Store model name and accuracy
    model_names.append(model.__class__.__name__)
    accuracies.append(accuracy)

    # Print the results for each model
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)
    print("\n")

dataPrinting.print_compare_model_accuracies(model_names, accuracies)