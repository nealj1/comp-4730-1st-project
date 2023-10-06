# PACKAGES ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#SETTINGS ---------------------------------------------------------------
pd.set_option('display.max_colwidth', None)

# Functions -------------------------------------------------------------


# Load the dataset
mushroom_data = pd.read_csv('Mushroom/agaricus-lepiota.csv')

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

# Print or display the summary DataFrame
df_summary = summary(mushroom_data)
print(df_summary)

# Check for ? in the data
question_mark_columns = columns_with_question_marks(mushroom_data)
for column_name in question_mark_columns:
    count_question_marks = mushroom_data[column_name].str.count("\?").sum()
    print(f"Number of '?' in '{column_name}': {count_question_marks}")

#Drop Columns
mushroom_data = mushroom_data.drop(columns=['stalk-root']) # column has lots of ?
mushroom_data = mushroom_data.drop(columns=['veil-type']) # is a constant column so does not add anything

#Encodes
mushroom_data = pd.get_dummies(mushroom_data)
#or this one ????
'''
# Encode categorical features
label_encoder = LabelEncoder()
for column in mushroom_data.columns:
    mushroom_data[column] = label_encoder.fit_transform(mushroom_data[column])
'''



# Do ai training stuff
'''
# Split the data into train and test sets
X = mushroom_data.drop('class', axis=1)
y = mushroom_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
'''

print(mushroom_data.count)