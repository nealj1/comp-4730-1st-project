# PACKAGES ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import printing as dataPrinting

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

#Drop Columns
mushroom_data = mushroom_data.drop(columns=['stalk-root']) # column has lots of ?
mushroom_data = mushroom_data.drop(columns=['veil-type']) # is a constant column so does not add anything

#Encodes categorical features
label_encoder = LabelEncoder()
for column in mushroom_data.columns:
    mushroom_data[column] = label_encoder.fit_transform(mushroom_data[column])

# Split the data into train and test sets
X = mushroom_data.drop('class', axis=1)
y = mushroom_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of models
models = [
    RandomForestClassifier(random_state=42),
    LogisticRegression(random_state=42, max_iter=1000),
    SVC(random_state=42),
    DecisionTreeClassifier(random_state=42),
    GaussianNB()
]

# Lists to store model names and their corresponding accuracies
model_names = []
accuracies = []
confusions = []
reports = []

# Iterate through the list of models and train/evaluate each one
for model in models:
    # Scale the data before training the model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model on the scaled training data
    model.fit(X_train_scaled, y_train)

    # Make predictions on the scaled test data
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    # Store model name and accuracy
    model_names.append(model.__class__.__name__)
    accuracies.append(accuracy)
    confusions.append(confusion)
    reports.append(report)

    # Print the results for each model
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)
    print("\n")

# Lists to store model names and their corresponding fold accuracies
cv_model_names = []
fold_accuracies = []
mean_accuracies = []
fold_confusions = []
fold_reports = []

# Specify the number of folds for cross-validation
num_folds = 5  # You can adjust this number as needed

# Create a cross-validation splitter
cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterate through the list of models and perform cross-validation
for model in models:
    # Initialize lists to store fold-specific results
    fold_accuracies_model = []
    fold_confusions_model = []
    fold_reports_model = []
    
    # Perform cross-validation and calculate accuracy scores for each fold
    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Store fold-specific results
        fold_accuracies_model.append(accuracy)
        fold_confusions_model.append(confusion)
        fold_reports_model.append(report)
    
    # Store model name and fold accuracies
    cv_model_names.append(model.__class__.__name__)
    fold_accuracies.append(fold_accuracies_model)
    fold_confusions.append(fold_confusions_model)
    fold_reports.append(fold_reports_model)
    
    # Calculate and store mean accuracy across all folds
    mean_accuracy = np.mean(fold_accuracies_model)
    mean_accuracies.append(mean_accuracy)

# Print the results for each fold and model at the end
for model_name, fold_accuracy, fold_confusion, fold_report in zip(cv_model_names, fold_accuracies, fold_confusions, fold_reports):
    print(f"Model: {model_name}")
    for fold_num, (accuracy, confusion, report) in enumerate(zip(fold_accuracy, fold_confusion, fold_report), start=1):
        print(f"Fold {fold_num} Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:\n", confusion)
        print("Classification Report:\n", report)
        print("\n")
    print(f"Mean Accuracy: {mean_accuracies[cv_model_names.index(model_name)]:.4f}\n")


# DATA ANALYSIS
dataPrinting.print_class_compare(mushroom_data,"Distribution of Mushroom Classes Balanced" )
dataPrinting.print_heatmap(mushroom_data)
dataPrinting.print_compare_model_accuracies(model_names, accuracies)
dataPrinting.print_logistic_regression_visual(X, models)
dataPrinting.print_decision_tree_classifer(X, cv_model_names, models)
print(mean_accuracies)
dataPrinting.print_random_forest_classifier(X, models)

#CROSS VALIDATION DATA ANALYSIS
#dataPrinting.print_crossfold_data(cv_model_names, fold_accuracies, mean_accuracies, num_folds)
dataPrinting.print_crossfold_compare_all_same_graph(cv_model_names, fold_accuracies, mean_accuracies, num_folds)
dataPrinting.print_crossfold_compare_separate(cv_model_names, fold_accuracies, mean_accuracies, num_folds)

