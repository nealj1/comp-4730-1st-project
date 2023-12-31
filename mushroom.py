import pandas as pd
import os 
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, learning_curve, train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import plot_tree

def read_data(path):
    df = pd.read_csv(path)
    return df
    
def check_missing_values(df):
    missing_values = (df == '?').sum()
    
    max_missing_col = missing_values.idxmax()
    max_missing_val = missing_values.max()
    
    print(f"Column '{max_missing_col}' has the most missing values: {max_missing_val}")
    
    print(df.describe())

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
    
    x = df.drop(columns=["class"])
    y = df['class']
    
    encoder = OneHotEncoder(sparse=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(x), columns=encoder.get_feature_names_out(x.columns))
    y_encoded = y.map({'EDIBLE': 0, 'POISONOUS': 1})
    
    x_train, x_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, shuffle=True, stratify=y_encoded)
    return x_train, x_test, y_train, y_test

def plot_decision_tree(model, feature_names, class_names):
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, feature_names, class_names):
    print(f"\n starting {model.__class__.__name__}....")
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    print("Performance on Training data:")
    print_metrics(y_train, y_train_pred)
    
    # Evaluate on testing data
    y_test_pred = model.predict(x_test)
    print("\nPerformance on Test data:")
    print_metrics(y_test, y_test_pred)

    if isinstance(model, DecisionTreeClassifier):
        plot_decision_tree(model, feature_names, class_names)

def hyperparameter_search(model, params, x_train, y_train, x_test, y_test, feature_names, class_names):
    print(f"\nStarting {model.__class__.__name__}....")
    
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    
    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(x_train)
    print("Performance on Training data:")
    print_metrics(y_train, y_train_pred)
    y_pred = best_model.predict(x_test)
    print("\nPerformance on Test data:")
    print_metrics(y_test, y_pred)
    
    
    # if isinstance(model, DecisionTreeClassifier):
    #     plot_decision_tree(model, feature_names, class_names)

def print_metrics(y_test, y_pred):
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    precision = precision_score(y_test, y_pred)
    print(f'Precision: {precision * 100:.2f}%')

    recall = recall_score(y_test, y_pred)
    print(f'Recall: {recall * 100:.2f}%')

    f1 = f1_score(y_test, y_pred)
    print(f'F1 Score: {f1 * 100:.2f}%')

def plot_learning_curve(model, modelName, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
    
    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.plot(train_sizes, train_mean, '--', color="r",  label="Training score")
    plt.plot(train_sizes, test_mean, color="g", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.1)

    # Create plot
    plt.title(f"Learning Curve {modelName}")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    data_path = os.path.join( "Mushroom", "agaricus-lepiota.csv")
    mushroom_data = read_data(data_path)

    #handle missing values
    # ADD returned and did a loop dbl check main and add in here
    check_missing_values(mushroom_data)
    
    # DROP duplicate settings
    mushroom_data = mushroom_data.drop_duplicates()

    mushroom_data = mushroom_data.drop(columns=["stalk-root"]) # we end up dropping this because it has too many missing values

    # now we check for other columns that will not contribute in helping us find poisonous and non poisnonous mushrooms
    #from the above describe call, we can see that veil has only 1 type of value same for everything so it wont add anything
    mushroom_data = mushroom_data.drop(columns=["veil-type"]) # we drop it 

    x_train, x_test, y_train, y_test  = preprocess_data(mushroom_data)
    feature_names = x_train.columns.tolist()
    class_names = ['poisonous', 'edible'] 
    
    models = [
        LogisticRegression(),
        DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0),
        RandomForestClassifier(criterion="gini", max_depth=4, random_state=0),
        GaussianNB(),
        SVC(kernel="linear", C=0.025, random_state=42),
        AdaBoostClassifier(random_state=42, n_estimators=10),
    ]
    
    
    for model in models: 
        train_and_evaluate_model(model, x_train, y_train, x_test, y_test, feature_names, class_names) #loop to train and evaluate 
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy') # 5-fold CV
        print(f"{model.__class__.__name__} CV Accuracy: {np.mean(cv_scores):.2f} +/- {np.std(cv_scores):.2f}")
        plot_learning_curve(model, model.__class__.__name__, x_train, y_train)
    
        
    print("\nPerforming hyperparameter search now...")
    
    models_with_params = [
        (LogisticRegression(), {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'], 'max_iter': [100, 200, 300]}),
        (DecisionTreeClassifier(random_state=0), {'criterion': ['entropy','gini', 'log_loss'], 'max_depth': [3, 5, 7, 9]}),
        (RandomForestClassifier(random_state=0), {'criterion': ['entropy','gini', 'log_loss'], 'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]}),
        (GaussianNB(), {'var_smoothing': [1e-9, 1e-8, 1e-10]}),
        (SVC(random_state=42), {'C': [0.025, 0.25, 2.5], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2, 3, 4]}),
        (AdaBoostClassifier(random_state=42), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 1]})
        
    ]

    for model, params in models_with_params:
        hyperparameter_search(model, params, x_train, y_train, x_test, y_test, feature_names, class_names)
        