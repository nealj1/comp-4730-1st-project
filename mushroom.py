import pandas as pd
import os 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



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
    df, mappings = encode_labels(df)
    x = df.drop(columns=["class"])
    y = df['class']
    
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test, mappings

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    print(f"\n starting {model.__class__.__name__}....")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print_metrics(y_test, y_pred)


def print_metrics(y_test, y_pred):
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    precision = precision_score(y_test, y_pred)
    print(f'Precision: {precision * 100:.2f}%')

    recall = recall_score(y_test, y_pred)
    print(f'Recall: {recall * 100:.2f}%')

    f1 = f1_score(y_test, y_pred)
    print(f'F1 Score: {f1 * 100:.2f}%')



if __name__ == "__main__":

    data_path = os.path.join( "Mushroom", "agaricus-lepiota.csv")
    mushroom_data = read_data(data_path)

    #handle missing values
    # ADD returned and did a loop dbl check main and add in here
    check_missing_values(mushroom_data)

    mushroom_data = mushroom_data.drop(columns=["stalk-root"]) # we end up dropping this because it has too many missing values

    # now we check for other columns that will not contribute in helping us find poisonous and non poisnonous mushrooms
    #from the above describe call, we can see that veil has only 1 type of value same for everything so it wont add anything
    mushroom_data = mushroom_data.drop(columns=["veil-type"]) # we drop it 

    x_train, x_test, y_train, y_test, mappings  = preprocess_data(mushroom_data)
    
    models = [
        LogisticRegression(),
        DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0),
        RandomForestClassifier(criterion="entropy", max_depth=4, random_state=0),
        GaussianNB(),
        SVC(kernel="linear", C=0.025, random_state=42)
    ]
    
    
    for model in models:
        train_and_evaluate_model(model, x_train, y_train, x_test, y_test)

