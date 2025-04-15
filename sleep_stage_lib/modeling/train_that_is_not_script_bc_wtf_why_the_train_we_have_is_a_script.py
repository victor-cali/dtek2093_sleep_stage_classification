import pandas as pd
from numpy import ravel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_eog_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def eval_eog_random_forest(rf: RandomForestClassifier, le: LabelEncoder, X_test: pd.DataFrame, y_test: pd.Series):
    # Evaluate the model
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=le.classes_)
    return accuracy, report


def train_and_eval_eog_random_forest(features: pd.DataFrame, labels: pd.DataFrame):
    # Encode labels and split the data
    le = LabelEncoder()
    labels = labels.values.ravel()
    labels_encoded = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.3, random_state=42)
    # Train and evaluate the model
    rf = train_eog_random_forest(X_train, y_train)
    # Evaluate on the test set
    accuracy, report = eval_eog_random_forest(rf, le, X_test, y_test)
    return accuracy, report
