import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


def train_eog_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def recursive_feature_elimination(rf: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> RFECV:
    rfecv = RFECV(
        estimator=rf,
        step=1,
        cv=StratifiedKFold(5),
        scoring='accuracy',
        n_jobs=-1
    )
    rfecv.fit(X_train, y_train)
    print("Optimal features:", rfecv.n_features_)
    return rfecv


def find_best_rf_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series):
    # Using grid search, find best random forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    feature_eliminator = recursive_feature_elimination(rf, X_train, y_train)
    X_train = feature_eliminator.transform(X_train)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    return best_rf, feature_eliminator


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
    rf, feature_eliminator = find_best_rf_hyperparameters(X_train, y_train)
    # Evaluate on the test set
    X_test = feature_eliminator.transform(X_test)
    accuracy, report = eval_eog_random_forest(rf, le, X_test, y_test)
    return accuracy, report
