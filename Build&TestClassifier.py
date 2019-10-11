"""
## Version history:
2019:
   Original script by Jodie Ashford [ashfojsm], Aston University
"""

import pandas as pd
import FeatureSelection as fs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def build_classifier(training_path, testing_path, clf_output_file):
    """
    Builds and saves a trained random forest classifier.
    :param training_path: File path for the training matrix.
    :param testing_path: File path for the testing matrix.
    :param clf_output_file: Name of file to save the classifier to.
    """

    data = pd.read_csv(training_path)

    # Feature selection
    selected_features = fs.feature_selection(training_path)

    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()
    x_train = selected_data.drop('Label', axis=1)
    y_train = selected_data['Label']

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)

    # Testing
    testing_data = pd.read_csv(testing_path)
    selected_testing_data = testing_data[feature_names_plus_label].copy()
    x_test = selected_testing_data.drop('Label', axis=1)
    y_true = selected_testing_data['Label']
    y_predict = clf.predict(x_test)
    print("Accuracy of classifier = " + str(accuracy_score(y_true, y_predict)))

    return None


def classification_accuracy(clf, training_path, testing_path):
    """
    Returns the accuracy of a trained classifier given unseen data.
    :param clf: the classifier to test.
    :param training_path: File path for the training matrix.
    :param testing_path: File path for the testing matrix.
    :return: The accuracy of the given classifier
    """

    # Feature selection
    selected_features = fs.feature_selection(training_path)

    testing_data = pd.read_csv(testing_path)

    # Feature selection
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_testing_data = testing_data[feature_names_plus_label].copy()

    y_true = selected_testing_data['Label']
    x_test = selected_testing_data.drop('Label', axis=1)

    y_predict = clf.predict(x_test)
    return accuracy_score(y_true, y_predict)
