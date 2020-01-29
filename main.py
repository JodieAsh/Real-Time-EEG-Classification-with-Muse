"""
## Version history:
2019:
   Original script by Jodie Ashford [ashfojsm], Aston University
"""

import sys
import ModifiedRecord as MyRecord
import ModifiedStream as MyStream
from MockDataStream import MockStreamInlet
from EEGFeatureExtraction import generate_feature_vectors_from_samples
import FeatureSelection as fs
import numpy as np
import pandas as pd
import joblib

"""
Note: for streaming to work the BlueMuse application must already be open and the headset connected/online 
"""


def main(clf_path, training_file_path):
    # Feature selection
    selected_features = fs.feature_selection(training_file_path)

    # Load classifier
    clf = joblib.load(clf_path)

    # Begins LSL stream from a Muse with a given address with data sources determined by arguments
    MyStream.stream("00:55:da:b3:9a:2c")

    # Finds existing LSL stream & stars acquiring data
    inlet = MyRecord.start_stream()

    while True:
        """
        Generates a 2D array containing features as columns and time windows as rows and a list containing all feature names
        cols_to_ignore: -1 to remove last column from csv (remove Right AUX column)
        """
        # Feature extraction
        results, names = generate_feature_vectors_from_samples(MyRecord.record_numpy(2, inlet), 150, 1, cols_to_ignore=-1)
        # results, names = generate_feature_vectors_from_samples(MyRecord.record_numpy(2, MockStreamInlet()), 150, 1, cols_to_ignore=-1)
        data = pd.DataFrame(data=results, columns=names)

        # Feature selection
        selected_data = data[selected_features].copy()

        # Classification
        probability = clf.predict_proba(selected_data)
        for sample in probability:
            print(sample)
            if sample[0] >0.5:
                print("Relaxed")
                if sample[1] == 0:
                    odds = sys.maxsize
                else:
                    odds = round((sample[0] / sample[1]), 2)
                print("Estimated odds are : " + str(odds))
            elif sample[1] >0.5:
                print("Concentrating")
                if sample[0] == 0:
                    odds = sys.maxsize
                else:
                    odds = round((sample[1] / sample[0]), 2)
                print("Estimated odds are : " + str(odds))
            else:
                print("Unknown")


    return None


# TODO enter file path of the training matrix
training_file_path = r""

# TODO enter file path of the classifier
clf_path = r""

if __name__ == "__main__":
    main(clf_path, training_file_path)
