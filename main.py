import os
import time
import random
import numpy as np

import util
import naiveBayes
import perceptron

# --------------------------------------------------------------------------------------
# Global Configuration Constants
# --------------------------------------------------------------------------------------
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
FACE_WIDTH = 60
FACE_HEIGHT = 70

TRAIN_PERCENTAGES = [round(i * 0.1, 1) for i in range(1, 11)]
MAX_ITERATIONS = 10
RANDOM_TRIALS = 5

# Configure NumPy print options
np.set_printoptions(linewidth=400)


# --------------------------------------------------------------------------------------
# Feature Extraction Functions
# --------------------------------------------------------------------------------------
def extract_digit_features(image: util.Picture):
    """
    Convert a single digit image into a Counter of binary features.
    Each pixel > 0 is treated as 1, otherwise 0.
    """
    features = util.Counter()
    for x in range(DIGIT_WIDTH):
        for y in range(DIGIT_HEIGHT):
            features[(x, y)] = 1 if image.getPixel(x, y) > 0 else 0
    return features


def extract_face_features(image: util.Picture):
    """
    Convert a single face image into a Counter of binary features.
    Each pixel > 0 is treated as 1, otherwise 0.
    """
    features = util.Counter()
    for x in range(FACE_WIDTH):
        for y in range(FACE_HEIGHT):
            features[(x, y)] = 1 if image.getPixel(x, y) > 0 else 0
    return features


# --------------------------------------------------------------------------------------
# Data Loading and Directory Setup
# --------------------------------------------------------------------------------------
def ensure_result_dirs_exist(data_type, classifier_type):
    """
    Ensure that directories for storing results exist.
    """
    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists(f'result/{data_type}'):
        os.mkdir(f'result/{data_type}')
    if not os.path.exists(f'result/{data_type}/{classifier_type}'):
        os.mkdir(f'result/{data_type}/{classifier_type}')


def get_result_paths(data_type, classifier_type):
    """
    Construct and return paths for result files.
    """
    stats_path = f"result/{data_type}/{classifier_type}/StatisticData.txt"
    weights_path = f"result/{data_type}/{classifier_type}/WeightsData.txt"
    weights_graph_path = f"result/{data_type}/{classifier_type}/WeightGraph.txt"
    return stats_path, weights_path, weights_graph_path


def load_and_extract_data(data_type, usage_fraction, test_indices, extract_digit_fn, extract_face_fn):
    """
    Load and extract features for training, validation, and testing depending on the data_type.
    Returns tuples of (train_data, train_labels, val_data, val_labels, test_data, test_labels).
    """

    if data_type == "digit":
        train_label_count = len(open("data/digitdata/traininglabels", "r").readlines())
        val_label_count = len(open("data/digitdata/validationlabels", "r").readlines())
        total_train_size = int(train_label_count * usage_fraction)

        if test_indices:
            test_size = len(test_indices)
        else:
            test_size = len(open("data/digitdata/testlabels", "r").readlines())

        print(f"Training Data Usage: {usage_fraction * 100:.1f}%")
        print(f"Training Set Size: {total_train_size}")
        print(f"Validation Set Size: {val_label_count}")
        print(f"Test Set Size: {test_size}")

        random_train_indices = random.sample(range(train_label_count), total_train_size)
        raw_train = util.loadDataFileRandomly("data/digitdata/trainingimages", random_train_indices,
                                              DIGIT_WIDTH, DIGIT_HEIGHT)
        train_labels = util.loadLabelFileRandomly("data/digitdata/traininglabels", random_train_indices)

        raw_val = util.loadDataFile("data/digitdata/validationimages", val_label_count,
                                    DIGIT_WIDTH, DIGIT_HEIGHT)
        val_labels = util.loadLabelFile("data/digitdata/validationlabels", val_label_count)

        if test_indices:
            raw_test = util.loadDataFileRandomly("data/digitdata/testimages", test_indices,
                                                 DIGIT_WIDTH, DIGIT_HEIGHT)
            test_labels = util.loadLabelFileRandomly("data/digitdata/testlabels", test_indices)
        else:
            raw_test = util.loadDataFile("data/digitdata/testimages", test_size,
                                         DIGIT_WIDTH, DIGIT_HEIGHT)
            test_labels = util.loadLabelFile("data/digitdata/testlabels", test_size)

        print("\tExtracting digit features...", end="")
        train_data = list(map(extract_digit_fn, raw_train))
        val_data = list(map(extract_digit_fn, raw_val))
        test_data = list(map(extract_digit_fn, raw_test))
        print("Done!")

    else:  # data_type == "face"
        train_label_path = f"data/facedata/facedatatrainlabels"
        val_label_path = f"data/facedata/facedatavalidationlabels"
        test_label_path = f"data/facedata/facedatatestlabels"

        train_label_count = len(open(train_label_path, "r").readlines())
        val_label_count = len(open(val_label_path, "r").readlines())
        test_label_count = len(open(test_label_path, "r").readlines())
        total_train_size = int(train_label_count * usage_fraction)

        if test_indices:
            test_size = len(test_indices)
        else:
            test_size = test_label_count

        print(f"Training Data Usage: {usage_fraction * 100:.1f}%")
        print(f"Training Set Size: {total_train_size}")
        print(f"Validation Set Size: {val_label_count}")
        print(f"Test Set Size: {test_size}")

        random_train_indices = random.sample(range(train_label_count), total_train_size)

        raw_train = util.loadDataFileRandomly("data/facedata/facedatatrain", random_train_indices,
                                              FACE_WIDTH, FACE_HEIGHT)
        train_labels = util.loadLabelFileRandomly(train_label_path, random_train_indices)

        raw_val = util.loadDataFile("data/facedata/facedatavalidation", val_label_count,
                                    FACE_WIDTH, FACE_HEIGHT)
        val_labels = util.loadLabelFile(val_label_path, val_label_count)

        if test_indices:
            raw_test = util.loadDataFileRandomly("data/facedata/facedatatest", test_indices,
                                                 FACE_WIDTH, FACE_HEIGHT)
            test_labels = util.loadLabelFileRandomly(test_label_path, test_indices)
        else:
            raw_test = util.loadDataFile("data/facedata/facedatatest", test_size, FACE_WIDTH, FACE_HEIGHT)
            test_labels = util.loadLabelFile(test_label_path, test_size)

        print("\tExtracting face features...", end="")
        train_data = list(map(extract_face_fn, raw_train))
        val_data = list(map(extract_face_fn, raw_val))
        test_data = list(map(extract_face_fn, raw_test))
        print("Done!")

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# --------------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    # You can switch classifier_type and data_type as needed
    classifier_type = "naiveBayes"
    #classifier_type = "perceptron"
    data_type = "face"
    legal_labels = range(2) if data_type == "face" else range(10)
    test_data_indices = []

    ensure_result_dirs_exist(data_type, classifier_type)
    result_stats_path, result_weights_path, result_weights_graph_path = get_result_paths(data_type, classifier_type)

    # Always train anew: do not load existing weights or check if they exist
    if classifier_type == "naiveBayes":
        classifier = naiveBayes.NaiveBayesClassifier(legal_labels)
        print("Classifier Type: Naive Bayes")
    else:
        classifier = perceptron.PerceptronClassifier(legal_labels, MAX_ITERATIONS)
        print("Classifier Type: Perceptron")
        # Removed any condition that would skip training.

    # Run experiments for different training set sizes
    for usage_fraction in TRAIN_PERCENTAGES:
        accuracy_list = []
        statistics_text = ""

        for trial in range(RANDOM_TRIALS):
            print(f"Random Trial: {trial}")
            # Load and extract data
            (train_data, train_labels,
             val_data, val_labels,
             test_data, test_labels) = load_and_extract_data(
                 data_type, usage_fraction, test_data_indices,
                 extract_digit_fn=extract_digit_features,
                 extract_face_fn=extract_face_features
             )

            statistics_text += f"Training Data Usage: {usage_fraction * 100:.1f}%\tRandom Trial: {trial}\n"

            # Always train classifier (no loading weights)
            print("\tTraining...")
            start = time.time()
            classifier.train(train_data, train_labels, val_data, val_labels)
            end = time.time()
            print("\tTraining completed!")
            train_time = end - start
            print(f"\tTraining Time: {train_time:.2f} s")
            statistics_text += f"\tTraining Time: {train_time:.2f} s\n"

            # Validate
            print("\tValidating...", end="")
            val_guesses = classifier.classify(val_data)
            val_correct = sum(val_guesses[i] == int(val_labels[i]) for i in range(len(val_labels)))
            print("Done!")
            val_accuracy = 100.0 * val_correct / len(val_labels)
            print(f"\t\t{val_correct} correct out of {len(val_labels)} ({val_accuracy:.2f}%).")
            statistics_text += f"\tValidation Accuracy: {val_correct} correct out of {len(val_labels)} ({val_accuracy:.2f}%)\n"

            # Test
            print("\tTesting...", end="")
            test_guesses = classifier.classify(test_data)
            test_correct = sum(test_guesses[i] == int(test_labels[i]) for i in range(len(test_labels)))
            print("Done!")
            test_accuracy = 100.0 * test_correct / len(test_labels)
            print(f"\t\t{test_correct} correct out of {len(test_labels)} ({test_accuracy:.2f}%).")
            statistics_text += f"\tTest Accuracy: {test_correct} correct out of {len(test_labels)} ({test_accuracy:.2f}%)\n"
            accuracy_list.append(round(test_correct / len(test_labels), 4))

            # If test_data_indices set, print predictions
            if test_data_indices:
                print(f"\t\tTest Data Predicted: {test_guesses}")
                print(f"\t\tTest Data Actual: {[int(lbl) for lbl in test_labels]}")

            # Save weights if perceptron after each trial (optional)
            if classifier_type == "perceptron":
                with open(result_weights_path, "a") as wfile:
                    wfile.write(f"{str(classifier.weights)}\n")

            # Generate weight visualization if perceptron
            if classifier_type == "perceptron":
                if data_type == "digit":
                    weight_pixels = ""
                    for lbl in classifier.legalLabels:
                        coords = classifier.findHighWeightFeatures(int(lbl), int(DIGIT_WIDTH * DIGIT_HEIGHT / 10))
                        weight_mat = np.zeros((DIGIT_WIDTH, DIGIT_HEIGHT))
                        for (x, y) in coords:
                            weight_mat[x][y] = 1
                        weight_pixels += f"Training Data Usage: {usage_fraction * 100:.1f}%\tRandom Trial: {trial}\tDigit: {lbl}\n"
                        weight_mat = np.rot90(weight_mat, 1)
                        for line in weight_mat:
                            for ch in line:
                                weight_pixels += "#" if int(ch) == 1 else " "
                            weight_pixels += "\n"
                    with open(result_weights_graph_path, "a") as gf:
                        gf.write(weight_pixels + "\n")

                elif data_type == "face":
                    coords = classifier.findHighWeightFeatures(int(classifier.legalLabels[1]),
                                                               int(FACE_WIDTH * FACE_HEIGHT / 8))
                    weight_mat = np.zeros((FACE_WIDTH, FACE_HEIGHT))
                    for (x, y) in coords:
                        weight_mat[x][y] = 1
                    weight_pixels = f"Training Data Usage: {usage_fraction * 100:.1f}%\tRandom Trial: {trial}\n"
                    weight_mat = np.rot90(weight_mat, 1)
                    for line in weight_mat:
                        for ch in line:
                            weight_pixels += "#" if int(ch) == 1 else " "
                        weight_pixels += "\n"
                    with open(result_weights_graph_path, "a") as gf:
                        gf.write(weight_pixels + "\n")

            print()

        # Compute statistics across all RANDOM_TRIALS
        accuracy_mean = np.mean(accuracy_list)
        accuracy_std = np.std(accuracy_list)
        print("Accuracy per trial:", accuracy_list)
        print(f"Accuracy Mean: {accuracy_mean * 100:.2f}%")
        statistics_text += f"Accuracy Mean: {accuracy_mean * 100:.2f}%\t"
        print(f"Accuracy Standard Deviation: {accuracy_std * 100:.2f}%")
        statistics_text += f"Accuracy Standard Deviation: {accuracy_std:.8f}\n"

        # Record statistics
        with open(result_stats_path, "a") as sf:
            sf.write(statistics_text)
        print()
