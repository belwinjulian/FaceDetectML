import random
import util

class PerceptronClassifier:
    """
    A Perceptron classifier that maintains a weight vector for each legal label.
    The classifier iteratively adjusts weights based on misclassifications.
    """

    def __init__(self, legal_labels, max_iterations):
        """
        Initialize the Perceptron with given legal labels and max training iterations.
        
        Args:
            legal_labels (iterable): Possible labels for classification.
            max_iterations (int): Maximum number of passes over the training set.
        """
        self.legalLabels = legal_labels
        self.maxIteration = max_iterations
        self.type = "perceptron"
        self.weights = {lbl: util.Counter() for lbl in self.legalLabels}

    def setWeight(self, weights):
        """
        Manually set the perceptron weights.
        
        Args:
            weights (dict): A dictionary of label->Counter(weight_vector).
        """
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, training_data, training_labels, validation_data, validation_labels):
        """
        Train the Perceptron. Initialize weights, then run multiple passes through 
        the training data. On each pass, if an instance is misclassified, adjust weights.
        Track the best weights according to validation accuracy.
        """
        self._initialize_weights(training_data)

        best_weights = {}
        best_accuracy = 0.0

        for iteration in range(self.maxIteration):
            print(f"\t\tStarting iteration {iteration}...", end="")
            updated = self._train_one_iteration(training_data, training_labels)

            # Check performance on validation set
            guesses = self.classify(validation_data)
            correct_count = sum(guesses[i] == int(validation_labels[i]) for i in range(len(validation_labels)))
            accuracy = correct_count / len(validation_labels)

            # If this iteration yields a better accuracy, record the weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = {lbl: w.copy() for lbl, w in self.weights.items()}

            # If no updates were made, all training examples are classified correctly
            if not updated:
                print("\033[1;32mDone!\033[0m")
                break
            print("\033[1;32mDone!\033[0m")

        self.weights = best_weights

    def classify(self, data):
        """
        Classify each datum using the learned weights.
        
        Args:
            data (list[Counter]): Feature vectors for each instance.
        
        Returns:
            list: Predicted labels for each instance.
        """
        predictions = []
        for features in data:
            scores = util.Counter()
            for lbl in self.legalLabels:
                scores[lbl] = self.weights[lbl] * features + self.weights[lbl][0]
            predictions.append(scores.argMax())
        return predictions

    def findHighWeightFeatures(self, label, count):
        """
        Return a list of the features with the highest weights for the given label.
        
        Args:
            label: The label of interest.
            count (int): Number of top-weighted features to return.
        
        Returns:
            list: Coordinates (feature keys) of the top-weighted features.
        """
        # Sort features by weight value
        sorted_features = self.weights[label].sortedKeys()
        high_weight_features = []
        for f in sorted_features:
            # Ensure we're looking at actual feature keys (typically tuples), skip non-tuple keys like bias
            if isinstance(f, tuple):
                high_weight_features.append(f)
            if len(high_weight_features) == count:
                break
        return high_weight_features

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    def _initialize_weights(self, training_data):
        """
        Initialize weights and bias for each label and feature.
        Set bias to 0.1 and each feature weight to 0.5 as in the original code.
        """
        self.features = training_data[0].keys()
        for lbl in self.legalLabels:
            self.weights[lbl][0] = 0.1  # Treat index 0 as bias
            for feat in self.features:
                self.weights[lbl][feat] = 0.5

    def _train_one_iteration(self, training_data, training_labels):
        """
        Run one iteration over the training data. For each example,
        make a prediction and update weights if incorrect.
        
        Returns:
            bool: True if at least one update was performed, False otherwise.
        """
        updated = False
        learning_rate = 1.0

        for i, datum in enumerate(training_data):
            correct_label = int(training_labels[i])
            prediction = self._predict(datum)

            # If incorrect, update weights
            if prediction != correct_label:
                # Decrease weights for the predicted label
                self.weights[prediction] -= datum
                self.weights[prediction][0] -= learning_rate

                # Increase weights for the correct label
                self.weights[correct_label] += datum
                self.weights[correct_label][0] += learning_rate

                updated = True

        return updated

    def _predict(self, features):
        """
        Predict the label for a single feature vector.
        
        Args:
            features (Counter): Feature vector for a single instance.
        
        Returns:
            int: Predicted label.
        """
        scores = util.Counter()
        for lbl in self.legalLabels:
            scores[lbl] = self.weights[lbl] * features + self.weights[lbl][0]
        return scores.argMax()
