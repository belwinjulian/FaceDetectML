import math
import util

class NaiveBayesClassifier:
    """
    A Naive Bayes classifier for image-based classification tasks.
    Features are binary (0 or 1), and we apply Laplace smoothing.
    """

    def __init__(self, legal_labels):
        """
        Initialize the Naive Bayes classifier.
        
        Args:
            legal_labels (iterable): A collection of possible label values.
        """
        self.legalLabels = legal_labels
        self.type = "naivebayes"
        self.k = 1  # Smoothing parameter
        # self.automaticTuning = False  # automatic tuning, use this flag.

    def setSmoothing(self, k):
        """
        Set the smoothing parameter k (Laplace smoothing).
        Do not modify this method.
        """
        self.k = k

    def train(self, training_data, training_labels, validation_data, validation_labels):
        """
        Train the classifier and select the best smoothing parameter.
        """
        # Extract all unique features from the training data
        self.features = list(set(f for datum in training_data for f in datum.keys()))

        # Candidate smoothing values
        # If automatic tuning were implemented, we'd pick from kgrid.
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]

        self.trainAndTune(training_data, training_labels, validation_data, validation_labels, kgrid)

    def trainAndTune(self, training_data, training_labels, validation_data, validation_labels, kgrid):
        """
        Train with various smoothing parameters and pick the best one based on validation performance.

        Args:
            training_data (list[Counter]): Feature counts for training.
            training_labels (list): Labels for training instances.
            validation_data (list[Counter]): Feature counts for validation.
            validation_labels (list): Labels for validation instances.
            kgrid (list[float]): Candidate smoothing (k) values.
        """
        # Compute counts from the training data
        label_feature_counts, label_counts, data_count = self._compute_training_statistics(training_data, training_labels)

        # Compute priors from the validation set labels (or alternatively from training set)
        label_priors = self._compute_label_priors(validation_labels)

        # Select the best k
        best_k, best_accuracy = self._choose_best_k(validation_data, validation_labels, 
                                                    label_feature_counts, label_counts, 
                                                    label_priors, kgrid)

        # Store model parameters
        self.setSmoothing(best_k)
        self.result = label_feature_counts
        self.numberOfLabel = label_counts
        self.pOfLabel = label_priors

    def classify(self, test_data):
        """
        Classify the test data based on learned probabilities.
        Do not modify this method.
        """
        guesses = []
        self.posteriors = []
        for datum in test_data:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Compute log-joint probabilities over all labels for the given datum.
        """
        log_joint = util.Counter()

        for lbl in self.legalLabels:
            log_prob = math.log(self.pOfLabel[lbl])
            # Incorporate feature likelihoods
            for f_key in datum:
                count_for_label = self.result[lbl][f_key]
                label_total = self.numberOfLabel[lbl]

                # If feature is 0, use one probability; if 1, use the complementary probability
                if datum[f_key] == 0:
                    prob = (count_for_label + self.k) / (label_total + 2 * self.k)
                else:
                    prob = ((label_total - count_for_label) + self.k) / (label_total + 2 * self.k)
                log_prob += math.log(prob)

            log_joint[lbl] = log_prob

        return log_joint

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    def _compute_training_statistics(self, training_data, training_labels):
        """
        Compute label-specific feature counts from the training data. 
        We track how often a given feature is zero for each label.

        Returns:
            label_feature_counts (list[Counter]): For each label, a Counter of feature counts.
            label_counts (list[int]): Counts of how many training samples per label.
            data_count (int): Total number of training samples.
        """
        data_count = len(training_labels)
        # Initialize counters
        label_feature_counts = []
        label_counts = []

        # Prepare structure based on legal labels
        for _ in self.legalLabels:
            label_feature_counts.append(util.Counter())
            label_counts.append(0)

        # Initialize all feature counts to zero based on the first datum's features
        for lbl in self.legalLabels:
            for f_key in training_data[0]:
                label_feature_counts[lbl][f_key] = 0

        # Fill counts
        for i, lbl_str in enumerate(training_labels):
            lbl = int(lbl_str)
            label_counts[lbl] += 1
            datum = training_data[i]
            for f_key in datum:
                # Count how many times this feature is zero for this label
                if datum[f_key] == 0:
                    label_feature_counts[lbl][f_key] += 1

        return label_feature_counts, label_counts, data_count

    def _compute_label_priors(self, validation_labels):
        """
        Compute label prior probabilities from the validation labels.

        Args:
            validation_labels (list): Labels from the validation set.

        Returns:
            list[float]: Prior probabilities per label index.
        """
        total = len(validation_labels)
        priors = []
        for lbl in self.legalLabels:
            count = sum(int(vl) == lbl for vl in validation_labels)
            priors.append(count / total)
        return priors

    def _choose_best_k(self, validation_data, validation_labels, 
                       label_feature_counts, label_counts, label_priors, kgrid):
        """
        Test each smoothing parameter k and choose the best one based on validation accuracy.

        Args:
            validation_data (list[Counter]): Validation features.
            validation_labels (list): Validation labels.
            label_feature_counts (list[Counter]): Feature counts per label from training.
            label_counts (list[int]): Sample counts per label from training.
            label_priors (list[float]): Label prior probabilities.
            kgrid (list[float]): Candidate smoothing parameters.

        Returns:
            (float, float): Best k and its corresponding accuracy.
        """
        best_k = None
        best_accuracy = -1.0
        val_size = len(validation_labels)

        for k_val in kgrid:
            correct_count = 0
            for i, datum in enumerate(validation_data):
                real_label = int(validation_labels[i])
                predicted_label = self._predict_with_k(datum, label_feature_counts, label_counts, label_priors, k_val)
                if predicted_label == real_label:
                    correct_count += 1

            accuracy = (correct_count / val_size) * 100
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k_val

        return best_k, best_accuracy

    def _predict_with_k(self, datum, label_feature_counts, label_counts, label_priors, k_val):
        """
        Predict the label for a single datum given a candidate smoothing parameter k.

        Args:
            datum (Counter): Features of a single instance.
            label_feature_counts (list[Counter]): Feature-zero counts per label.
            label_counts (list[int]): Counts of samples per label.
            label_priors (list[float]): Prior probabilities per label.
            k_val (float): Smoothing parameter to test.

        Returns:
            int: Predicted label index.
        """
        log_probs = []
        for lbl in self.legalLabels:
            # Start with log prior
            log_p = math.log(label_priors[lbl])
            lbl_total = label_counts[lbl]

            # Add feature likelihoods
            for f_key in datum:
                zero_count = label_feature_counts[lbl][f_key]
                if datum[f_key] == 0:
                    p_feature = (zero_count + k_val) / (lbl_total + 2 * k_val)
                else:
                    p_feature = ((lbl_total - zero_count) + k_val) / (lbl_total + 2 * k_val)
                log_p += math.log(p_feature)

            log_probs.append(log_p)

        return log_probs.index(max(log_probs))
