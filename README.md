# FaceDetectML
A machine learning project focused on face detection using grayscale images of size 60x70 pixels. This project implements and compares two classifiers—Perceptron and Naive Bayes—trained on binary pixel features.

# Face Detection Using Perceptron and Naive Bayes Classifiers

## Introduction
This project aims to classify whether a given image contains a face. We consider grayscale images of size 60x70 pixels and use a simple feature extraction method that converts each pixel into a binary feature (1 if pixel intensity > 0, else 0). Using these binary features, two classifiers are trained and evaluated:

- **Perceptron**: A linear classifier that iteratively updates weights whenever it misclassifies a training example.
- **Naive Bayes**: A probabilistic classifier that assumes feature independence given the label to estimate `P(label|features)`.

The project investigates how the amount of training data affects the performance of these classifiers. Training set usage varies from 10% to 100% in increments of 10%, with multiple trials (typically 5) for each fraction. Performance metrics include:

- **Training Time**: Time taken to train the classifier.
- **Validation and Test Accuracy**: Correctly classified instances on independent validation and test sets.
- **Mean and Standard Deviation of Test Accuracy**: Assessment of consistency and robustness across multiple runs.

---

## Implemented Algorithms

### 1. Perceptron Classifier
The Perceptron is a simple linear model with a weight vector for each label. Training involves:

1. Predicting the label of a training sample using the dot product of the feature vector and the weight vector.
2. Updating weights only when a misclassification occurs:
   - Increase weights for the correct label.
   - Decrease weights for the incorrect label.
3. Repeating until a predefined number of iterations or no more errors remain.

This simplicity makes the Perceptron effective for linearly separable data with sufficient examples.

### 2. Naive Bayes Classifier
Naive Bayes uses Bayes' theorem and assumes conditional independence of features given the label. Key steps:

1. Estimate probabilities for each feature given the label.
2. For classification, calculate `P(label) * Π(P(feature|label))` for all features and select the label with the maximum posterior probability.
3. Apply Laplace smoothing to handle zero probabilities.

Naive Bayes is computationally efficient and performs well even with relatively small datasets.

---

## Results

### 1. Perceptron Results
- **10% Training Data**: Test accuracy ~70-75%, with high variability due to sensitivity to the subset.
- **50% Training Data**: Mean test accuracy ~83%, with reduced variability.
- **100% Training Data**: Mean test accuracy ~88-89%, with low variability.
- **Training Time**: Increases with dataset size, from a few seconds at 10% to 17-25 seconds at 100%.

### 2. Naive Bayes Results
- **10% Training Data**: Test accuracy ~74-78%, often better than Perceptron at this stage.
- **30-40% Training Data**: Test accuracy approaches mid-80s.
- **100% Training Data**: Test accuracy stabilizes at ~88-89%, with lower standard deviations than Perceptron.
- **Training Time**: Efficient across all data sizes (~14-20 seconds).

### 3. Comparative Observations
- **Performance**:
  - Naive Bayes performs better with less data.
  - Perceptron shows significant gains with larger datasets.
- **Accuracy**:
  - Both converge to ~88-90% at 100% training data.
  - Naive Bayes has lower variance at smaller fractions.
- **Training Time**:
  - Both scale with dataset size but remain practical for the project’s scope.

---

## Lessons Learned

1. **More Data, Better Performance**:
   - Accuracy increases with training data size for both models.
   - Highlights the importance of labeled data for model improvement.
2. **Data Efficiency**:
   - Naive Bayes achieves good results with less data, while Perceptron excels with more.
3. **Stability**:
   - Larger training sets reduce test accuracy variance, ensuring consistent performance.
4. **Model Simplicity**:
   - Binary pixel features, despite their simplicity, deliver high accuracies with sufficient data.
5. **Trade-Offs**:
   - Perceptron requires more data for high accuracy, while Naive Bayes is resilient with smaller datasets.

---

## Conclusion and Future Directions
### Key Findings:
- Both classifiers achieve ~70-75% accuracy with 10% data and ~88-90% with 100%.
- Naive Bayes stabilizes faster with less data, while Perceptron benefits significantly from more data.
- Increased training data reduces accuracy variance, making performance more reliable.

### Future Directions:
1. **Feature Enhancements**:
   - Explore advanced techniques like edge detection or gradient-based features.
2. **Algorithm Exploration**:
   - Test more complex models for improved performance.
3. **Semi-Supervised Learning**:
   - Investigate methods requiring fewer labeled samples.
4. **Active Learning**:
   - Explore strategies for selective data labeling.

This project reinforces key principles of machine learning: the importance of training set size, algorithmic data efficiency, and the potential for improved performance through better feature engineering and data utilization.
