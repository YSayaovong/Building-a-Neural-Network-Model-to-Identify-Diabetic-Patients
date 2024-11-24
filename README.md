# Diabetes Prediction Using MLP Classifier

This repository contains a project aimed at predicting whether a patient has diabetes based on their health attributes using a Multi-Layer Perceptron (MLP) Classifier. 

The dataset used includes information about patients' pregnancies, glucose levels, blood pressure, BMI, and other health-related features. The neural network model was designed, trained, and evaluated to classify patients as diabetic or non-diabetic.

---

## Dataset Details

The dataset contains 800 rows, where each row represents a patient described by the following attributes:
- **Number of Pregnancies**
- **Glucose Level**
- **Blood Pressure Level**
- **Skin Thickness**
- **BMI**
- **Age**
- **Diabetic Pedigree**
- **Insulin Level**

The target attribute is **Outcome**:
- `1` = Patient has diabetes.
- `0` = Patient does not have diabetes.

You can download the dataset [here](data/diabetes.csv).

---

## Methodology

### Steps Performed:
1. **Dataset Preprocessing**:
   - Normalized the feature columns for consistent scaling.
   - Split the dataset into training (75%) and testing (25%) sets.

2. **MLP Classifier Design**:
   - Built a neural network with three hidden layers, each containing a variable number of nodes.
   - Used the ReLU activation function and the Adam optimizer.
   - Trained the model over 500 iterations.

3. **Model Evaluation**:
   - Evaluated the model on the test set.
   - Measured accuracy for different numbers of nodes in the hidden layers (3 to 9 nodes per layer).

---

## Code Implementation

### Core Steps:
```python
# Import libraries
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('diabetes.csv')
print("Dataset columns:", df.columns.tolist())

# Normalize features
x_columns = df.columns.tolist()
x_columns.pop()  # Remove the target column
df[x_columns] = df[x_columns] / df[x_columns].max()

# Extract features (X) and labels (y)
X = df[x_columns].values
y = df[['Outcome']].values.ravel()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

# Build and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(6, 6, 6), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train, y_train)

# Test accuracy
predict_test = mlp.predict(X_test)
test_accuracy = accuracy_score(y_test, predict_test) * 100
print("Accuracy on test data = %.2f" % test_accuracy)
