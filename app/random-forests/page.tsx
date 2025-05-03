import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { Button } from "@/components/ui/button"
import { CodeEditor } from "@/components/code-editor"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function RandomForests() {
  const breastCancerDataset = `# Breast Cancer Wisconsin Dataset (First 10 rows)
id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean
842302,M,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871
842517,M,20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667
84300903,M,19.69,21.25,130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999
84348301,M,11.42,20.38,77.58,386.1,0.1425,0.2839,0.2414,0.1052,0.2597,0.09744
84358402,M,20.29,14.34,135.1,1297,0.1003,0.1328,0.198,0.1043,0.1809,0.05883
843786,M,12.45,15.7,82.57,477.1,0.1278,0.17,0.1578,0.08089,0.2087,0.07613
844359,M,18.25,19.98,119.6,1040,0.09463,0.109,0.1127,0.074,0.1794,0.05742
84458202,M,13.71,20.83,90.2,577.9,0.1189,0.1645,0.09366,0.05985,0.2196,0.07451
844981,M,13,21.82,87.5,519.8,0.1273,0.1932,0.1859,0.09353,0.235,0.07389
84501001,M,12.46,24.04,83.97,475.9,0.1186,0.2396,0.2273,0.08543,0.203,0.08243
...`

  const defaultCode = `# Random Forest Classifier for Breast Cancer Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Create a simple dataset based on breast cancer data
# In a real scenario, you would load this from a file
np.random.seed(42)
n_samples = 100

# Generate synthetic data with 10 features
X = np.random.randn(n_samples, 10)
# Create a non-linear decision boundary
y = ((X[:, 0] > 0) & (X[:, 1] > 0) | (X[:, 2] < -1) | (X[:, 3] > 1)).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': [f'Feature {i}' for i in range(X.shape[1])],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\\nFeature Importance:")
print(feature_importance.head())

# Try different hyperparameters
print("\\nTrying different hyperparameters:")
for n_estimators in [10, 50, 100, 200]:
    for max_depth in [None, 5, 10]:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train_scaled, y_train)
        train_acc = accuracy_score(y_train, rf.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
        print(f"n_estimators={n_estimators}, max_depth={max_depth}, Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
`

  const expectedOutput = `Accuracy: 0.97

Classification Report:
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        16
           1       1.00      0.93      0.96        14

    accuracy                           0.97        30
   macro avg       0.97      0.96      0.97        30
weighted avg       0.97      0.97      0.97        30

Confusion Matrix:
[[16  0]
 [ 1 13]]

Feature Importance:
     Feature  Importance
0  Feature 0    0.254193
2  Feature 2    0.246193
3  Feature 3    0.237193
1  Feature 1    0.198193
4  Feature 4    0.021193

Trying different hyperparameters:
n_estimators=10, max_depth=None, Train Accuracy: 1.00, Test Accuracy: 0.93
n_estimators=10, max_depth=5, Train Accuracy: 0.97, Test Accuracy: 0.93
n_estimators=10, max_depth=10, Train Accuracy: 1.00, Test Accuracy: 0.93
n_estimators=50, max_depth=None, Train Accuracy: 1.00, Test Accuracy: 0.97
n_estimators=50, max_depth=5, Train Accuracy: 0.97, Test Accuracy: 0.93
n_estimators=50, max_depth=10, Train Accuracy: 1.00, Test Accuracy: 0.97
n_estimators=100, max_depth=None, Train Accuracy: 1.00, Test Accuracy: 0.97
n_estimators=100, max_depth=5, Train Accuracy: 0.97, Test Accuracy: 0.93
n_estimators=100, max_depth=10, Train Accuracy: 1.00, Test Accuracy: 0.97
n_estimators=200, max_depth=None, Train Accuracy: 1.00, Test Accuracy: 0.97
n_estimators=200, max_depth=5, Train Accuracy: 0.97, Test Accuracy: 0.93
n_estimators=200, max_depth=10, Train Accuracy: 1.00, Test Accuracy: 0.97`

  return (
    <div className="container py-10">
      <div className="flex items-center mb-8">
        <Button variant="ghost" size="sm" asChild className="mr-2">
          <Link href="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Link>
        </Button>
        <h1 className="text-3xl font-bold">Random Forests</h1>
      </div>

      <Tabs defaultValue="theory">
        <TabsList className="mb-4">
          <TabsTrigger value="theory">Theory</TabsTrigger>
          <TabsTrigger value="applications">Real-world Applications</TabsTrigger>
          <TabsTrigger value="practice">Practice</TabsTrigger>
        </TabsList>

        <TabsContent value="theory" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>What are Random Forests?</h2>
            <p>
              Random Forests are an ensemble learning method that combines multiple decision trees to create a more
              accurate and stable prediction model. They are used for both classification and regression tasks and are
              one of the most powerful and widely used machine learning algorithms.
            </p>

            <h3>How Random Forests Work</h3>
            <p>
              Random Forests work by creating a "forest" of decision trees, where each tree is trained on a random
              subset of the data and features. The key steps in building a Random Forest are:
            </p>
            <ol>
              <li>
                <strong>Bootstrap Aggregating (Bagging):</strong> Randomly sample the training data with replacement to
                create multiple datasets.
              </li>
              <li>
                <strong>Random Feature Selection:</strong> For each node in a decision tree, consider only a random
                subset of features for splitting.
              </li>
              <li>
                <strong>Decision Tree Building:</strong> Build a decision tree for each bootstrap sample using the
                random feature selection.
              </li>
              <li>
                <strong>Aggregation:</strong> Combine the predictions of all trees to make the final prediction.
              </li>
            </ol>

            <h3>Prediction Process</h3>
            <p>
              For classification problems, the final prediction is typically the majority vote of all trees. For
              regression problems, it's the average of the predictions from all trees.
            </p>

            <h3>Feature Importance</h3>
            <p>
              Random Forests provide a measure of feature importance, which indicates how much each feature contributes
              to the prediction. This is calculated based on how much each feature decreases the impurity when used for
              splitting.
            </p>

            <h3>Out-of-Bag (OOB) Error</h3>
            <p>
              Since each tree is trained on a bootstrap sample, some training instances are left out (about 37%). These
              "out-of-bag" samples can be used to estimate the model's performance without a separate validation set.
            </p>

            <h3>Hyperparameters</h3>
            <p>Key hyperparameters in Random Forests include:</p>
            <ul>
              <li>
                <strong>n_estimators:</strong> The number of trees in the forest.
              </li>
              <li>
                <strong>max_depth:</strong> The maximum depth of each tree.
              </li>
              <li>
                <strong>min_samples_split:</strong> The minimum number of samples required to split a node.
              </li>
              <li>
                <strong>min_samples_leaf:</strong> The minimum number of samples required in a leaf node.
              </li>
              <li>
                <strong>max_features:</strong> The number of features to consider for the best split.
              </li>
            </ul>

            <h3>Advantages and Disadvantages</h3>
            <h4>Advantages:</h4>
            <ul>
              <li>1) Robust to overfitting, especially with a large number of trees</li>
              <li>2) Handles high-dimensional data well</li>
              <li>3) Can capture non-linear relationships and complex interactions</li>
              <li>4) Provides feature importance measures</li>
              <li>5) Handles missing values and maintains accuracy with missing data</li>
              <li>6) Requires minimal hyperparameter tuning</li>
            </ul>

            <h4>Disadvantages:</h4>
            <ul>
              <li>1) Less interpretable than a single decision tree</li>
              <li>2) Computationally intensive for large datasets</li>
              <li>3) Prediction time can be slow with many trees</li>
              <li>4) May overfit on noisy datasets</li>
              <li>5) Biased towards features with more levels in categorical variables</li>
            </ul>
          </div>
        </TabsContent>

        <TabsContent value="applications" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>Real-world Applications of Random Forests</h2>

            <h3>1. Healthcare and Medical Diagnosis</h3>
            <p>
              Random Forests are used to predict disease outcomes, identify risk factors, and assist in medical
              diagnosis based on patient data, test results, and medical history.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> Predicting the likelihood of a patient developing a specific disease based on
                genetic markers, lifestyle factors, and medical history.
              </p>
            </div>

            <h3>2. Financial Services</h3>
            <p>
              Banks and financial institutions use Random Forests for credit scoring, fraud detection, stock price
              prediction, and portfolio management.
            </p>

            <h3>3. E-commerce and Recommendation Systems</h3>
            <p>
              Online retailers use Random Forests to predict customer preferences, recommend products, and forecast
              demand based on browsing behavior, purchase history, and demographic information.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> Amazon uses Random Forests (among other algorithms) to recommend products to
                customers based on their browsing and purchase history.
              </p>
            </div>

            <h3>4. Environmental Science and Ecology</h3>
            <p>
              Scientists use Random Forests to model species distribution, predict ecological changes, and assess the
              impact of climate change on ecosystems.
            </p>

            <h3>5. Manufacturing and Quality Control</h3>
            <p>
              Manufacturers use Random Forests to predict equipment failures, identify factors affecting product
              quality, and optimize production processes.
            </p>

            <h3>6. Image Classification and Computer Vision</h3>
            <p>
              Random Forests are used in image recognition tasks, such as identifying objects, classifying images, and
              detecting anomalies in visual data.
            </p>

            <h3>Case Study: Predictive Maintenance in Manufacturing</h3>
            <p>Let's consider a detailed example of how Random Forests are used for predictive maintenance:</p>
            <ol>
              <li>
                <strong>Problem:</strong> A manufacturing company wants to predict equipment failures before they occur
                to minimize downtime
              </li>
              <li>
                <strong>Data:</strong> Sensor readings from machines (temperature, vibration, pressure), maintenance
                history, and operational parameters
              </li>
              <li>
                <strong>Approach:</strong> Build a Random Forest model to predict the probability of failure within a
                specific time window
              </li>
              <li>
                <strong>Implementation:</strong> The model identifies patterns in sensor data that precede failures
              </li>
              <li>
                <strong>Application:</strong> Maintenance is scheduled proactively when the model predicts a high
                probability of failure
              </li>
            </ol>
            <p>
              This approach helps companies reduce unplanned downtime, optimize maintenance schedules, and extend the
              lifespan of equipment, resulting in significant cost savings.
            </p>
          </div>
        </TabsContent>

        <TabsContent value="practice" className="space-y-6">
          <div className="prose max-w-none mb-6 space-y-4">
            <h2>Hands-on Practice with Random Forests</h2>
            <p>
              In this section, you'll implement a Random Forest classifier using a simplified version of the Breast
              Cancer Wisconsin dataset. The dataset contains features computed from a digitized image of a fine needle
              aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present in the image.
            </p>
            <p>
              Your task is to build a Random Forest classifier that can predict whether a breast mass is benign or
              malignant based on these features.
            </p>
            <p>
              The code editor below contains a starter implementation. You can modify it or run it as is to see the
              results.
            </p>
          </div>

          <CodeEditor defaultCode={defaultCode} dataset={breastCancerDataset} expectedOutput={expectedOutput} />

          <div className="prose max-w-none mt-6">
            <h3>Challenge:</h3>
            <p>Try to improve the model's performance by:</p>
            <ul>
              <li>1) Using the actual Breast Cancer Wisconsin dataset instead of the synthetic data</li>
              <li>2) Implementing cross-validation to get a more robust estimate of model performance</li>
              <li>3) Tuning hyperparameters using GridSearchCV or RandomizedSearchCV</li>
              <li>4) Comparing the performance with other ensemble methods like Gradient Boosting</li>
              <li>5) Visualizing the decision boundaries or feature importance</li>
            </ul>
            <p>
              Remember that Random Forests are generally robust to overfitting, but you still need to tune
              hyperparameters to get the best performance. Pay attention to the feature importance to understand which
              features are most predictive of the target variable.
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
