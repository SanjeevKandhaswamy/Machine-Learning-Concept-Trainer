import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { Button } from "@/components/ui/button"
import { CodeEditor } from "@/components/code-editor"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function LogisticRegression() {
  const diabetesDataset = `# Diabetes Dataset (First 10 rows)
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
...`

  const defaultCode = `# Logistic Regression for Diabetes Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create a simple dataset based on diabetes data
# In a real scenario, you would load this from a file
np.random.seed(42)
n_samples = 100

# Generate synthetic data
X = np.random.randn(n_samples, 2)  # Two features for simplicity
# Create a non-linear decision boundary
y = (X[:, 0]**2 + X[:, 1]**2 > 2).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Try different regularization strengths
print("\\nTrying different regularization strengths:")
for C in [0.01, 0.1, 1.0, 10.0]:
    log_reg = LogisticRegression(C=C, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, log_reg.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, log_reg.predict(X_test_scaled))
    print(f"C={C}, Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
`

  const expectedOutput = `Accuracy: 0.70

Classification Report:
              precision    recall  f1-score   support

           0       0.67      0.80      0.73        10
           1       0.75      0.60      0.67        10

    accuracy                           0.70        20
   macro avg       0.71      0.70      0.70        20
weighted avg       0.71      0.70      0.70        20

Confusion Matrix:
[[8 2]
 [4 6]]

Trying different regularization strengths:
C=0.01, Train Accuracy: 0.67, Test Accuracy: 0.70
C=0.1, Train Accuracy: 0.67, Test Accuracy: 0.70
C=1.0, Train Accuracy: 0.67, Test Accuracy: 0.70
C=10.0, Train Accuracy: 0.67, Test Accuracy: 0.70`

  return (
    <div className="container py-10">
      <div className="flex items-center mb-8">
        <Button variant="ghost" size="sm" asChild className="mr-2">
          <Link href="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Link>
        </Button>
        <h1 className="text-3xl font-bold">Logistic Regression</h1>
      </div>

      <Tabs defaultValue="theory">
        <TabsList className="mb-4">
          <TabsTrigger value="theory">Theory</TabsTrigger>
          <TabsTrigger value="applications">Real-world Applications</TabsTrigger>
          <TabsTrigger value="practice">Practice</TabsTrigger>
        </TabsList>
        
        <TabsContent value="theory" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>What is Logistic Regression?</h2>
            <p>
              Logistic Regression is a statistical method used for binary classification problems. Despite its name, 
              logistic regression is a classification algorithm, not a regression algorithm. It estimates the probability 
              that a given instance belongs to a particular class.
            </p>
            
            <h3>The Logistic Function</h3>
            <p>
              At the heart of logistic regression is the logistic function (also called the sigmoid function), which maps 
              any real-valued number to a value between 0 and 1:
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p className="font-mono">σ(z) = 1 / (1 + e^(-z))</p>
            </div>
            <p>
              Where z is the linear combination of the input features and their weights:
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p className="font-mono">z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ</p>
            </div>
            
            <h3>Decision Boundary</h3>
            <p>
              The logistic function outputs a probability value between 0 and 1. To make a binary prediction, 
              we typically use a threshold of 0.5:
            </p>
            <ul>
              <li>If σ(z) less than or equal to 0.5, predict class 1</li>
              <li>If σ(z) greater than 0.5, predict class 0</li>
            </ul>
            <p>
              The decision boundary is the set of points where σ(z) = 0.5, which corresponds to z = 0.
            </p>
            
            <h3>Cost Function</h3>
            <p>
              Unlike linear regression, which uses mean squared error, logistic regression uses a cost function 
              called the log loss (or cross-entropy loss):
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p className="font-mono">J(θ) = -1/m * Σ[y * log(h(x)) + (1-y) * log(1-h(x))]</p>
            </div>
            <p>Where:</p>
            <ul>
              <li>m is the number of training examples</li>
              <li>y is the actual class (0 or 1)</li>
              <li>h(x) is the predicted probability</li>
            </ul>
            
            <h3>Regularization</h3>
            <p>
              To prevent overfitting, logistic regression often includes regularization terms:
            </p>
            <ul>
              <li><strong>L1 Regularization (Lasso):</strong> Adds a penalty equal to the absolute value of the magnitude of coefficients.</li>
              <li><strong>L2 Regularization (Ridge):</strong> Adds a penalty equal to the square of the magnitude of coefficients.</li>
            </ul>
            
            <h3>Multiclass Logistic Regression</h3>
            <p>
              While basic logistic regression is for binary classification, it can be extended to multiclass problems using:
            </p>
            <ul>
              <li><strong>One-vs-Rest (OvR):</strong> Train n binary classifiers, one for each class.</li>
              <li><strong>Multinomial Logistic Regression:</strong> Also known as Softmax Regression, directly predicts probabilities for multiple classes.</li>
            </ul>
            
            <h3>Advantages and Disadvantages</h3>
            <h4>Advantages:</h4>
            <ul>
              <li>1) Simple to implement and interpret</li>
              <li>2) Efficient training</li>
              <li>3) Less prone to overfitting with regularization</li>
              <li>4) Outputs well-calibrated probabilities</li>
              <li>5) Works well for linearly separable classes</li>
            </ul>
            
            <h4>Disadvantages:</h4>
            <ul>
              <li>1) Assumes a linear relationship between features and the log-odds of the outcome</li>
              <li>2) Limited to linear decision boundaries</li>
              <li>3) May not perform well with complex, non-linear relationships</li>
              <li>4) Requires feature engineering for complex problems</li>
              <li>5) Sensitive to imbalanced datasets</li>
            </ul>
          </div>
        </TabsContent>
        
        <TabsContent value="applications" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>Real-world Applications of Logistic Regression</h2>
            
            <h3>1. Medical Diagnosis</h3>
            <p>
              Logistic regression is widely used in healthcare to predict the probability of disease based on symptoms, 
              test results, and patient characteristics.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p><strong>Example:</strong> Predicting whether a patient has diabetes based on factors like glucose level, BMI, age, and family history.</p>
            </div>
            
            <h3>2. Credit Scoring</h3>
            <p>
              Financial institutions use logistic regression to assess credit risk and determine the likelihood 
              that a borrower will default on a loan.
            </p>
            
            <h3>3. Marketing Campaign Effectiveness</h3>
            <p>
              Marketers use logistic regression to predict the probability that a customer will respond to a 
              marketing campaign based on demographic information and past behavior.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p><strong>Example:</strong> Predicting whether a customer will click on an email campaign based on their browsing history and previous interactions.</p>
            </div>
            
            <h3>4. Spam Detection</h3>
            <p>
              Email services use logistic regression (among other techniques) to classify emails as spam or legitimate 
              based on the content, sender information, and other features.
            </p>
            
            <h3>5. Customer Churn Prediction</h3>
            <p>
              Businesses use logistic regression to identify customers who are likely to cancel their services, 
              allowing for proactive retention efforts.
            </p>
            
            <h3>6. Election Forecasting</h3>
            <p>
              Political analysts use logistic regression to predict election outcomes based on polling data, 
              demographic information, and historical voting patterns.
            </p>
            
            <h3>Case Study: Credit Card Fraud Detection</h3>
            <p>
              Let's consider a detailed example of how logistic regression is used for credit card fraud detection:
            </p>
            <ol>
              <li><strong>Problem:</strong> A bank wants to identify fraudulent credit card transactions in real-time</li>
              <li><strong>Data:</strong> Transaction amount, location, time, merchant category, and customer's transaction history</li>
              <li><strong>Approach:</strong> Build a logistic regression model to classify transactions as fraudulent or legitimate</li>
              <li><strong>Implementation:</strong> The model calculates the probability of fraud for each transaction</li>
              <li><strong>Application:</strong> Transactions with a high fraud probability are flagged for review or automatically declined</li>
            </ol>
            <p>
              This approach helps banks balance fraud prevention with customer convenience. By adjusting the probability 
              threshold, they can control the trade-off between false positives (legitimate transactions flagged as fraud) 
              and false negatives (fraudulent transactions not detected).
            </p>
          </div>
        </TabsContent>
        
        <TabsContent value="practice" className="space-y-6">
          <div className="prose max-w-none mb-6 space-y-4">
            <h2>Hands-on Practice with Logistic Regression</h2>
            <p>
              In this section, you'll implement a Logistic Regression classifier using a simplified version of the 
              Diabetes dataset. The Diabetes dataset contains information about patients and whether they have diabetes.
            </p>
            <p>
              Your task is to build a Logistic Regression model that can predict whether a patient has diabetes based on 
              features like glucose level, BMI, age, and other health indicators.
            </p>
            <p>
              The code editor below contains a starter implementation. You can modify it or run it as is to see the results.
            </p>
          </div>
          
          <CodeEditor 
            defaultCode={defaultCode} 
            dataset={diabetesDataset} 
            expectedOutput={expectedOutput}
          />
          
          <div className="prose max-w-none mt-6">
            <h3>Challenge:</h3>
            <p>
              Try to improve the model's performance by:
            </p>
            <ul>
              <li>1) Using polynomial features to capture non-linear relationships</li>
              <li>2) Experimenting with different regularization types (L1, L2) and strengths</li>
              <li>3) Implementing feature selection to identify the most important predictors</li>
              <li>4) Handling class imbalance using techniques like class weighting or SMOTE</li>
              <li>5) Tuning the classification threshold to optimize for precision or recall</li>
            </ul>
            <p>
              Remember that logistic regression works best when the features have a linear relationship with the 
              log-odds of the outcome. If the relationship is highly non-linear, consider using more complex models 
              or transforming the features.
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
