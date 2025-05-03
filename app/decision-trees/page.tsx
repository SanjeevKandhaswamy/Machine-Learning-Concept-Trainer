import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { Button } from "@/components/ui/button"
import { CodeEditor } from "@/components/code-editor"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function DecisionTrees() {
  const titanicDataset = `# Titanic Dataset (First 10 rows)
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S
10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)",female,14,1,0,237736,30.0708,,C
...`

  const defaultCode = `# Decision Tree Classifier for Titanic Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Create a simple dataset from the Titanic data
# In a real scenario, you would load this from a file
data = {
    'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
    'Age': [22, 38, 26, 35, 35, np.nan, 54, 2, 27, 14],
    'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
    'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
    'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C'],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Prepare features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Encode categorical features
categorical_features = ['Sex', 'Embarked']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Create and train the decision tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\\nFeature Importance:")
print(feature_importance.head())

# Try with different max_depth values
print("\\nTrying different max_depth values:")
for depth in [2, 3, 4, 5]:
    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree_model.predict(X_train))
    test_acc = accuracy_score(y_test, tree_model.predict(X_test))
    print(f"max_depth={depth}, Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
`

  const expectedOutput = `Accuracy: 0.67

Feature Importance:
           Feature  Importance
0           Pclass    0.500000
3  Sex_male         0.333333
1             Age    0.166667
2           SibSp    0.000000
4           Parch    0.000000
5            Fare    0.000000
6  Embarked_Q       0.000000
7  Embarked_S       0.000000

Trying different max_depth values:
max_depth=2, Train Accuracy: 0.86, Test Accuracy: 0.67
max_depth=3, Train Accuracy: 1.00, Test Accuracy: 0.67
max_depth=4, Train Accuracy: 1.00, Test Accuracy: 0.67
max_depth=5, Train Accuracy: 1.00, Test Accuracy: 0.67`

  return (
    <div className="container py-10">
      <div className="flex items-center mb-8">
        <Button variant="ghost" size="sm" asChild className="mr-2">
          <Link href="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Link>
        </Button>
        <h1 className="text-3xl font-bold">Decision Trees</h1>
      </div>

      <Tabs defaultValue="theory">
        <TabsList className="mb-4">
          <TabsTrigger value="theory">Theory</TabsTrigger>
          <TabsTrigger value="applications">Real-world Applications</TabsTrigger>
          <TabsTrigger value="practice">Practice</TabsTrigger>
        </TabsList>

        <TabsContent value="theory" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>What is Decision Tree</h2>
            <p>
              Decision Trees are versatile machine learning algorithms used for both classification and regression
              tasks. They create a model that predicts the value of a target variable by learning simple decision rules
              inferred from the data features.
            </p>

            <h3>How Decision Trees Work</h3>
            <p>
              A Decision Tree works by recursively splitting the dataset into subsets based on the value of a feature.
              The splitting process continues until a stopping criterion is met, such as reaching a maximum depth or
              having a minimum number of samples in a leaf node.
            </p>
            <p>The key steps in building a decision tree are:</p>
            <ol>
              <li>1) Select the best feature to split the data</li>
              <li>2) Split the dataset based on the selected feature</li>
              <li>3) Recursively build subtrees for each split</li>
              <li>4) Stop when a stopping criterion is met</li>
            </ol>

            <h3>Splitting Criteria</h3>
            <p>The quality of a split is determined by various metrics:</p>
            <ul>
              <li>
                <strong>Gini Impurity:</strong> Measures the probability of misclassifying a randomly chosen element.
              </li>
              <li>
                <strong>Entropy:</strong> Measures the level of impurity or randomness in the data.
              </li>
              <li>
                <strong>Information Gain:</strong> The reduction in entropy after a dataset is split.
              </li>
              <li>
                <strong>Mean Squared Error:</strong> Used for regression trees to minimize prediction error.
              </li>
            </ul>

            <h3>Tree Pruning</h3>
            <p>
              Decision trees are prone to overfitting, especially when they grow deep. Pruning is a technique to reduce
              the size of a decision tree by removing sections that provide little predictive power.
            </p>
            <p>Types of pruning include:</p>
            <ul>
              <li>
                <strong>Pre-pruning:</strong> Stop the tree from growing before it perfectly classifies the training
                set.
              </li>
              <li>
                <strong>Post-pruning:</strong> Build the full tree, then remove branches that don't improve performance
                on a validation set.
              </li>
            </ul>

            <h3>Advantages and Disadvantages</h3>
            <h4>Advantages:</h4>
            <ul>
              <li>1) Easy to understand and interpret</li>
              <li>2) Requires little data preprocessing (no normalization needed)</li>
              <li>3) Can handle both numerical and categorical data</li>
              <li>4) Can handle multi-output problems</li>
              <li>5) Implicitly performs feature selection</li>
            </ul>

            <h4>Disadvantages:</h4>
            <ul>
              <li>1) Prone to overfitting, especially with deep trees</li>
              <li>2) Can create biased trees if some classes dominate</li>
              <li>3) Unstable: small variations in the data can result in a completely different tree</li>
              <li>4) May not capture complex relationships if the decision boundary is not axis-parallel</li>
              <li>5) Greedy algorithms used for tree construction may not find the globally optimal tree</li>
            </ul>
          </div>
        </TabsContent>

        <TabsContent value="applications" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>Real-world Applications of Decision Trees</h2>

            <h3>1. Credit Risk Assessment</h3>
            <p>
              Banks and financial institutions use decision trees to evaluate the creditworthiness of loan applicants.
              The algorithm can consider factors like income, credit history, employment status, and debt-to-income
              ratio to predict the likelihood of default.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> A bank might use a decision tree to automatically approve or reject loan
                applications based on risk factors.
              </p>
            </div>

            <h3>2. Medical Diagnosis</h3>
            <p>
              Decision trees help healthcare professionals diagnose diseases by considering symptoms, test results, and
              patient history. They provide a clear, interpretable path to a diagnosis, which is crucial in medical
              settings.
            </p>

            <h3>3. Customer Churn Prediction</h3>
            <p>
              Companies use decision trees to identify customers who are likely to cancel their services. By analyzing
              usage patterns, customer service interactions, and billing information, businesses can proactively address
              issues and retain customers.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> A telecom company might use decision trees to identify customers at risk of
                switching to competitors.
              </p>
            </div>

            <h3>4. Fraud Detection</h3>
            <p>
              Decision trees can detect fraudulent transactions by learning patterns from historical data. They can
              quickly process large volumes of transactions and flag suspicious activities for further investigation.
            </p>

            <h3>5. Recommendation Systems</h3>
            <p>
              E-commerce platforms and content providers use decision trees as part of their recommendation engines.
              They help categorize users and items to make personalized suggestions.
            </p>

            <h3>6. Quality Control in Manufacturing</h3>
            <p>
              Decision trees help identify factors that affect product quality in manufacturing processes. By analyzing
              production data, they can pinpoint the conditions that lead to defects.
            </p>

            <h3>Case Study: Customer Segmentation for Targeted Marketing</h3>
            <p>Let's consider a detailed example of how decision trees are used for customer segmentation:</p>
            <ol>
              <li>
                <strong>Problem:</strong> A retail company wants to segment customers for targeted marketing campaigns
              </li>
              <li>
                <strong>Data:</strong> Customer demographics, purchase history, browsing behavior, and response to
                previous campaigns
              </li>
              <li>
                <strong>Approach:</strong> Build a decision tree to classify customers into segments based on their
                likelihood to respond to different types of promotions
              </li>
              <li>
                <strong>Implementation:</strong> The decision tree might first split customers by age, then by purchase
                frequency, and then by average order value
              </li>
              <li>
                <strong>Application:</strong> Each segment receives tailored marketing messages and offers based on
                their characteristics
              </li>
            </ol>
            <p>
              This approach allows marketers to allocate resources efficiently and increase the return on investment for
              their campaigns.
            </p>
          </div>
        </TabsContent>

        <TabsContent value="practice" className="space-y-6">
          <div className="prose max-w-none mb-6 space-y-4">
            <h2>Hands-on Practice with Decision Trees</h2>
            <p>
              In this section, you'll implement a Decision Tree classifier using the Titanic dataset. The Titanic
              dataset contains information about passengers aboard the RMS Titanic, including whether they survived the
              disaster.
            </p>
            <p>
              Your task is to build a Decision Tree classifier that can predict whether a passenger survived based on
              features like passenger class, sex, age, and fare.
            </p>
            <p>
              The code editor below contains a starter implementation. You can modify it or run it as is to see the
              results.
            </p>
          </div>

          <CodeEditor defaultCode={defaultCode} dataset={titanicDataset} expectedOutput={expectedOutput} />

          <div className="prose max-w-none mt-6">
            <h3>Challenge:</h3>
            <p>Try to improve the model's performance by:</p>
            <ul>
              <li>
                Adding feature engineering (e.g., creating a 'FamilySize' feature by combining 'SibSp' and 'Parch')
              </li>
              <li>Experimenting with different hyperparameters (max_depth, min_samples_split, etc.)</li>
              <li>Implementing cross-validation to find the optimal hyperparameters</li>
              <li>Visualizing the decision tree to understand the decision-making process</li>
              <li>Comparing the performance with Random Forests (an ensemble of decision trees)</li>
            </ul>
            <p>
              Remember that decision trees are prone to overfitting, so it's important to control the tree's complexity
              through hyperparameters like max_depth and min_samples_leaf.
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
