import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { Button } from "@/components/ui/button"
import { CodeEditor } from "@/components/code-editor"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function KNN() {
  const wineDataset = `# Wine Dataset (First 10 rows)
alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280/od315_of_diluted_wines,proline,class
14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065,1
13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050,1
13.16,2.36,2.67,18.6,101,2.8,3.24,0.3,2.81,5.68,1.03,3.17,1185,1
14.37,1.95,2.5,16.8,113,3.85,3.49,0.24,2.18,7.8,0.86,3.45,1480,1
13.24,2.59,2.87,21,118,2.8,2.69,0.39,1.82,4.32,1.04,2.93,735,1
14.2,1.76,2.45,15.2,112,3.27,3.39,0.34,1.97,6.75,1.05,2.85,1450,1
14.39,1.87,2.45,14.6,96,2.5,2.52,0.3,1.98,5.25,1.02,3.58,1290,1
14.06,2.15,2.61,17.6,121,2.6,2.51,0.31,1.25,5.05,1.06,3.58,1295,1
14.83,1.64,2.17,14,97,2.8,2.98,0.29,1.98,5.2,1.08,2.85,1045,1
13.86,1.35,2.27,16,98,2.98,3.15,0.22,1.85,7.22,1.01,3.55,1045,1
...`

  const defaultCode = `# K-Nearest Neighbors for Wine Dataset
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Scale the features (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create a KNN classifier with k=5
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Try different values of k
k_values = [3, 5, 7, 9, 11]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print(f"k={k}, Accuracy: {accuracy:.2f}")
`

  const expectedOutput = `Accuracy: 0.94
k=3, Accuracy: 0.96
k=5, Accuracy: 0.94
k=7, Accuracy: 0.94
k=9, Accuracy: 0.91
k=11, Accuracy: 0.91`

  return (
    <div className="container py-10">
      <div className="flex items-center mb-8">
        <Button variant="ghost" size="sm" asChild className="mr-2">
          <Link href="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Link>
        </Button>
        <h1 className="text-3xl font-bold">K-Nearest Neighbors</h1>
      </div>

      <Tabs defaultValue="theory">
        <TabsList className="mb-4">
          <TabsTrigger value="theory">Theory</TabsTrigger>
          <TabsTrigger value="applications">Real-world Applications</TabsTrigger>
          <TabsTrigger value="practice">Practice</TabsTrigger>
        </TabsList>

        <TabsContent value="theory" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>What is K-Nearest Neighbors?</h2>
            <p>
              K-Nearest Neighbors (KNN) is a simple, versatile, and non-parametric supervised learning algorithm used
              for both classification and regression. The algorithm makes predictions based on the similarity (or
              distance) between data points.
            </p>

            <h3>How KNN Works</h3>
            <p>The core idea behind KNN is remarkably straightforward:</p>
            <ol>
              <li>1) Calculate the distance between the query instance and all training instances</li>
              <li>2) Sort the distances and determine the K nearest neighbors</li>
              <li>3) For classification: take a majority vote of the K neighbors' classes</li>
              <li>4) For regression: take the average of the K neighbors' values</li>
            </ol>

            <h3>Distance Metrics</h3>
            <p>
              KNN relies on distance metrics to measure similarity between instances. Common distance metrics include:
            </p>
            <ul>
              <li>
                <strong>Euclidean Distance:</strong> The straight-line distance between two points in Euclidean space.
              </li>
              <li>
                <strong>Manhattan Distance:</strong> The sum of absolute differences between coordinates.
              </li>
              <li>
                <strong>Minkowski Distance:</strong> A generalization of Euclidean and Manhattan distance.
              </li>
              <li>
                <strong>Hamming Distance:</strong> Used for categorical variables, counts the number of positions at
                which corresponding symbols differ.
              </li>
            </ul>

            <h3>Choosing the Value of K</h3>
            <p>The choice of K is crucial in KNN:</p>
            <ul>
              <li>1) Small K: More sensitive to noise, but can capture fine-grained patterns</li>
              <li>2) Large K: More robust to noise, but might miss important patterns</li>
              <li>3) K should be an odd number (for binary classification) to avoid ties</li>
              <li>4) Common approach: Use cross-validation to find the optimal K</li>
            </ul>

            <h3>Feature Scaling</h3>
            <p>
              Since KNN uses distance calculations, features with larger scales can dominate the distance calculation.
              Therefore, feature scaling (normalization or standardization) is essential for KNN to work effectively.
            </p>

            <h3>Advantages and Disadvantages</h3>
            <h4>Advantages:</h4>
            <ul>
              <li>1) Simple to understand and implement</li>
              <li>2) No training phase (lazy learning)</li>
              <li>3) Naturally handles multi-class problems</li>
              <li>4) Can be used for both classification and regression</li>
              <li>5) Makes no assumptions about the underlying data distribution</li>
            </ul>

            <h4>Disadvantages:</h4>
            <ul>
              <li>1) Computationally expensive for large datasets</li>
              <li>2) Sensitive to irrelevant features and the curse of dimensionality</li>
              <li>3) Requires feature scaling</li>
              <li>4) Memory-intensive as it stores all training data</li>
              <li>5) Prediction can be slow for large training sets</li>
            </ul>
          </div>
        </TabsContent>

        <TabsContent value="applications" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>Real-world Applications of KNN</h2>

            <h3>1. Recommendation Systems</h3>
            <p>
              KNN is widely used in recommendation systems to suggest products, movies, or content based on user
              preferences. The algorithm identifies users with similar tastes (nearest neighbors) and recommends items
              they've liked.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> Netflix might recommend movies based on what similar users have enjoyed.
              </p>
            </div>

            <h3>2. Credit Scoring</h3>
            <p>
              Financial institutions use KNN to assess credit risk by comparing a new applicant's profile with existing
              customers who have similar characteristics.
            </p>

            <h3>3. Pattern Recognition</h3>
            <p>
              KNN is effective for various pattern recognition tasks, including image recognition, handwriting
              detection, and speech recognition.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> Optical character recognition (OCR) systems can use KNN to identify characters
                based on their features.
              </p>
            </div>

            <h3>4. Medical Diagnosis</h3>
            <p>
              In healthcare, KNN can help diagnose diseases by comparing a patient's symptoms and test results with
              those of previously diagnosed patients.
            </p>

            <h3>5. Anomaly Detection</h3>
            <p>
              KNN can identify outliers or anomalies by finding data points that are distant from their nearest
              neighbors, which is useful for fraud detection and network security.
            </p>

            <h3>6. Gene Expression Analysis</h3>
            <p>
              In bioinformatics, KNN helps classify genes based on their expression levels and identify genes with
              similar functions.
            </p>

            <h3>Case Study: Customer Segmentation</h3>
            <p>Let's consider a detailed example of how KNN is used for customer segmentation:</p>
            <ol>
              <li>
                <strong>Problem:</strong> A retail company wants to segment its customers for targeted marketing
                campaigns
              </li>
              <li>
                <strong>Data:</strong> Customer attributes like age, income, purchase history, website activity
              </li>
              <li>
                <strong>Preprocessing:</strong> Scale features to ensure equal contribution to distance calculations
              </li>
              <li>
                <strong>Analysis:</strong> Use KNN to group similar customers together
              </li>
              <li>
                <strong>Application:</strong> Develop personalized marketing strategies for each segment
              </li>
            </ol>
            <p>
              This approach allows businesses to understand their customer base better and create more effective
              marketing strategies tailored to specific customer segments.
            </p>
          </div>
        </TabsContent>

        <TabsContent value="practice" className="space-y-6">
          <div className="prose max-w-none mb-6 space-y-4">
            <h2>Hands-on Practice with KNN</h2>
            <p>
              In this section, you'll implement a K-Nearest Neighbors classifier using the Wine dataset. The Wine
              dataset contains the results of a chemical analysis of wines grown in the same region in Italy but derived
              from three different cultivars.
            </p>
            <p>
              Your task is to build a KNN classifier that can predict the cultivar (class) of a wine based on its
              chemical attributes.
            </p>
            <p>
              The code editor below contains a starter implementation. You can modify it or run it as is to see the
              results.
            </p>
          </div>

          <CodeEditor defaultCode={defaultCode} dataset={wineDataset} expectedOutput={expectedOutput} />

          <div className="prose max-w-none mt-6">
            <h3>Challenge:</h3>
            <p>Try to improve the model's performance by:</p>
            <ul>
              <li>1) Experimenting with different distance metrics (e.g., Manhattan, Minkowski)</li>
              <li>2) Implementing feature selection to identify the most important attributes</li>
              <li>3) Using cross-validation to find the optimal value of K</li>
              <li>4) Comparing KNN with other classification algorithms</li>
            </ul>
            <p>
              Remember that KNN is sensitive to the scale of the features, so standardization or normalization is
              crucial for good performance. Also, consider the computational cost when working with large datasets.
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
