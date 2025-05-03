import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { Button } from "@/components/ui/button"
import { CodeEditor } from "@/components/code-editor"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function NaiveBayes() {
  const irisDataset = `# Iris Dataset (First 10 rows)
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
...`

  const defaultCode = `# Naive Bayes Classifier for Iris Dataset
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Gaussian Naive Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Predict a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example of an iris setosa
prediction = model.predict(new_sample)
print(f"Predicted class: {iris.target_names[prediction[0]]}")
`

  const expectedOutput = `Accuracy: 0.96
Predicted class: setosa`

  return (
    <div className="container py-10">
      <div className="flex items-center mb-8">
        <Button variant="ghost" size="sm" asChild className="mr-2">
          <Link href="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Link>
        </Button>
        <h1 className="text-3xl font-bold">Naive Bayes</h1>
      </div>

      <Tabs defaultValue="theory">
        <TabsList className="mb-4">
          <TabsTrigger value="theory">Theory</TabsTrigger>
          <TabsTrigger value="applications">Real-world Applications</TabsTrigger>
          <TabsTrigger value="practice">Practice</TabsTrigger>
        </TabsList>

        <TabsContent value="theory" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>What is Naive Bayes?</h2>
            <p>
              Naive Bayes is a family of probabilistic algorithms based on applying Bayes' theorem with strong (naive)
              independence assumptions between the features. Despite its simplicity, Naive Bayes can often outperform
              more sophisticated classification methods.
            </p>

            <h3>Bayes' Theorem</h3>
            <p>
              At the heart of Naive Bayes is Bayes' theorem, which describes the probability of an event based on prior
              knowledge of conditions that might be related to the event:
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p className="font-mono">P(A|B) = P(B|A) * P(A) / P(B)</p>
            </div>
            <p>Where:</p>
            <ul>
              <li>P(A|B) is the posterior probability of class A given predictor B</li>
              <li>P(B|A) is the likelihood of predictor B given class A</li>
              <li>P(A) is the prior probability of class A</li>
              <li>P(B) is the prior probability of predictor B</li>
            </ul>

            <h3>The "Naive" Assumption</h3>
            <p>
              The algorithm is called "naive" because it assumes that all features are independent of each other, which
              is rarely true in real-world scenarios. Despite this oversimplification, the algorithm works surprisingly
              well in many complex situations.
            </p>

            <h3>Types of Naive Bayes Classifiers</h3>
            <ul>
              <li>
                <strong>Gaussian Naive Bayes:</strong> Assumes that features follow a normal distribution.
              </li>
              <li>
                <strong>Multinomial Naive Bayes:</strong> Suitable for discrete counts (e.g., word counts for text
                classification).
              </li>
              <li>
                <strong>Bernoulli Naive Bayes:</strong> Useful when features are binary (0s and 1s).
              </li>
            </ul>

            <h3>Mathematical Formulation</h3>
            <p>
              For a feature vector X = (x₁, x₂, ..., xₙ) and a class variable C, Naive Bayes wants to calculate P(C|X).
              Using Bayes' theorem:
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p className="font-mono">P(C|X) = P(X|C) * P(C) / P(X)</p>
            </div>
            <p>With the naive independence assumption, P(X|C) can be decomposed as:</p>
            <div className="bg-muted p-4 rounded-md">
              <p className="font-mono">P(X|C) = P(x₁|C) * P(x₂|C) * ... * P(xₙ|C)</p>
            </div>

            <h3>Advantages and Disadvantages</h3>
            <h4>Advantages:</h4>
            <ul>
              <li>1) Simple and easy to implement</li>
              <li>2) Works well with high-dimensional data</li>
              <li>3) Requires less training data</li>
              <li>4) Fast training and prediction</li>
              <li>5) Not sensitive to irrelevant features</li>
            </ul>

            <h4>Disadvantages:</h4>
            <ul>
              <li>1) Assumes independence of features (rarely true in real-world)</li>
              <li>
                2) If a categorical variable has a category not observed in training, model will assign zero probability
              </li>
              <li>3) Not ideal for regression problems</li>
            </ul>
          </div>
        </TabsContent>

        <TabsContent value="applications" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>Real-world Applications of Naive Bayes</h2>

            <h3>1. Text Classification and Spam Filtering</h3>
            <p>
              One of the most common applications of Naive Bayes is in text classification, particularly spam filtering.
              Email services use Naive Bayes classifiers to identify whether an incoming email is spam or not based on
              the words it contains.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> Gmail's spam filter uses Naive Bayes (among other techniques) to classify
                emails.
              </p>
            </div>

            <h3>2. Document Categorization</h3>
            <p>
              Naive Bayes is used to categorize documents into different topics or genres. News articles can be
              classified into categories like sports, politics, technology, etc.
            </p>

            <h3>3. Sentiment Analysis</h3>
            <p>
              Companies use Naive Bayes to analyze customer reviews and social media posts to determine whether the
              sentiment expressed is positive, negative, or neutral.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> A company might use sentiment analysis to track public opinion about a new
                product launch.
              </p>
            </div>

            <h3>4. Medical Diagnosis</h3>
            <p>
              Naive Bayes can be used to diagnose diseases based on symptoms. Given a set of symptoms, the algorithm can
              calculate the probability of various diseases.
            </p>

            <h3>5. Recommendation Systems</h3>
            <p>
              Simple recommendation systems can use Naive Bayes to predict user preferences based on their past behavior
              and similar users' preferences.
            </p>

            <h3>6. Real-time Prediction</h3>
            <p>
              Due to its computational efficiency, Naive Bayes is suitable for real-time prediction scenarios where
              decisions need to be made quickly.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> Credit card fraud detection systems that need to make instant decisions.
              </p>
            </div>

            <h3>Case Study: Document Classification</h3>
            <p>Let's consider a more detailed example of how Naive Bayes is used for document classification:</p>
            <ol>
              <li>
                <strong>Problem:</strong> Classify news articles into categories (Sports, Politics, Technology,
                Entertainment)
              </li>
              <li>
                <strong>Data:</strong> A corpus of labeled news articles
              </li>
              <li>
                <strong>Feature Extraction:</strong> Convert articles to feature vectors using techniques like
                bag-of-words or TF-IDF
              </li>
              <li>
                <strong>Training:</strong> Train a Multinomial Naive Bayes classifier on the feature vectors
              </li>
              <li>
                <strong>Prediction:</strong> For new articles, predict the most likely category
              </li>
            </ol>
            <p>
              This approach is widely used by news aggregators and content recommendation systems to organize and
              personalize content for users.
            </p>
          </div>
        </TabsContent>

        <TabsContent value="practice" className="space-y-6">
          <div className="prose max-w-none mb-6 space-y-4">
            <h2>Hands-on Practice with Naive Bayes</h2>
            <p>
              In this section, you'll implement a Naive Bayes classifier using the Iris dataset. The Iris dataset
              contains measurements of 150 iris flowers from three different species: setosa, versicolor, and virginica.
            </p>
            <p>
              Your task is to build a Gaussian Naive Bayes classifier that can predict the species of an iris flower
              based on its sepal length, sepal width, petal length, and petal width.
            </p>
            <p>
              The code editor below contains a starter implementation. You can modify it or run it as is to see the
              results.
            </p>
          </div>

          <CodeEditor defaultCode={defaultCode} dataset={irisDataset} expectedOutput={expectedOutput} />

          <div className="prose max-w-none mt-6">
            <h3>Challenge:</h3>
            <p>Try to improve the model's accuracy by:</p>
            <ul>
              <li>1) Using different train-test splits</li>
              <li>2) Normalizing the features</li>
              <li>3) Comparing with other Naive Bayes variants (MultinomialNB, BernoulliNB)</li>
            </ul>
            <p>
              Remember that Naive Bayes works best when the features are truly independent. In the Iris dataset, the
              features (petal and sepal measurements) are somewhat correlated, which might affect the performance of the
              Naive Bayes classifier.
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
