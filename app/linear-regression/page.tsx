import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { Button } from "@/components/ui/button"
import { CodeEditor } from "@/components/code-editor"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function LinearRegression() {
  const bostonDataset = `# Boston Housing Dataset (First 10 rows)
CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV
0.00632,18.0,2.31,0,0.538,6.575,65.2,4.0900,1,296.0,15.3,396.90,4.98,24.0
0.02731,0.0,7.07,0,0.469,6.421,78.9,4.9671,2,242.0,17.8,396.90,9.14,21.6
0.02729,0.0,7.07,0,0.469,7.185,61.1,4.9671,2,242.0,17.8,392.83,4.03,34.7
0.03237,0.0,2.18,0,0.458,6.998,45.8,6.0622,3,222.0,18.7,394.63,2.94,33.4
0.06905,0.0,2.18,0,0.458,7.147,54.2,6.0622,3,222.0,18.7,396.90,5.33,36.2
0.02985,0.0,2.18,0,0.458,6.430,58.7,6.0622,3,222.0,18.7,394.12,5.21,28.7
0.08829,12.5,7.87,0,0.524,6.012,66.6,5.5605,5,311.0,15.2,395.60,12.43,22.9
0.14455,12.5,7.87,0,0.524,6.172,96.1,5.9505,5,311.0,15.2,396.90,19.15,27.1
0.21124,12.5,7.87,0,0.524,5.631,100.0,6.0821,5,311.0,15.2,386.63,29.93,16.5
0.17004,12.5,7.87,0,0.524,6.004,85.9,6.5921,5,311.0,15.2,386.71,17.10,18.9
...`

  const defaultCode = `# Linear Regression for Boston Housing Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Create a simple dataset based on Boston Housing data
# In a real scenario, you would load this from a file
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Single feature for simplicity
y = 2 * X.squeeze() + 3 + np.random.randn(100) * 1.5  # Linear relationship with noise

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficient (slope): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Multiple Linear Regression with more features
# Generate synthetic data with multiple features
X_multi = np.random.rand(100, 3) * 10
y_multi = 2 * X_multi[:, 0] + 1.5 * X_multi[:, 1] - 0.5 * X_multi[:, 2] + 3 + np.random.randn(100) * 2

# Split the data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_multi_scaled = scaler.fit_transform(X_train_multi)
X_test_multi_scaled = scaler.transform(X_test_multi)

# Create and train the multiple linear regression model
model_multi = LinearRegression()
model_multi.fit(X_train_multi_scaled, y_train_multi)

# Make predictions
y_pred_multi = model_multi.predict(X_test_multi_scaled)

# Evaluate the model
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)

print("\\nMultiple Linear Regression:")
print(f"Coefficients: {model_multi.coef_}")
print(f"Intercept: {model_multi.intercept_:.4f}")
print(f"Mean Squared Error: {mse_multi:.4f}")
print(f"R² Score: {r2_multi:.4f}")
`

  const expectedOutput = `Coefficient (slope): 2.0392
Intercept: 3.0825
Mean Squared Error: 2.2747
R² Score: 0.7241

Multiple Linear Regression:
Coefficients: [ 1.97646767  1.44242485 -0.51879795]
Intercept: 14.9841
Mean Squared Error: 3.9903
R² Score: 0.7553`

  return (
    <div className="container py-10">
      <div className="flex items-center mb-8">
        <Button variant="ghost" size="sm" asChild className="mr-2">
          <Link href="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Link>
        </Button>
        <h1 className="text-3xl font-bold">Linear Regression</h1>
      </div>

      <Tabs defaultValue="theory">
        <TabsList className="mb-4">
          <TabsTrigger value="theory">Theory</TabsTrigger>
          <TabsTrigger value="applications">Real-world Applications</TabsTrigger>
          <TabsTrigger value="practice">Practice</TabsTrigger>
        </TabsList>

        <TabsContent value="theory" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>What is Linear Regression?</h2>
            <p>
              Linear Regression is one of the most fundamental and widely used algorithms in machine learning and
              statistics. It models the relationship between a dependent variable (target) and one or more independent
              variables (features) by fitting a linear equation to the observed data.
            </p>

            <h3>Simple Linear Regression</h3>
            <p>
              Simple Linear Regression involves only one independent variable and can be represented by the equation:
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p className="font-mono">y = β₀ + β₁x + ε</p>
            </div>
            <p>Where:</p>
            <ul>
              <li>y is the dependent variable (target)</li>
              <li>x is the independent variable (feature)</li>
              <li>β₀ is the y-intercept (the value of y when x = 0)</li>
              <li>β₁ is the slope (the change in y for a unit change in x)</li>
              <li>ε is the error term (the difference between the predicted and actual values)</li>
            </ul>

            <h3>Multiple Linear Regression</h3>
            <p>Multiple Linear Regression extends the simple linear model to include multiple independent variables:</p>
            <div className="bg-muted p-4 rounded-md">
              <p className="font-mono">y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε</p>
            </div>
            <p>
              This allows the model to capture more complex relationships in the data by considering multiple factors
              that might influence the target variable.
            </p>

            <h3>Estimating the Parameters</h3>
            <p>
              The goal of linear regression is to find the values of the coefficients (β₀, β₁, ..., βₙ) that minimize
              the sum of squared differences between the predicted and actual values. This method is known as Ordinary
              Least Squares (OLS).
            </p>
            <p>The objective function to minimize is:</p>
            <div className="bg-muted p-4 rounded-md">
              <p className="font-mono">min Σ(yᵢ - ŷᵢ)²</p>
            </div>
            <p>Where:</p>
            <ul>
              <li>yᵢ is the actual value</li>
              <li>ŷᵢ is the predicted value</li>
            </ul>

            <h3>Assumptions of Linear Regression</h3>
            <p>Linear regression makes several assumptions about the data:</p>
            <ul>
              <li>
                <strong>Linearity:</strong> The relationship between the independent and dependent variables is linear.
              </li>
              <li>
                <strong>Independence:</strong> The observations are independent of each other.
              </li>
              <li>
                <strong>Homoscedasticity:</strong> The variance of the error terms is constant across all levels of the
                independent variables.
              </li>
              <li>
                <strong>Normality:</strong> The error terms are normally distributed.
              </li>
              <li>
                <strong>No Multicollinearity:</strong> The independent variables are not highly correlated with each
                other (for multiple linear regression).
              </li>
            </ul>

            <h3>Evaluating the Model</h3>
            <p>Common metrics to evaluate linear regression models include:</p>
            <ul>
              <li>
                <strong>Mean Squared Error (MSE):</strong> The average of the squared differences between predicted and
                actual values.
              </li>
              <li>
                <strong>Root Mean Squared Error (RMSE):</strong> The square root of MSE, which provides an error measure
                in the same units as the target variable.
              </li>
              <li>
                <strong>R-squared (R²):</strong> The proportion of the variance in the dependent variable that is
                predictable from the independent variables.
              </li>
              <li>
                <strong>Adjusted R-squared:</strong> A modified version of R² that adjusts for the number of predictors
                in the model.
              </li>
            </ul>

            <h3>Advantages and Disadvantages</h3>
            <h4>Advantages:</h4>
            <ul>
              <li>1) Simple to understand and implement</li>
              <li>2) Computationally efficient</li>
              <li>3) Provides clear interpretability of the relationship between features and target</li>
              <li>4) Works well when the relationship between variables is approximately linear</li>
              <li>5) Serves as a good baseline for more complex models</li>
            </ul>

            <h4>Disadvantages:</h4>
            <ul>
              <li>1) Assumes a linear relationship between variables</li>
              <li>2) Sensitive to outliers</li>
              <li>3) Cannot capture complex, non-linear relationships</li>
              <li>4) Assumes independence of features (can be problematic with multicollinearity)</li>
              <li>5) May overfit with too many features relative to the number of observations</li>
            </ul>
          </div>
        </TabsContent>

        <TabsContent value="applications" className="space-y-6">
          <div className="prose max-w-none space-y-4">
            <h2>Real-world Applications of Linear Regression</h2>

            <h3>1. Real Estate Price Prediction</h3>
            <p>
              Linear regression is widely used to predict house prices based on features like square footage, number of
              bedrooms, location, age of the building, and other factors.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> Zillow's Zestimate uses linear regression (among other techniques) to estimate
                home values.
              </p>
            </div>

            <h3>2. Sales Forecasting</h3>
            <p>
              Businesses use linear regression to predict future sales based on historical data, seasonal trends,
              marketing expenditure, and economic indicators.
            </p>

            <h3>3. Risk Assessment in Insurance</h3>
            <p>
              Insurance companies use linear regression to assess risk and determine premium rates based on factors like
              age, health indicators, driving history, and property characteristics.
            </p>
            <div className="bg-muted p-4 rounded-md">
              <p>
                <strong>Example:</strong> Auto insurers might use linear regression to predict the likelihood of
                accidents based on driver characteristics.
              </p>
            </div>

            <h3>4. Salary Prediction</h3>
            <p>
              HR departments and job platforms use linear regression to estimate salary ranges based on factors like
              years of experience, education level, skills, and location.
            </p>

            <h3>5. Medical Research</h3>
            <p>
              Researchers use linear regression to understand the relationship between variables in medical studies,
              such as the effect of dosage on patient response or the correlation between risk factors and disease
              prevalence.
            </p>

            <h3>6. Environmental Science</h3>
            <p>
              Scientists use linear regression to model relationships between environmental variables, such as the
              impact of temperature on crop yields or the relationship between pollution levels and health outcomes.
            </p>

            <h3>Case Study: Retail Demand Forecasting</h3>
            <p>Let's consider a detailed example of how linear regression is used for demand forecasting in retail:</p>
            <ol>
              <li>
                <strong>Problem:</strong> A retail chain wants to predict demand for products to optimize inventory
                levels
              </li>
              <li>
                <strong>Data:</strong> Historical sales data, seasonal indicators, promotional information, price
                changes, and competitor activities
              </li>
              <li>
                <strong>Approach:</strong> Build a multiple linear regression model to predict sales volume for each
                product category
              </li>
              <li>
                <strong>Implementation:</strong> The model might include variables like previous sales, seasonality
                indices, price elasticity, and marketing spend
              </li>
              <li>
                <strong>Application:</strong> Use the predictions to adjust inventory levels, plan promotions, and
                optimize pricing strategies
              </li>
            </ol>
            <p>
              This approach helps retailers reduce stockouts and overstock situations, leading to improved customer
              satisfaction and reduced inventory costs.
            </p>
          </div>
        </TabsContent>

        <TabsContent value="practice" className="space-y-6">
          <div className="prose max-w-none mb-6 space-y-4">
            <h2>Hands-on Practice with Linear Regression</h2>
            <p>
              In this section, you'll implement Linear Regression using a simplified version of the Boston Housing
              dataset. The Boston Housing dataset contains information about various houses in Boston and their prices.
            </p>
            <p>
              Your task is to build a Linear Regression model that can predict house prices based on features like crime
              rate, number of rooms, accessibility to highways, and more.
            </p>
            <p>
              The code editor below contains a starter implementation. You can modify it or run it as is to see the
              results.
            </p>
          </div>

          <CodeEditor defaultCode={defaultCode} dataset={bostonDataset} expectedOutput={expectedOutput} />

          <div className="prose max-w-none mt-6">
            <h3>Challenge:</h3>
            <p>Try to improve the model's performance by:</p>
            <ul>
              <li>1)Using polynomial features to capture non-linear relationships</li>
              <li>2)Implementing regularization techniques like Ridge or Lasso regression to prevent overfitting</li>
              <li>3)Performing feature selection to identify the most important predictors</li>
              <li>4)Visualizing the relationship between features and the target variable</li>
              <li>5)Comparing the performance with more advanced regression techniques</li>
            </ul>
            <p>
              Remember to check the assumptions of linear regression (linearity, independence, homoscedasticity,
              normality) to ensure the model is appropriate for the data.
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
