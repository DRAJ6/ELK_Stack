import logging
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Configure the logging module
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Logging important information
logging.info("Starting model training...")
logging.info(f"Number of training samples: {len(X_train)}")
logging.info(f"Number of testing samples: {len(X_test)}")

# Training the model and logging progress
model.fit(X_train, y_train)
logging.info("Model training completed.")

# Evaluate the model
score = model.score(X_test, y_test)
logging.info(f"Model accuracy on test data: {score:.2f}")

# Log model parameters
logging.info(f"Model coefficients: {model.coef_}")
logging.info(f"Model intercept: {model.intercept_}")
