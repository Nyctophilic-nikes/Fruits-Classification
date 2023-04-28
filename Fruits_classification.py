from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load the train.csv dataset
train_df = pd.read_csv("C:\\Users\\HP\\SML\\Sample data\\train.csv")
X = train_df.drop(['category'], axis=1)
y = train_df["category"] 

# Apply Local Outlier Factor to remove outliers from the dataset
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(X)
X = X[y_pred == 1]
y = y[y_pred == 1]

# Apply StandardScaler to scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce the dimensionality of the features
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Create an MLPClassifier with default parameters
mlp = MLPClassifier()

# Create a bagging classifier with 10 base estimators
bagging = BaggingClassifier(base_estimator=mlp, n_estimators=10)

# Fit the model on the training data
bagging.fit(X_pca, y)

# Predict the target variable using cross-validation
y_pred_cv = bagging.predict(X_pca)
accuracy_cv = cross_val_score(bagging, X_pca, y, cv=5)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y, y_pred_cv)
print('Accuracy on training data:', accuracy_train) 

# Calculate the mean cross-validation accuracy
mean_cv_accuracy = accuracy_cv.mean()
print('Mean cross-validation accuracy:', mean_cv_accuracy)

test_data = pd.read_csv("C:\\Users\\HP\\SML\\Sample data\\test.csv")

# Apply StandardScaler to scale the test data
X_test_scaled = scaler.transform(test_data)

# Apply PCA to reduce the dimensionality of the test features
X_test_pca = pca.transform(X_test_scaled)

# Predict the target variable on the test data
y_pred = bagging.predict(X_test_pca)

# Add the predicted target variable to the test dataset
test_data["category"] = y_pred

# Save the test dataset with predicted target variable
test_data.to_csv("C:\\Users\\HP\\SML\\Sample data\\submission.csv", index=False)
