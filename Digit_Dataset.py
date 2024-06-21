import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = load_digits()
X = data.data
y = data.target
# Select two classes 0 and 1) for binary classification
binary_mask = (y == 0) | (y == 1)
X_binary = X[binary_mask]
y_binary = y[binary_mask]
# Apply PCA to reduce dimensionality to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_binary)
# Logistic Regression for binary classification
clf = LogisticRegression(random_state=42)
clf.fit(X_binary, y_binary)
predictions = clf.predict(X_binary)
# Calculate accuracy
accuracy = accuracy_score(y_binary, predictions)
print(f"Accuracy: {accuracy}")
#before classification
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='green', s=50, alpha=0.8)
plt.title('True Labels Before Classification')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
# after classification
plt.subplot(1, 2, 2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=predictions, cmap='viridis', s=50, alpha=0.8)
plt.title('Predicted Labels After Classification')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.tight_layout()
plt.show()