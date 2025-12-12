import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=42)

judge = LogisticRegression().fit(X, y)

# Create a background grid for the decision boundary
x_rng = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y_rng = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
xx, yy = np.meshgrid(x_rng, y_rng)
grid_preds = judge.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

artist = GaussianMixture(n_components=2).fit(X)
X_gen, _ = artist.sample(200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Discriminative (Boundary)
ax1.set_title("Discriminative: Finding the Boundary")
ax1.contourf(xx, yy, grid_preds, alpha=0.3, cmap='RdBu') # Draw the boundary
ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdBu')
ax1.set_label("Learns: P(y | x)")

# Plot 2: Generative (Creating)
ax2.set_title("Generative: Creating New Data")
ax2.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3, label="Original")
ax2.scatter(X_gen[:, 0], X_gen[:, 1], c='green', marker='x', label="Generated")
ax2.legend()
ax2.set_label("Learns: P(x)")

plt.tight_layout()

plt.savefig('model_comparison.png')
print("Success! Plot saved as 'model_comparison.png' in your current folder.")