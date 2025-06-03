import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Iris.csv")

df.drop("Id", axis=1, inplace=True)

X = df.drop("Species", axis=1)
y = df["Species"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"K = {k}, Accuracy = {acc:.2f}")

k_final = 5
knn = KNeighborsClassifier(n_neighbors=k_final)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

def plot_decision_boundary(X, y, model, title):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X = X[:, :2]
    model.fit(X, y_encoded)

    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    for idx, cls in enumerate(np.unique(y_encoded)):
        plt.scatter(X[y_encoded == cls][:, 0], X[y_encoded == cls][:, 1],
                    c=[cmap_bold(idx)], label=le.inverse_transform([cls])[0], edgecolor='k', s=50) # Use inverse_transform for legend

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.show()

X_2d = X_scaled[:, :2]
plot_decision_boundary(X_2d, y, KNeighborsClassifier(n_neighbors=k_final), f"KNN (K={k_final}) Decision Boundary")