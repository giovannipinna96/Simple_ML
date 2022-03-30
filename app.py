import numpy as np
import streamlit as st
import Data
import plotUtils
from appUtils import get_classifier

from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Simple interactive dashboard for simple ML with standard dataset")
st.write("""
## Explore different classifier
you can play with parameter, classifier and dataset!!
""")
dataset_name = st.sidebar.selectbox("Select dataset", ("Iris", "Breast Cancer", "Wine dataset"))
st.write("""
## Information
""")
st.write(f"Dataset : {dataset_name}")
classifier_name = st.sidebar.selectbox("Select classifier",
                                       ("Perceptron", "Adeline SGD", "SVM", "KNN", "Random Forest"))
st.write(f"Classifier : {classifier_name}")

if dataset_name == "Iris":
    X_train, X_test, y_train, y_test = Data.load_split_iris(test_size=0.3, shuffle=False)
elif dataset_name == "Breast Cancer":
    X_train, X_test, y_train, y_test = Data.load_split_besast_cancer_wisconsin(test_size=0.3)
else:
    X_train, X_test, y_train, y_test = Data.load_split_wine(test_size=0.3)

X = np.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))

# Plot PCA of the original dataset
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

st.write(
    f"The dataset has {X.shape[0]} rows and {X.shape[1]} columns ")
st.write(f"The dataset has {len(np.unique(y))} classes")


# Set params and classifier
def add_paramiters_ui(clf_name):
    params = dict()
    if clf_name == "Random Forest":
        n_tree = st.sidebar.selectbox("n_tree", (5, 15, 25, 50, 100))
        max_depth = st.sidebar.slider("max_depth", 0.01, 10.0)
        params["max_depth"] = max_depth
        params["n_tree"] = n_tree
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 1.0)
        params["C"] = C
    elif clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    else:
        eta = st.sidebar.slider("eta", 0.01, 1.0)
        n_iter = st.sidebar.slider("n_iter", 1, 50)
        params["eta"] = eta
        params["n_iter"] = n_iter
    return params


params = add_paramiters_ui(classifier_name)
clf = get_classifier(classifier_name, params)

# Classification
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# calculate accuracy
acc = accuracy_score(y_test, y_pred)

# write data
st.write(f"Accuracy : {acc}")

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar()

st.pyplot(fig)

# Plot dataset with classifier regions
fig2 = plt.figure()
# setup marker generator and color map
markers = ('s', 'o', 'x', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])

# plot the decision surface
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))
Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

# plot class examples
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.8,
                c=colors[idx],
                marker=markers[idx],
                label=cl,
                edgecolor='black')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
# plt.title('Logistic Regression')
plt.legend(loc='upper left')
plt.tight_layout()
st.pyplot(fig2)
