import numpy as np
import streamlit as st
import Data
from appUtils import get_classifier

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

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

standardize = st.sidebar.radio("Do you want standardize the data?", ("Yes", "No"))
st.write(f"Are data standardize? : {standardize}")

if dataset_name == "Iris":
    X, y = Data.load_iris_dataset()
elif dataset_name == "Breast Cancer":
    X, y = Data.load_besast_cancer_wisconsin_dataset()
else:
    X, y = Data.load_wine_dataset()

if standardize == "Yes":
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

# Plot PCA of the original dataset
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)
X_projected_train, X_projected_test, y_projected_train, y_projected_test = train_test_split(X_projected, y,
                                                                                            test_size=0.3)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
X_combined = np.column_stack((x1, x2))

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
clf.fit(X_projected_train, y_projected_train)
y_pred = clf.predict(X_projected_test)

# calculate accuracy
acc = accuracy_score(y_projected_test, y_pred)

# write data
st.write(f"Accuracy : {acc}")
st.write("""## Graphs : """)

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.title("Graph of the data on which the PCA has been applied (dimensionality reduction)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar()
plt.tight_layout()
st.pyplot(fig)

if acc == 0:
    st.write(""" ## Attention : """)
    st.write(
        "The graph is not displayed because the accuracy is zero. Unfortunately, the classifier is not suitable for "
        "classifying the dataset or the parameters entered are poor. Try with another classifier or with other "
        "parameters or trying to standardize the data !! :-)")
else:
    # Plot dataset with classifier regions
    fig2 = plt.figure()
    plot_decision_regions(X_combined, y, clf)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title(f'Regions identified on the reduced dimensionality dataset by the classifier: {classifier_name}')
    plt.legend(loc='upper left')
    plt.tight_layout()
    st.pyplot(fig2)
