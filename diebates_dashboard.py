import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions

st.set_page_config(
    page_title="Decision Tree Dashboard",
    layout="wide"
)

st.title("Decision Tree Training Dashboard")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/social.csv"
    return pd.read_csv(url)

df = load_data()

X = df[["Age", "EstimatedSalary"]].values
y = df["Purchased"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.sidebar.header("Model Parameters")

criterion = st.sidebar.selectbox(
    "Criterion", ["gini", "entropy", "log_loss"]
)

max_depth = st.sidebar.number_input(
    "Max Depth",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
    format="%d"
)


min_samples_split = st.sidebar.slider(
    "Min Samples Split", 2, 50, 2
)

min_samples_leaf = st.sidebar.slider(
    "Min Samples Leaf", 1, 50, 1
)

max_features = st.sidebar.slider(
    "Max Feature", 1, 2, 1
)

max_leaf_node = st.sidebar.number_input(
    "Max Leaf Node",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
    format="%d"
)

ccp_alpha = st.sidebar.slider(
    "CCP Alpha", 0.0, 0.05, 0.0
)

dtc = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features = max_features,
    max_leaf_node = max_leaf_node,
    ccp_alpha=ccp_alpha,
    random_state=42
)

dtc.fit(X_train, y_train)

train_score = dtc.score(X_train, y_train)
test_score = dtc.score(X_test, y_test)

c1, c2 = st.columns(2)
c1.metric("Training Accuracy", f"{train_score*100:.2f}%")
c2.metric("Testing Accuracy", f"{test_score*100:.2f}%")

st.subheader("Decision Boundary")

fig, ax = plt.subplots(figsize=(10, 4))
plot_decision_regions(X, y, clf=dtc)
plt.xlabel("Age")
plt.ylabel("EstimatedSalary")
st.pyplot(fig)

st.subheader("Confusion Matrix")

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Decision Tree Structure")

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    dtc,
    feature_names=["Age", "EstimatedSalary"],
    class_names=["No", "Yes"],
    filled=True
)
st.pyplot(fig)
