import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions

st.set_page_config("Decision Tree Dashboard", layout="wide", page_icon="https://i.pinimg.com/1200x/3a/fb/36/3afb368ceb5d7e9bc7dd6e6a4affdd5f.jpg")
st.title("🌳 Decision Tree Training Dashboard")

#  Load Data 
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/social.csv"
    return pd.read_csv(url)

df = load_data()

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#  Sidebar Parameters 
st.sidebar.header("Model Parameters")

criterion = st.sidebar.selectbox(
    "Criterion", ["gini", "entropy", "log_loss"]
)

max_depth = st.sidebar.slider(
    "Max Depth", 1, 20, 4
)

min_samples_split = st.sidebar.slider(
    "Min Samples Split", 2, 50, 2
)

min_samples_leaf = st.sidebar.slider(
    "Min Samples Leaf", 1, 50, 1
)

max_features = st.sidebar.selectbox(
    "Max Features", [None, "sqrt", "log2"]
)

ccp_alpha = st.sidebar.slider(
    "CCP Alpha (Post Pruning)", 0.0, 0.05, 0.0
)

#  Train Model 
dtc = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    ccp_alpha=ccp_alpha,
    random_state=42
)

dtc.fit(X_train, y_train)

#  Scores 
train_score = dtc.score(X_train, y_train)
test_score = dtc.score(X_test, y_test)

col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{train_score*100:.2f}%")
col2.metric("Testing Accuracy", f"{test_score*100:.2f}%")

#  Decision Boundary 
st.subheader("Decision Boundary")

fig, ax = plt.subplots(figsize=(10, 4))
plot_decision_regions(X_scaled, y.to_numpy(), clf=dtc)
plt.xlabel("Age (Scaled)")
plt.ylabel("Estimated Salary (Scaled)")
st.pyplot(fig)

#  Confusion Matrix
st.subheader("Confusion Matrix")

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

#  Classification Report 
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

#  Tree Visualization 
st.subheader("Decision Tree Structure")

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dtc,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)
st.pyplot(fig)
