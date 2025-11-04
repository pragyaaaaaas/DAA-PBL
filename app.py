pip install streamlit scikit-learn pandas numpy matplotlib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---------------------------------------------
# 🧠 Helper Function: Knapsack Optimization
# ---------------------------------------------
def knapsack(values, weights, capacity):
    n = len(values)
    dp = np.zeros((n + 1, capacity + 1))

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]

    return selected[::-1]

# ---------------------------------------------
# 🎯 Streamlit App Layout
# ---------------------------------------------
st.set_page_config(page_title="Feature Selection using Knapsack Optimization", layout="wide")
st.title("🎯 Feature Selection for ML Models – Knapsack-based Optimization")
st.markdown("This app selects the most important features using **Knapsack Optimization** to maximize model performance while minimizing computational cost.")

# ---------------------------------------------
# 📥 Data Loading
# ---------------------------------------------
st.sidebar.header("⚙️ Configuration")

dataset_option = st.sidebar.selectbox("Select Dataset", ["Breast Cancer (Sklearn)", "Upload CSV"])

if dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("✅ Dataset Loaded Successfully!")
        target_col = st.sidebar.selectbox("Select Target Column", data.columns)
        X = data.drop(columns=[target_col])
        y = data[target_col]
    else:
        st.warning("Please upload a dataset to proceed.")
        st.stop()
else:
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target

# ---------------------------------------------
# 🧮 Model Training - Baseline
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
base_accuracy = accuracy_score(y_test, model.predict(X_test))

importances = model.feature_importances_
costs = np.random.randint(1, 10, size=len(importances))

st.subheader("📊 Feature Importance (Baseline)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(X.columns, importances, color='skyblue')
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance from RandomForest")
st.pyplot(fig)

# ---------------------------------------------
# 🧮 Knapsack Optimization
# ---------------------------------------------
capacity = st.sidebar.slider("Select Feature Cost Capacity", 10, 100, 50)
selected_features_idx = knapsack(importances, costs, capacity)
selected_features = X.columns[selected_features_idx]

st.success(f"✅ Selected {len(selected_features)} features using Knapsack Optimization")
st.write("**Selected Features:**", list(selected_features))

# ---------------------------------------------
# 🔍 Model with Selected Features
# ---------------------------------------------
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

model_sel = RandomForestClassifier(random_state=42)
model_sel.fit(X_train_sel, y_train)
y_pred_sel = model_sel.predict(X_test_sel)

opt_accuracy = accuracy_score(y_test, y_pred_sel)

# ---------------------------------------------
# 📈 Result Comparison
# ---------------------------------------------
st.subheader("📈 Model Performance Comparison")

col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy (All Features)", f"{base_accuracy*100:.2f}%")
with col2:
    st.metric("Accuracy (Selected Features)", f"{opt_accuracy*100:.2f}%")

st.markdown("### 🔧 Performance Gain/Loss:")
if opt_accuracy >= base_accuracy:
    st.success(f"Model performance improved or maintained with fewer features ✅")
else:
    st.warning(f"Model performance slightly reduced but feature count and computation decreased ⚙️")

# ---------------------------------------------
# 📉 Visualization
# ---------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(["All Features", "Selected Features"], [len(X.columns), len(selected_features)], color=['lightcoral', 'lightgreen'])
ax2.set_title("Feature Reduction")
ax2.set_ylabel("Number of Features")
st.pyplot(fig2)

st.markdown("---")
st.markdown("💡 *Knapsack-based feature selection helps optimize performance and computation by mimicking the 0/1 Knapsack decision process.*")
