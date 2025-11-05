import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Smart Knapsack Selector", layout="wide")

# ------------------------------
# 0/1 Knapsack DP
# ------------------------------
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = np.zeros((n+1, capacity+1), dtype=int)

    for i in range(1, n+1):
        for c in range(capacity+1):
            if weights[i-1] <= c:
                dp[i][c] = max(dp[i-1][c], values[i-1] + dp[i-1][c-weights[i-1]])
            else:
                dp[i][c] = dp[i-1][c]

    picks = np.zeros(n, dtype=int)
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i-1][c]:
            picks[i-1] = 1
            c -= weights[i-1]

    return picks, dp[n][capacity]

st.title("🤖 Smart Knapsack Feature Selector")
st.subheader("Knapsack + ML Model | Optimization Demo")

uploaded = st.file_uploader("Upload `knapsack_5_items.csv`", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df["Weights"] = df["Weights"].apply(lambda x: ast.literal_eval(x))
    df["Prices"] = df["Prices"].apply(lambda x: ast.literal_eval(x))
    df["Best picks"] = df["Best picks"].apply(lambda x: np.array(ast.literal_eval(x)))

    st.success("✅ Dataset loaded successfully")
    st.dataframe(df.head())

    # ---------------- ML Model ----------------
    st.info("Training ML model to predict optimal picks...")

    X = []
    y = []

    for _, row in df.iterrows():
        X.append(row["Weights"] + row["Prices"] + [row["Capacity"]])
        y.append("".join(map(str, row["Best picks"])))

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    ml_acc = accuracy_score(y_test, pred)

    st.write(f"### 🤖 ML Model Accuracy: **{ml_acc*100:.2f}%**")

    if st.button("Run Knapsack Evaluation"):
        correct = 0
        predictions = []

        for idx, row in df.iterrows():
            pred_picks, pred_val = knapsack(row["Weights"], row["Prices"], row["Capacity"])
            predictions.append(pred_picks.tolist())
            if np.array_equal(pred_picks, row["Best picks"]):
                correct += 1

        df["Predicted picks"] = predictions
        accuracy = correct / len(df)

        st.write(f"### 🎯 DP Accuracy vs Provided Optimal: **{accuracy*100:.2f}%**")
        st.dataframe(df.head())

        st.download_button(
            "📥 Download Results",
            df.to_csv(index=False),
            file_name="knapsack_results.csv"
        )

        # -------- Plot Bar Chart of First Row --------
        st.write("### 📊 Profit & Weight chart (first item set)")
        sample = df.iloc[0]
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(range(len(sample["Prices"])), sample["Prices"])
        ax.set_title("Profit per Item")
        st.pyplot(fig)

        # -------- Heatmap ----------
        st.write("### 🔥 Heatmap (Feature Importance)")
        feat_importance = model.feature_importances_
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.heatmap([feat_importance], cmap="viridis", ax=ax2)
        ax2.set_title("Feature Importance Heatmap")
        st.pyplot(fig2)

# ------------------------------
# Manual Input
# ------------------------------
st.write("---")
st.subheader("🧪 Try Your Own Items")

num_items = st.number_input("Number of items", 1, 10, 5)
weights = st.text_input("Enter weights", "10,20,30,40,50")
prices = st.text_input("Enter prices", "5,7,10,15,20")
capacity = st.number_input("Capacity", 1, 200, 60)

if st.button("Solve Custom Knapsack"):
    W = list(map(int, weights.split(",")))
    P = list(map(int, prices.split(",")))

    picks, value = knapsack(W, P, capacity)

    st.write(f"### ✅ Selected: `{picks}`")
    st.write(f"### 💰 Total Value: `{value}`")

# ------------------------------
# Footer
# ------------------------------
st.write("---")
st.markdown("<h4 style='text-align:center'>✨ Made by <b>Pragya Srivastava</b> ✨</h4>", unsafe_allow_html=True)
