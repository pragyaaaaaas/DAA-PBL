import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Smart Knapsack Optimizer", layout="wide")

# ---------------- DP Solver ----------------
def knapsack_dp(weights, values, capacity, animate=False):
    n = len(weights)
    dp = np.zeros((n+1, capacity+1), dtype=int)

    if animate:
        st.write("### 🧮 DP Table Progress")
        progress = st.progress(0)

    for i in range(1, n+1):
        for c in range(capacity+1):
            if weights[i-1] <= c:
                dp[i][c] = max(dp[i-1][c], values[i-1] + dp[i-1][c-weights[i-1]])
            else:
                dp[i][c] = dp[i-1][c]

        if animate:
            progress.progress(i / n)
            st.write(f"Iteration **{i}/{n}**")
            fig, ax = plt.subplots()
            sns.heatmap(dp[:i+1,:], annot=True, cmap="coolwarm", ax=ax)
            ax.set_title(f"DP Table after item {i}")
            st.pyplot(fig)

    picks = np.zeros(n, dtype=int)
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i-1][c]:
            picks[i-1] = 1
            c -= weights[i-1]

    return picks, dp[n][capacity], dp

# ---------------- Greedy ----------------
def greedy(weights, values, capacity, mode):
    items = list(range(len(weights)))

    if mode == "Greedy by Weight":
        items.sort(key=lambda i: weights[i])
    elif mode == "Greedy by Profit":
        items.sort(key=lambda i: values[i], reverse=True)
    else:
        items.sort(key=lambda i: (values[i] / weights[i]), reverse=True)

    total_profit = 0
    total_weight = 0
    picks = np.zeros(len(weights), int)

    for i in items:
        if total_weight + weights[i] <= capacity:
            picks[i] = 1
            total_weight += weights[i]
            total_profit += values[i]

    return picks, total_profit

# ---------------- ML Predictor ----------------
def train_ml_model(df):
    X, y = [], []

    for _, row in df.iterrows():
        w = ast.literal_eval(row["Weights"])
        p = ast.literal_eval(row["Prices"])
        cap = row["Capacity"]
        picks = np.array(ast.literal_eval(row["Best picks"]))

        for i in range(len(w)):
            X.append([w[i], p[i], cap])
            y.append(picks[i])

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center; color:#2e8b57;'>🤖 Smart Knapsack Optimizer</h1>",
            unsafe_allow_html=True)

uploaded = st.file_uploader("Upload knapsack_5_items.csv", type=["csv"])

model = None

if uploaded:
    df = pd.read_csv(uploaded)
    df["Weights"] = df["Weights"].apply(lambda x: ast.literal_eval(x))
    df["Prices"] = df["Prices"].apply(lambda x: ast.literal_eval(x))
    df["Best picks"] = df["Best picks"].apply(lambda x: np.array(ast.literal_eval(x)))

    st.success("✅ Dataset Loaded Successfully")
    st.write("### 📁 Preview")
    st.dataframe(df.head())

    # Train ML model
    model = train_ml_model(df)
    st.info("🤖 ML Model Trained (Random Forest)")

    row_index = st.slider("Select dataset row to test", 0, len(df) - 1, 0)

    weights = df.iloc[row_index]["Weights"]
    values = df.iloc[row_index]["Prices"]
    capacity = df.iloc[row_index]["Capacity"]

    method = st.selectbox("Choose Optimization Method", 
        ["DP Optimal Solution", "Greedy by Weight", "Greedy by Profit", "Greedy by Profit/Weight", "ML Predicted Picks"])

    animate = st.checkbox("🎞️ Show Step-by-Step DP Animation", value=False)

    if st.button("Run Optimization"):
        if method == "ML Predicted Picks":
            feat = [[weights[i], values[i], capacity] for i in range(len(weights))]
            picks = model.predict(feat)
            best_profit = sum(np.array(values) * picks)
            dp_table = None
        elif method == "DP Optimal Solution":
            picks, best_profit, dp_table = knapsack_dp(weights, values, capacity, animate)
        else:
            picks, best_profit = greedy(weights, values, capacity, method)
            dp_table = None

        st.write(f"### ✅ Selected Items: `{picks}`")
        st.write(f"### 💰 Total Profit: `{best_profit}`")

        # DP Table Heatmap
        if dp_table is not None:
            st.write("### 📊 Final DP Table Heatmap")
            fig = plt.figure(figsize=(6,4))
            sns.heatmap(dp_table, annot=False, cmap="coolwarm")
            st.pyplot(fig)

        # Profit & Weight bar chart
        fig2, ax2 = plt.subplots(1, 2, figsize=(10,4))
        ax2[0].bar(range(len(values)), values); ax2[0].set_title("Profit per Item")
        ax2[1].bar(range(len(weights)), weights); ax2[1].set_title("Weight per Item")
        st.pyplot(fig2)

        # Bubble chart
        ratios = np.array(values) / np.array(weights)
        colors = ['green' if picks[i] == 1 else 'gray' for i in range(len(picks))]

        st.write("### ⚖️ Weight vs Profit Bubble Chart")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.scatter(weights, values, s=ratios * 300, c=colors)

        for i in range(len(weights)):
            ax3.text(weights[i], values[i], f"I{i+1}")

        ax3.set_xlabel("Weight"); ax3.set_ylabel("Profit")
        st.pyplot(fig3)

# ---------------- Footer ----------------
st.write("---")
st.markdown("<h4 style='text-align:center; color:#FF0080;'>✨ Made by <b>Pragya Srivastava</b> ✨</h4>",
            unsafe_allow_html=True)
