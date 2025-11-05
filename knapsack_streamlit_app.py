import streamlit as st
import pandas as pd
import numpy as np
import ast

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


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎒 Knapsack Optimization — Feature Selection Style Demo")

uploaded = st.file_uploader("Upload `knapsack_5_items.csv`", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # parse list strings into real lists
    df["Weights"] = df["Weights"].apply(lambda x: ast.literal_eval(x))
    df["Prices"] = df["Prices"].apply(lambda x: ast.literal_eval(x))
    df["Best picks"] = df["Best picks"].apply(lambda x: np.array(ast.literal_eval(x)))

    st.success("✅ Dataset loaded successfully")
    st.write("### Preview")
    st.dataframe(df.head())

    if st.button("Run Knapsack Evaluation"):
        correct = 0
        predictions = []

        for idx, row in df.iterrows():
            pred_picks, pred_value = knapsack(row["Weights"], row["Prices"], row["Capacity"])
            predictions.append(pred_picks.tolist())

            if np.array_equal(pred_picks, row["Best picks"]):
                correct += 1

        df["Predicted picks"] = predictions
        accuracy = correct / len(df)

        st.write("### ✅ Results")
        st.write(f"### 🎯 Accuracy vs provided optimal: **{accuracy*100:.2f}%**")
        st.dataframe(df.head())

        st.download_button(
            "📥 Download Results CSV",
            df.to_csv(index=False),
            file_name="knapsack_results.csv"
        )


# ------------------------------
# Manual Test Section
# ------------------------------
st.write("---")
st.write("### 🧪 Try Your Own Input")

# allow manual testing
num_items = st.number_input("Number of items", 1, 10, 5)
weights = st.text_input("Enter weights (comma-separated)", "10,20,30,40,50")
prices = st.text_input("Enter prices (comma-separated)", "5,7,10,15,20")
capacity = st.number_input("Capacity", 1, 200, 60)

if st.button("Solve Custom Knapsack"):
    W = list(map(int, weights.split(",")))
    P = list(map(int, prices.split(",")))

    picks, value = knapsack(W, P, capacity)

    st.write(f"### 🎒 Selected Items: `{picks}`")
    st.write(f"### 💰 Total Value: `{value}`")
