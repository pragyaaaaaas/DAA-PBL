import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

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

    return picks, dp[n][capacity], dp

# ------------------------------
# Streamlit UI
# ------------------------------

st.markdown("<h1 style='text-align:center;'>🎒 Knapsack Optimization — Feature Selection Style Demo</h1>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload `knapsack_5_items.csv`", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    df["Weights"] = df["Weights"].apply(lambda x: ast.literal_eval(x))
    df["Prices"] = df["Prices"].apply(lambda x: ast.literal_eval(x))
    df["Best picks"] = df["Best picks"].apply(lambda x: np.array(ast.literal_eval(x)))

    st.success("✅ Dataset loaded successfully")
    st.write("### Preview")
    st.dataframe(df.head())

    # Graph 1 — total weights vs total prices
    st.write("### 📊 Dataset Summary Plot")

    df["Total Weight"] = df["Weights"].apply(sum)
    df["Total Price"] = df["Prices"].apply(sum)

    fig, ax = plt.subplots()
    ax.scatter(df["Total Weight"], df["Total Price"])
    ax.set_xlabel("Total Weight")
    ax.set_ylabel("Total Price")
    ax.set_title("Weight vs Price Distribution")
    st.pyplot(fig)

    if st.button("Run Knapsack Evaluation"):
        correct = 0
        predictions = []
        
        heatmaps = []

        for idx, row in df.iterrows():
            pred_picks, pred_value, dp = knapsack(row["Weights"], row["Prices"], row["Capacity"])
            predictions.append(pred_picks.tolist())

            heatmaps.append(dp)
            if np.array_equal(pred_picks, row["Best picks"]):
                correct += 1

        df["Predicted picks"] = predictions
        accuracy = correct / len(df)

        st.write("### ✅ Results")
        st.write(f"### 🎯 Accuracy vs Provided Optimal: **{accuracy*100:.2f}%**")
        st.dataframe(df.head())

        # DP heatmap for first item
        st.write("### 🔥 Dynamic Programming Matrix Heatmap (first test)")
        fig2, ax2 = plt.subplots()
        sns.heatmap(heatmaps[0], cmap="viridis", ax=ax2)
        st.pyplot(fig2)

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

num_items = st.number_input("Number of items", 1, 10, 5)
weights = st.text_input("Enter weights (comma-separated)", "10,20,30,40,50")
prices = st.text_input("Enter prices (comma-separated)", "5,7,10,15,20")
capacity = st.number_input("Capacity", 1, 200, 60)

if st.button("Solve Custom Knapsack"):
    W = list(map(int, weights.split(",")))
    P = list(map(int, prices.split(",")))

    picks, value, dp = knapsack(W, P, capacity)

    st.write(f"### 🎒 Selected Items: `{picks}`")
    st.write(f"### 💰 Total Value: `{value}`")

    # Visualize DP matrix
    st.write("#### 🔍 DP Table Visualization")
    fig3, ax3 = plt.subplots()
    sns.heatmap(dp, cmap="magma", ax=ax3)
    st.pyplot(fig3)

# Footer
st.write("---")
st.markdown("<h4 style='text-align:center;'>Made by Pragya Srivastava</h4>", unsafe_allow_html=True)
