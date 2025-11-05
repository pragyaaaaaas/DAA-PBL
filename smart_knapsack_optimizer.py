import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Knapsack Optimizer", layout="wide")

# ------------------------------
# 0/1 Knapsack DP
# ------------------------------
def knapsack_dp(weights, values, capacity):
    n = len(weights)
    dp = np.zeros((n+1, capacity+1), dtype=int)

    for i in range(1, n+1):
        for c in range(capacity + 1):
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
# Greedy Methods
# ------------------------------
def greedy(weights, values, capacity, mode):
    items = list(range(len(weights)))

    if mode == "Greedy by Weight":
        items.sort(key=lambda i: weights[i])
    elif mode == "Greedy by Profit":
        items.sort(key=lambda i: values[i], reverse=True)
    else:
        items.sort(key=lambda i: values[i] / weights[i], reverse=True)

    total_profit, total_weight = 0, 0
    picks = np.zeros(len(weights), int)

    for i in items:
        if total_weight + weights[i] <= capacity:
            picks[i] = 1
            total_weight += weights[i]
            total_profit += values[i]

    return picks, total_profit

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center; color:#2e8b57;'>🤖 Smart Knapsack Optimizer</h1>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload `knapsack_5_items.csv`", type=["csv"])

# ---------- Function to Draw Bubble Chart ----------
def draw_bubble_chart(weights, profits, picks, title):
    ratios = np.array(profits) / np.array(weights)
    colors = ['green' if picks[i] == 1 else 'gray' for i in range(len(picks))]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(weights, profits, s=ratios*300, c=colors)

    for i in range(len(weights)):
        ax.text(weights[i], profits[i], f" I{i+1}", fontsize=9)

    ax.set_xlabel("Weight")
    ax.set_ylabel("Profit")
    ax.set_title(title)
    st.pyplot(fig)

# ----------------------------------------------------
# CSV Mode
# ----------------------------------------------------
if uploaded:
    df = pd.read_csv(uploaded)
    df["Weights"] = df["Weights"].apply(lambda x: ast.literal_eval(x))
    df["Prices"] = df["Prices"].apply(lambda x: ast.literal_eval(x))
    df["Best picks"] = df["Best picks"].apply(lambda x: np.array(ast.literal_eval(x)))

    st.success("✅ Dataset Loaded Successfully")
    st.write("### Preview Data")
    st.dataframe(df)

    row_index = st.slider("Choose dataset row", 0, len(df)-1, 0)

    weights = df.iloc[row_index]["Weights"]
    values = df.iloc[row_index]["Prices"]
    capacity = df.iloc[row_index]["Capacity"]

    mode = st.selectbox("Method", 
                        ["DP Optimal Solution", "Greedy by Weight", "Greedy by Profit", "Greedy by Profit/Weight"])

    if st.button("Run Optimization"):
        if mode == "DP Optimal Solution":
            picks, best_profit = knapsack_dp(weights, values, capacity)
        else:
            picks, best_profit = greedy(weights, values, capacity, mode)

        st.success(f"✅ Selected Items: {picks}")
        st.info(f"💰 Total Profit: {best_profit}")

        # Bar Charts
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].bar(range(len(values)), values)
        ax[0].set_title("Profit per Item")
        ax[1].bar(range(len(weights)), weights)
        ax[1].set_title("Weight per Item")
        st.pyplot(fig)

        # Heatmap
        ratios = np.array(values) / np.array(weights)
        fig2, ax2 = plt.subplots(figsize=(4,2))
        sns.heatmap([ratios], cmap="viridis", annot=True, ax=ax2)
        st.pyplot(fig2)

        # dynamic bubble chart
        draw_bubble_chart(weights, values, picks, f"Selected Items — {mode}")

# ----------------------------------------------------
# Manual Mode
# ----------------------------------------------------
st.write("---")
st.subheader("🧪 Try Your Own Items")

item_count = st.slider("Number of items", 1, 10, 5)
capacity = st.slider("Knapsack Capacity", 10, 200, 60)

weights, profits = [], []
for i in range(item_count):
    weights.append(st.slider(f"Weight of Item {i+1}", 1, 50, 10, key=f"w{i}"))
    profits.append(st.slider(f"Profit of Item {i+1}", 1, 100, 20, key=f"p{i}"))

method = st.radio("Method",
                  ["DP Optimal Solution", "Greedy by Weight", "Greedy by Profit", "Greedy by Profit/Weight"])

if st.button("Solve Custom Case"):
    if method == "DP Optimal Solution":
        picks, val = knapsack_dp(weights, profits, capacity)
    else:
        picks, val = greedy(weights, profits, capacity, method)

    st.success(f"✅ Picks: {picks}")
    st.info(f"💰 Max Profit: {val}")

    draw_bubble_chart(weights, profits, picks, f"Custom Items — {method}")

# ---------------- Footer ----------------
st.write("---")
st.markdown("<h4 style='text-align:center; color:#ff0080;'>✨ Made by <b>Pragya Srivastava</b> ✨</h4>", unsafe_allow_html=True)
