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
st.markdown(
    "<h1 style='text-align:center; color:#2e8b57;'>🤖 Smart Knapsack Optimizer</h1>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader("Upload `knapsack_5_items.csv`", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df["Weights"] = df["Weights"].apply(lambda x: ast.literal_eval(x))
    df["Prices"] = df["Prices"].apply(lambda x: ast.literal_eval(x))
    df["Best picks"] = df["Best picks"].apply(lambda x: np.array(ast.literal_eval(x)))

    st.success("✅ Dataset Loaded Successfully")
    st.write("### Preview")
    st.dataframe(df)

    st.write("### 🎯 Select Row to Optimize")
    row_index = st.slider("Choose dataset row", 0, len(df)-1, 0)

    weights = df.iloc[row_index]["Weights"]
    values = df.iloc[row_index]["Prices"]
    capacity = df.iloc[row_index]["Capacity"]

    mode = st.selectbox("Choose Knapsack Method", 
                        ["DP Optimal Solution", "Greedy by Weight", "Greedy by Profit", "Greedy by Profit/Weight"])

    if st.button("Run Optimization"):
        if mode == "DP Optimal Solution":
            picks, best_profit = knapsack_dp(weights, values, capacity)
        else:
            picks, best_profit = greedy(weights, values, capacity, mode)

        st.write(f"### ✅ Selected: `{picks}`")
        st.write(f"### 💰 Total Profit: `{best_profit}`")

        # --------- Profit & Weight Charts ---------
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].bar(range(len(values)), values)
        ax[0].set_title("Profit per Item")

        ax[1].bar(range(len(weights)), weights)
        ax[1].set_title("Weight per Item")

        st.pyplot(fig)

        # -------- Heatmap for Profit-to-Weight --------
        ratios = np.array(values) / np.array(weights)
        st.write("### 🔥 Profit-to-Weight Importance Heatmap")
        fig2, ax2 = plt.subplots(figsize=(4, 2))
        sns.heatmap([ratios], cmap="viridis", annot=True, ax=ax2)
        st.pyplot(fig2)

# ---------------- Manual Input ----------------
st.write("---")
st.subheader("🧪 Try Your Own Items")

item_count = st.slider("Number of items", 1, 10, 5)
capacity = st.slider("Knapsack Capacity", 10, 200, 60)

weights = []
profits = []

for i in range(item_count):
    w = st.slider(f"Weight of Item {i+1}", 1, 50, 10, key=f"w{i}")
    p = st.slider(f"Profit of Item {i+1}", 1, 100, 20, key=f"p{i}")
    weights.append(w)
    profits.append(p)

method = st.radio("Choose Strategy", 
                  ["DP Optimal Solution", "Greedy by Weight", "Greedy by Profit", "Greedy by Profit/Weight"], key="manual_method")

if st.button("Solve Custom Case"):
    if method == "DP Optimal Solution":
        picks, val = knapsack_dp(weights, profits, capacity)
    else:
        picks, val = greedy(weights, profits, capacity, method)

    st.write(f"### ✅ Selected Items: `{picks}`")
    st.write(f"### 💰 Max Profit: `{val}`")

    # ---------- 📊 Bubble Chart ----------
    st.write("### 📍 Items — Weight vs Profit (Bubble size = Profit/Weight)")

    ratios = np.array(profits) / np.array(weights)
    colors = ['green' if picks[i] == 1 else 'gray' for i in range(len(picks))]

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.scatter(weights, profits, s=ratios*300, c=colors)

    for i in range(item_count):
        ax3.annotate(f"I{i+1}", (weights[i]+0.2, profits[i]+0.2), fontsize=9)

    ax3.set_xlabel("Weight")
    ax3.set_ylabel("Profit")
    ax3.set_title(f"Bubble Chart ({method})")
    st.pyplot(fig3)

    # ---------- 🟩 Gantt-Style Representation ----------
    st.write("### 📦 Gantt-Style Item Allocation Chart")

    fig4, ax4 = plt.subplots(figsize=(7, 3))
    y = 0
    for i in range(item_count):
        if picks[i] == 1:
            ax4.barh(y, weights[i], left=sum(np.array(weights)*picks) - sum(weights[i:]), height=0.6)
            ax4.text(sum(np.array(weights)*picks) - sum(weights[i:]) + weights[i]/2, y, f"Item {i+1}", ha='center', va='center')
            y += 1

    ax4.set_title("Items Packed in Knapsack (Timeline View)")
    ax4.set_xlabel("Weight Usage")
    ax4.set_yticks([])
    st.pyplot(fig4)

# ---------------- Footer ------------------
st.markdown("""
<br><hr>
<div style='text-align:center;'>
<img src='logo.png' width='60' style='margin-bottom:10px;'>

<br>
<b style='font-size:18px;'>Smart Knapsack Optimizer</b><br>
<span style='font-size:16px;'>Developed by Pragya Srivastava</span><br>
<p style='font-size:14px;'>
© 2025, GrowWise — AI powered Smart Knapsack Optimizer
</p>
</div>
""", unsafe_allow_html=True)
