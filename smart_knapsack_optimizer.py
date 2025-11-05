import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Knapsack Optimizer", layout="wide")

# ------------------------------
# Knapsack DP
# ------------------------------
def knapsack_dp(weights, values, capacity):
    n = len(weights)
    dp = np.zeros((n + 1, capacity + 1), dtype=int)

    for i in range(1, n + 1):
        for c in range(capacity + 1):
            if weights[i - 1] <= c:
                dp[i][c] = max(dp[i - 1][c], values[i - 1] + dp[i - 1][c - weights[i - 1]])
            else:
                dp[i][c] = dp[i - 1][c]

    picks = np.zeros(n, dtype=int)
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            picks[i - 1] = 1
            c -= weights[i - 1]

    return picks, dp[n][capacity]

# ------------------------------
# Greedy Methods
# ------------------------------
def greedy(weights, values, capacity, mode):
    items = list(range(len(weights)))

    if mode == "Greedy by Weight":
        items.sort(key=lambda x: weights[x])
    elif mode == "Greedy by Profit":
        items.sort(key=lambda x: values[x], reverse=True)
    else:
        items.sort(key=lambda x: values[x] / weights[x], reverse=True)

    picks = np.zeros(len(weights), int)
    total_w, total_p = 0, 0

    for i in items:
        if total_w + weights[i] <= capacity:
            picks[i] = 1
            total_w += weights[i]
            total_p += values[i]

    return picks, total_p

# ---------------- UI Header ----------------
st.markdown(
"<h1 style='text-align:center; color:#2e8b57;'>🤖 Smart Knapsack Optimizer</h1>",
unsafe_allow_html=True
)

# Theme Toggle
theme = st.radio("Choose Theme", ["Light", "Dark"], horizontal=True)
if theme == "Dark":
    st.markdown("""
        <style>
            body, .css-18e3th9 { background-color: #0f1116 !important; color:white }
        </style>
    """, unsafe_allow_html=True)

# ---------------- Manual Input ----------------
st.write("---")
st.subheader("📦 Enter Your Items")

item_count = st.slider("Number of items", 1, 10, 5)
capacity = st.slider("Knapsack Capacity", 10, 200, 60)

weights, profits = [], []
for i in range(item_count):
    weights.append(st.slider(f"Weight of Item {i+1}", 1, 50, 10, key=f"w{i}"))
    profits.append(st.slider(f"Profit of Item {i+1}", 1, 100, 20, key=f"p{i}"))

method = st.radio("Choose Strategy", 
                  ["DP Optimal Solution", "Greedy by Weight", "Greedy by Profit", "Greedy by Profit/Weight"],
                  horizontal=True)

if st.button("Run Optimization"):
    # Compute results
    dp_picks, dp_profit = knapsack_dp(weights, profits, capacity)
    picks, result_profit = (dp_picks, dp_profit) if method == "DP Optimal Solution" else greedy(weights, profits, capacity, method)

    # Results
    st.success(f"✅ Chosen Method: {method}")
    st.write(f"### 🎯 Selected: `{picks}`")
    st.write(f"### 💰 Total Profit: `{result_profit}`")

    # Comparison Bar Chart
    greedy_w = greedy(weights, profits, capacity, "Greedy by Weight")[1]
    greedy_p = greedy(weights, profits, capacity, "Greedy by Profit")[1]
    greedy_ratio = greedy(weights, profits, capacity, "Greedy by Profit/Weight")[1]

    comp_df = pd.DataFrame({
        "Method": ["DP", "Greedy-Weight", "Greedy-Profit", "Greedy-Ratio"],
        "Profit": [dp_profit, greedy_w, greedy_p, greedy_ratio]
    })

    st.write("### 📊 Profit Comparison Chart")
    figc, axc = plt.subplots()
    axc.bar(comp_df.Method, comp_df.Profit)
    st.pyplot(figc)

    # Bubble Chart
    st.write("### 📍 Weight vs Profit")
    ratios = np.array(profits) / np.array(weights)
    colors = ['green' if picks[i] == 1 else 'gray' for i in range(len(picks))]
    figb, axb = plt.subplots()
    axb.scatter(weights, profits, s=ratios*260 + 80, c=colors)
    for i in range(item_count):
        axb.text(weights[i]+0.2, profits[i]+0.2, f"I{i+1}")
    st.pyplot(figb)

    # Gantt Chart
    st.write("### 📦 Gantt Packing Timeline")
    figg, axg = plt.subplots(figsize=(7,2))
    start = 0
    for i in range(item_count):
        if picks[i] == 1:
            axg.barh("Knapsack", weights[i], left=start)
            axg.text(start+weights[i]/2, 0, f"I{i+1}", color="white", va='center', ha='center')
            start += weights[i]
    axg.set_xlim(0, capacity)
    st.pyplot(figg)

    # ✅ Download Results
    result_df = pd.DataFrame({
        "Item": [f"I{i+1}" for i in range(item_count)],
        "Weight": weights,
        "Profit": profits,
        "Selected": picks
    })

    st.download_button(
        "📥 Download Result CSV",
        result_df.to_csv(index=False),
        file_name="knapsack_result.csv",
        mime="text/csv"
    )

# ---------------- Footer ------------------
st.markdown("""
<br><hr>
<div style='text-align:center;'>
<span style='font-size:50px;'>🤖</span><br>
<b style='font-size:20px;'>Smart Knapsack Optimizer</b><br>
<span style='font-size:17px;'>Developed by <b>Pragya Srivastava</b></span><br>
<p style='font-size:14px;'>© 2025, GrowWise — AI powered Smart Knapsack Optimizer</p>
</div>
""", unsafe_allow_html=True)
