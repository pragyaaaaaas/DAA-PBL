import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    dp_picks, dp_profit = knapsack_dp(weights, profits, capacity)
    picks, result_profit = (dp_picks, dp_profit) if method == "DP Optimal Solution" else greedy(weights, profits, capacity, method)

    st.success(f"✅ Chosen Method: {method}")
    st.write(f"### 🎯 Selected Items: `{picks}`")
    st.write(f"### 💰 Total Profit: `{result_profit}`")

    # ---------------- Bubble Chart ----------------
    st.write("### 📍 Weight vs Profit")
    ratios = np.array(profits) / np.array(weights)
    colors = ['green' if picks[i] == 1 else 'gray' for i in range(len(picks))]

    figb, axb = plt.subplots(figsize=(5, 3))
    axb.scatter(weights, profits, s=ratios * 200 + 60, c=colors)

    # ✅ Clean label handling for overlapping points
    existing_positions = {}
    for i in range(item_count):
        pos = (weights[i], profits[i])
        label = f"I{i+1}"
        if pos in existing_positions:
            existing_positions[pos].append(label)
        else:
            existing_positions[pos] = [label]

    for pos, labels in existing_positions.items():
        axb.text(pos[0] + 0.15, pos[1] + 0.15, ", ".join(labels), fontsize=8)

    axb.set_xlabel("Weight")
    axb.set_ylabel("Profit")
    st.pyplot(figb)

    # ---------------- Gantt Chart ----------------
    st.write("### 📦 Gantt Packing Timeline")

    figg, axg = plt.subplots(figsize=(6, 1.8))
    start = 0
    segments = {}

    for i in range(item_count):
        if picks[i] == 1:
            length = weights[i]
            interval = (start, start + length)

            if interval not in segments:
                segments[interval] = []
            segments[interval].append(f"I{i+1}")

            start += length

    for (left, right), items in segments.items():
        width = right - left
        axg.barh("Knapsack", width, left=left)
        axg.text(left + width/2, 0, ", ".join(items), color="white",
                 ha='center', va='center', fontsize=8)

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
<p style='font-size:14px;'>© 2025 — AI powered Smart Knapsack Optimizer</p>
</div>
""", unsafe_allow_html=True)
