import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Smart Knapsack Optimizer", layout="wide")

# ---- Header ----
st.markdown(
    "<h2 style='text-align:center; font-weight:bold;'>Smart Knapsack Optimizer</h2>",
    unsafe_allow_html=True
)

# Sample Dataset
data = {
    "Object": ["I1", "I2", "I3", "I4", "I5"],
    "Profit": [10, 20, 30, 40, 35],
    "Weight": [2, 5, 10, 5, 7]
}
df = pd.DataFrame(data)
df["Profit/Weight"] = df["Profit"] / df["Weight"]

# Sidebar
st.sidebar.header("Knapsack Controls")
capacity = st.sidebar.slider("Knapsack Capacity", 5, 30, 15)
strategy = st.sidebar.radio(
    "Greedy Optimization Strategy",
    ["Greedy by Profit", "Greedy by Weight (Min)", "Greedy by Profit/Weight"]
)

# Sort based on strategy
if strategy == "Greedy by Profit":
    df_sorted = df.sort_values(by="Profit", ascending=False)
elif strategy == "Greedy by Weight (Min)":
    df_sorted = df.sort_values(by="Weight", ascending=True)
else:
    df_sorted = df.sort_values(by="Profit/Weight", ascending=False)

# Apply Knapsack logic
total_weight = 0
total_profit = 0
selected_items = []

for _, row in df_sorted.iterrows():
    if total_weight + row["Weight"] <= capacity:
        total_weight += row["Weight"]
        total_profit += row["Profit"]
        selected_items.append(row["Object"])

# Result
st.success(f"📦 **Selected Items:** {', '.join(selected_items)}")
st.info(f"💰 **Total Profit:** {total_profit} | ⚖ Total Weight: {total_weight}")

# ---- Download Results ----
result_df = pd.DataFrame({
    "Selected Items": [", ".join(selected_items)],
    "Total Profit": [total_profit],
    "Total Weight": [total_weight]
})
st.download_button("📥 Download Result", result_df.to_csv(index=False), "knapsack_result.csv")

# ---- Visualization Layout ----
col1, col2 = st.columns(2)

# ✅ Bubble Chart
with col1:
    st.write("### 🎈 Feature Bubble Chart")

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.scatter(df["Weight"], df["Profit"], s=df["Profit"] * 8)

    for i, row in df.iterrows():
        ax.text(row["Weight"], row["Profit"], row["Object"], fontsize=8)

    ax.set_xlabel("Weight")
    ax.set_ylabel("Profit")
    st.pyplot(fig)

# ✅ Profit Comparison Chart
with col2:
    st.write("### 📊 Profit Comparison Chart")

    fig2, ax2 = plt.subplots(figsize=(4.5, 3))
    bars = ax2.bar(df["Object"], df["Profit"])
    ax2.set_ylabel("Profit")
    st.pyplot(fig2)

# ✅ Gantt Timeline Chart (NOW BLUE ✅ + combined labels)
st.write("### 📦 Knapsack Packing Timeline")

fig3, ax3 = plt.subplots(figsize=(9, 1.8))
start = 0
segments = {}

for obj in selected_items:
    w = float(df[df["Object"] == obj]["Weight"])
    interval = (start, start + w)

    if interval not in segments:
        segments[interval] = []
    segments[interval].append(obj)

    start += w

for (l, r), items in segments.items():
    width = r - l
    label = ", ".join(items)

    # 🔵 Change Gantt bar color here
    ax3.barh("Knapsack", width, left=l, color="#3b82f6")  # Blue!
    ax3.text(l + width/2, 0, label, ha="center", va="center", color="white", fontsize=8)

ax3.set_xlim(0, capacity)
st.pyplot(fig3)

# ---- Footer ----
st.markdown("""
<div style='text-align:center; margin-top:25px; font-size:14px;'>
🤖 <b>Smart Knapsack Optimizer | Developed by Pragya Srivastava</b><br>
© 2025 — AI powered Smart Knapsack Optimizer
</div>
""", unsafe_allow_html=True)
