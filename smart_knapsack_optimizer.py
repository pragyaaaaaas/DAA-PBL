import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import StringIO

# ----------------- Page Config -----------------
st.set_page_config(page_title="Smart Knapsack Optimizer", layout="wide")

# ----------------- Theme Toggle -----------------
mode = st.radio("Theme Mode:", ["🌙 Dark", "☀️ Light"], horizontal=True)

if mode == "🌙 Dark":
    st.markdown("""
        <style>
            body { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body { background-color: #FFFFFF; color: black; }
        </style>
    """, unsafe_allow_html=True)

# ----------------- Heading -----------------
st.markdown("<h1 style='text-align:center;'>Smart Knapsack Optimizer 🤖</h1>", unsafe_allow_html=True)

# ----------------- Dataset -----------------
data = {
    "Item": ["A", "B", "C", "D", "E"],
    "Profit": [60, 100, 120, 90, 30],
    "Weight": [10, 20, 30, 15, 5]
}
df = pd.DataFrame(data)
df["Profit/Weight"] = (df["Profit"] / df["Weight"]).round(2)

# Sidebar capacity slider
capacity = st.sidebar.slider("Knapsack Capacity", 10, 100, 50)

criterion = st.sidebar.radio("Select Optimization Strategy",
                             ["Greedy: Max Profit", "Greedy: Min Weight", "Greedy: Profit/Weight"])

# ----------------- Knapsack Logic -----------------
df_sorted = df.copy()
if criterion == "Greedy: Max Profit":
    df_sorted = df.sort_values(by="Profit", ascending=False)
elif criterion == "Greedy: Min Weight":
    df_sorted = df.sort_values(by="Weight")
else:
    df_sorted = df.sort_values(by="Profit/Weight", ascending=False)

selected_items = []
total_profit = 0
total_weight = 0

for _, row in df_sorted.iterrows():
    if total_weight + row["Weight"] <= capacity:
        selected_items.append(row["Item"])
        total_profit += row["Profit"]
        total_weight += row["Weight"]

# ----------------- Results -----------------
st.subheader("Selected Items")
st.write(", ".join(selected_items) if selected_items else "No items selected")

st.metric("📦 Total Weight", total_weight)
st.metric("💰 Total Profit", total_profit)

# ----------------- Bubble Plot -----------------
st.subheader("📊 Profit vs Weight Bubble Chart")

fig = px.scatter(df, x="Weight", y="Profit", size="Profit", text="Item")
fig.update_traces(textposition="top right")  # Keep labels close
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

# ----------------- Gantt Chart -----------------
st.subheader("⏱ Gantt Chart - Item Timeline (Sequential fill)")

gantt = pd.DataFrame({
    "Task": selected_items,
    "Start": np.cumsum([0] + [df[df["Item"] == i]["Weight"].values[0] for i in selected_items[:-1]]),
})

gantt["Finish"] = gantt["Start"] + [df[df["Item"] == i]["Weight"].values[0] for i in selected_items]

fig2 = px.timeline(gantt, x_start="Start", x_end="Finish", y="Task")
fig2.update_layout(height=350)
st.plotly_chart(fig2, use_container_width=True)

# ----------------- Download Results -----------------
st.subheader("⬇️ Download Results")

result_csv = StringIO()
result_data = pd.DataFrame({
    "Selected Items": [", ".join(selected_items)],
    "Total Profit": [total_profit],
    "Total Weight": [total_weight]
})
result_data.to_csv(result_csv, index=False)

st.download_button("Download Results CSV", result_csv.getvalue(), "knapsack_results.csv")

# ----------------- Footer -----------------
st.markdown("""
<br><hr>
<div style='text-align:center;'>
<span style='font-size:28px;'>🤖</span><br>
<b style='font-size:18px;'>Smart Knapsack Optimizer</b><br>
<span style='font-size:16px;'>Developed by Pragya Srivastava</span><br>
<p style='font-size:14px;'>
© 2025, GrowWise — AI powered Smart Knapsack Optimizer
</p>
</div>
""", unsafe_allow_html=True)
