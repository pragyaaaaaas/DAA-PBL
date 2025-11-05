# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Smart Knapsack Optimizer", layout="wide")

# -------------------------
# Helpers: knapsack solvers
# -------------------------
def knapsack_dp(weights, values, capacity):
    n = len(weights)
    dp = np.zeros((n + 1, capacity + 1), dtype=int)
    for i in range(1, n + 1):
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
    return picks.astype(int), int(dp[n][capacity])

def greedy(weights, values, capacity, mode):
    idxs = list(range(len(weights)))
    if mode == "Greedy by Weight":
        idxs.sort(key=lambda i: weights[i])
    elif mode == "Greedy by Profit":
        idxs.sort(key=lambda i: values[i], reverse=True)
    else:  # ratio
        idxs.sort(key=lambda i: (values[i] / weights[i]) if weights[i] != 0 else float('inf'), reverse=True)

    picks = np.zeros(len(weights), dtype=int)
    total_w = 0
    total_p = 0
    for i in idxs:
        if total_w + weights[i] <= capacity:
            picks[i] = 1
            total_w += weights[i]
            total_p += values[i]
    return picks, int(total_p)

# -------------------------
# Theme CSS
# -------------------------
def apply_theme(dark: bool):
    if dark:
        css = """
        <style>
        .stApp, .css-1d391kg {background-color: #0f1116; color: #E6E6E6;}
        .stButton>button {background-color:#1f6feb;color:white}
        .stSlider>div>div>input {color: white}
        .st-bdfh4z {color: white}
        .block-container {padding-top:1rem;}
        </style>
        """
    else:
        css = """
        <style>
        .stApp, .css-1d391kg {background-color: white; color: black;}
        .stButton>button {background-color:#0f6ebf;color:white}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Page header
# -------------------------
st.markdown("<h1 style='text-align:center; color:#2e8b57;'>🤖 Smart Knapsack Optimizer</h1>", unsafe_allow_html=True)
st.write("")  # spacer

# Theme toggle (session state)
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

col_t = st.columns([1, 3, 1])
with col_t[0]:
    pass
with col_t[1]:
    theme_choice = st.radio("Theme", options=["Light", "Dark"], index=1 if st.session_state.dark_mode else 0, horizontal=True)
    st.session_state.dark_mode = (theme_choice == "Dark")
    apply_theme(st.session_state.dark_mode)
with col_t[2]:
    pass

# -------------------------
# Manual Input section (no upload)
# -------------------------
st.markdown("---")
st.subheader("📦 Enter Items (manual input)")

left_col, right_col = st.columns([1, 1])
with left_col:
    item_count = st.number_input("Number of items", min_value=1, max_value=12, value=5, step=1)
    capacity = st.number_input("Knapsack Capacity", min_value=1, max_value=1000, value=60, step=1)
with right_col:
    method = st.selectbox("Choose Strategy", ["DP Optimal Solution", "Greedy by Weight", "Greedy by Profit", "Greedy by Profit/Weight"])

# create sliders for weights & profits in a compact grid
weights = []
profits = []
cols = st.columns(3)
for i in range(item_count):
    c_idx = i % 3
    with cols[c_idx]:
        w = st.slider(f"W{i+1}", 1, 200, 10, key=f"w{i}")
        p = st.slider(f"P{i+1}", 1, 1000, 20, key=f"p{i}")
    weights.append(w)
    profits.append(p)

# Run optimization
if st.button("Run Optimization"):
    if method == "DP Optimal Solution":
        picks, total_profit = knapsack_dp(weights, profits, capacity)
    else:
        picks, total_profit = greedy(weights, profits, capacity, method)

    # show results
    st.success(f"Method: {method} — Total Profit = {total_profit}")
    st.write("Selected vector (1 = picked):", picks.tolist())

    # Build DataFrame summarizing items
    df_items = pd.DataFrame({
        "Item": [f"I{i+1}" for i in range(item_count)],
        "Weight": weights,
        "Profit": profits,
        "Profit/Weight": [round(p/w, 3) for p, w in zip(profits, weights)],
        "Selected": picks.astype(int)
    })

    # -------------------------
    # Comparison chart (DP vs Greedy variants)
    # -------------------------
    dp_p = knapsack_dp(weights, profits, capacity)[1]
    gw_p = greedy(weights, profits, capacity, "Greedy by Weight")[1]
    gp_p = greedy(weights, profits, capacity, "Greedy by Profit")[1]
    gr_p = greedy(weights, profits, capacity, "Greedy by Profit/Weight")[1]

    comp_df = pd.DataFrame({
        "Method": ["DP", "Greedy-Weight", "Greedy-Profit", "Greedy-Ratio"],
        "Profit": [dp_p, gw_p, gp_p, gr_p]
    })

    # Plot smaller-sized comparison bar (plotly)
    fig_comp = px.bar(comp_df, x="Method", y="Profit",
                      title="Profit Comparison (smaller size)",
                      height=300)
    st.plotly_chart(fig_comp, use_container_width=True)

    # -------------------------
    # Bubble chart (Plotly) - interactive, labels close
    # -------------------------
    fig_bubble = px.scatter(df_items, x="Weight", y="Profit", size="Profit/Weight",
                            color=df_items["Selected"].map({1: "Selected", 0: "Not selected"}),
                            hover_name="Item",
                            hover_data={"Weight":True, "Profit":True, "Profit/Weight":True, "Selected":True},
                            title=f"Weight vs Profit — {method}",
                            height=360)
    # adjust marker/text so labels stay near points
    fig_bubble.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers+text'))
    fig_bubble.update_layout(legend_title_text="Picked")
    st.plotly_chart(fig_bubble, use_container_width=True)

    # -------------------------
    # Gantt / timeline (Plotly)
    # -------------------------
    picked_df = df_items[df_items["Selected"] == 1].reset_index(drop=True)
    if not picked_df.empty:
        # create start/finish times using cumulative weight packing order (use the chosen algorithm order)
        # order relies on picks positions (we will pack in increasing index order to show timeline)
        start = 0
        tasks = []
        for idx, row in picked_df.iterrows():
            tasks.append(dict(Task=row["Item"], Start=start, Finish=start + int(row["Weight"]), Resource=f"W={row['Weight']}"))
            start += int(row["Weight"])
        gantt_df = pd.DataFrame(tasks)
        # plot using px.timeline
        fig_gantt = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Task", color="Resource", title="Gantt — Selected Items (timeline)", height=300)
        fig_gantt.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.info("No items selected — Gantt chart empty.")

    # -------------------------
    # Cumulative Weight vs Profit line chart (smaller)
    # -------------------------
    df_items_sorted = df_items.copy().reset_index(drop=True)
    df_items_sorted["CumWeight"] = df_items_sorted["Weight"].cumsum()
    df_items_sorted["CumProfit"] = df_items_sorted["Profit"].cumsum()
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df_items_sorted["CumWeight"], y=df_items_sorted["CumProfit"], mode="lines+markers", name="Cum Profit"))
    fig_line.update_layout(title="Cumulative Weight vs Cumulative Profit", xaxis_title="Cumulative Weight", yaxis_title="Cumulative Profit", height=300)
    st.plotly_chart(fig_line, use_container_width=True)

    # -------------------------
    # Download results CSV
    # -------------------------
    csv_buffer = StringIO()
    df_items.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()
    st.download_button("📥 Download Results (CSV)", data=csv_bytes, file_name="knapsack_results.csv", mime="text/csv")

# Footer with 🤖
st.markdown(
    """
    <br><hr>
    <div style='text-align:center;'>
    <span style='font-size:40px;'>🤖</span><br>
    <b style='font-size:18px;'>Smart Knapsack Optimizer</b><br>
    <span style='font-size:14px;'>Developed by Pragya Srivastava</span><br>
    <p style='font-size:12px;'>© 2025, GrowWise — AI powered Smart Knapsack Optimizer</p>
    </div>
    """,
    unsafe_allow_html=True
)
