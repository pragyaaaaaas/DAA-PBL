 import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

st.set_page_config(page_title="ML Feature Selection | Knapsack Optimization", layout="wide")

# 🧮 Evaluate model
def model_score(model, X, y):
    y_pred = model.predict(X)
    return f1_score(y, y_pred, average='weighted')


# 🎒 Knapsack feature selector
def knapsack_select(features, importances, budget):
    # cost = 1 per feature
    ratio = [(f, imp/1) for f, imp in zip(features, importances)]
    ratio.sort(key=lambda x: x[1], reverse=True)
    
    selected = []
    total_cost = 0
    
    for f, score in ratio:
        if total_cost + 1 <= budget:
            selected.append(f)
            total_cost += 1
            
    return selected


# 🚀 Streamlit UI
st.title("⚙️ Feature Selection for ML Models using Knapsack Optimization")

uploaded = st.file_uploader("Upload Dataset (CSV)", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("### 🧾 Preview Data", df.head())
    
    target = st.selectbox("Select Target Column", df.columns)
    
    X = df.drop(columns=[target])
    y = df[target]

    budget = st.slider("Select Feature Budget", min_value=1, max_value=len(X.columns), value=3)

    if st.button("Run Feature Selection"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        full_score = model_score(model, X_test, y_test)

        # Feature importance
        importances = model.feature_importances_
        features = list(X.columns)

        selected_features = knapsack_select(features, importances, budget)

        st.write("### ✅ Selected Features:", selected_features)

        model.fit(X_train[selected_features], y_train)
        reduced_score = model_score(model, X_test[selected_features], y_test)

        st.write(f"🎯 **Full Model Score (all features)**: **{full_score:.3f}**")
        st.write(f"🎯 **Reduced Model Score (selected features)**: **{reduced_score:.3f}**")

        # Display importance bar chart
        st.bar_chart(pd.DataFrame({"Importance": importances}, index=features))

        st.download_button(
            label="📥 Download Selected Feature List",
            data="\n".join(selected_features),
            file_name="selected_features.txt"
        )

else:
    st.info("Upload the provided CSV file to begin.")

