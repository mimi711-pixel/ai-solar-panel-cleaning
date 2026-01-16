import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Solar Panel Cleaning Optimizer (UAE)", layout="centered")

st.title("🌞 AI Solar Panel Cleaning Optimizer (UAE)")
st.write("Predict the optimal time to clean solar panels under UAE desert conditions (synthetic demo data).")

# ---- Sidebar inputs ----
st.sidebar.header("Sensor / Environment Inputs")
dust = st.sidebar.slider("Dust Level (1=Low, 3=High)", 1, 3, 2)
wind = st.sidebar.slider("Wind Speed (km/h)", 3, 15, 8)
temp = st.sidebar.slider("Temperature (°C)", 30, 45, 38)
days_since = st.sidebar.slider("Days Since Last Cleaning", 0, 20, 7)

st.sidebar.markdown("---")
st.sidebar.caption("Model is trained on synthetic data (demo). For production, replace with measured data.")

# ---- Data generation and training (cached) ----
@st.cache_data(show_spinner=False)
def generate_synthetic_data(seed=1, days=180):
    np.random.seed(seed)
    data = []
    days_clean = 0
    for day in range(1, days + 1):
        dust_l = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        wind_s = np.random.randint(3, 16)
        temp_c = np.random.randint(30, 46)
        days_clean += 1

        # Synthetic "efficiency loss" (same formula as your notebook)
        eff = dust_l * 6 + days_clean * 0.8 - wind_s * 0.3 + np.random.normal(0, 1)
        eff = max(0, eff)

        # Action labels: 0 = No Cleaning, 1 = Clean Soon, 2 = Clean Now
        action = 0 if eff < 8 else 1 if eff < 18 else 2
        if action == 2:
            days_clean = 0

        data.append([dust_l, wind_s, temp_c, days_clean, eff, action])

    df = pd.DataFrame(data, columns=['Dust', 'Wind', 'Temp', 'Days', 'Efficiency', 'Action'])
    return df

@st.cache_data(show_spinner=False)
def train_models(df):
    X = df[['Dust', 'Wind', 'Temp', 'Days']]
    y = df['Action']
    y_eff = df['Efficiency']

    clf = DecisionTreeClassifier(max_depth=5, random_state=1)
    clf.fit(X, y)

    reg = LinearRegression()
    reg.fit(X, y_eff)

    return clf, reg

df = generate_synthetic_data()
clf, reg = train_models(df)

# ---- Prediction ----
input_df = pd.DataFrame([[dust, wind, temp, days_since]], columns=['Dust', 'Wind', 'Temp', 'Days'])

if st.button("Predict Cleaning"):
    action = int(clf.predict(input_df)[0])
    eff_pred = float(reg.predict(input_df)[0])

    # probabilities for each class (0,1,2)
    probs = clf.predict_proba(input_df)[0]
    actions = {0: "🟢 No Cleaning Needed", 1: "🟠 Clean Soon", 2: "🔴 Clean Now"}

    st.subheader("AI Recommendation")
    st.markdown(f"**{actions[action]}**")
    st.markdown(f"- Predicted efficiency loss: **{eff_pred:.2f}** (synthetic units / %)")
    st.markdown(f"- Model confidence: **{probs[action]*100:.1f}%**")

    # Show class probabilities
    prob_df = pd.DataFrame({
        "Action": ["No Cleaning", "Clean Soon", "Clean Now"],
        "Probability": probs
    })
    st.table(prob_df.style.format({"Probability": "{:.2%}"}))

    # Show decision tree rules (short)
    st.subheader("Decision Tree Rules (excerpt)")
    tree_text = export_text(clf, feature_names=list(input_df.columns))
    st.code(tree_text, language="")

    # Simple explanation: top features
    st.subheader("Feature importances")
    fi = pd.Series(clf.feature_importances_, index=input_df.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    fi.plot(kind="bar", ax=ax)
    ax.set_ylabel("Importance")
    st.pyplot(fig)

# ---- Data preview and diagnostics ----
with st.expander("View synthetic training data (first 20 rows)"):
    st.dataframe(df.head(20))

with st.expander("Model diagnostics"):
    st.write("Decision tree max depth: 5 (demo). Consider replacing synthetic data with real sensor logs.")
    st.write(f"Training samples: {len(df)}")
    st.write("Action thresholds (used when generating synthetic labels): eff < 8 -> No Cleaning, 8 <= eff < 18 -> Clean Soon, eff >= 18 -> Clean Now")

st.markdown("---")
st.caption("Note: This is a demo using synthetic data. For production use, collect historical panel output and cleaning logs and retrain the models.")
