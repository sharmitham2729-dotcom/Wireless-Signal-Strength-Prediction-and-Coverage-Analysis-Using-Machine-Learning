import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Wireless Signal Virtual Lab", layout="wide")

# -----------------------------
# DATA + MODEL
# -----------------------------
@st.cache_data
def generate_data(n=3000):
    data = []
    for _ in range(n):
        d = np.random.uniform(1, 100)
        o = np.random.randint(0, 5)
        rssi = -30 - 10*np.log10(d) - o*5 + np.random.normal(0, 2)
        data.append([d, o, rssi])
    return pd.DataFrame(data, columns=['distance','obstacles','rssi'])

@st.cache_resource
def train_model(df):
    X = df[['distance','obstacles']]
    y = df['rssi']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    model = RandomForestRegressor(n_estimators=150)
    model.fit(X_train, y_train)
    return model

df = generate_data()
model = train_model(df)

# -----------------------------
# NAVIGATION
# -----------------------------
menu = st.sidebar.radio("Navigation", [
    "Aim", "Theory", "Experiment", "Quiz", "Feedback"
])

# -----------------------------
# AIM
# -----------------------------
if menu == "Aim":
    st.title("🎯 Aim")
    st.write("""
    To simulate wireless signal propagation and predict signal strength (RSSI)
    using machine learning based on distance and environmental obstacles.
    """)

# -----------------------------
# THEORY
# -----------------------------
elif menu == "Theory":
    st.title("📘 Theory")

    st.write("""
    Wireless signals attenuate with distance due to path loss.

    **Log-distance Path Loss Model:**
    RSSI = -30 - 10 log10(d) - obstacle_loss

    Where:
    - d = distance
    - obstacles reduce signal strength
    - noise simulates real-world interference
    """)

# -----------------------------
# EXPERIMENT
# -----------------------------
elif menu == "Experiment":
    st.title("🧪 Experiment")

    st.sidebar.header("Input Parameters")
    distance = st.sidebar.slider("Distance (m)", 1, 100, 20)
    obstacles = st.sidebar.slider("Obstacles", 0, 5, 2)

    prediction = model.predict([[distance, obstacles]])[0]

    st.subheader("📊 Prediction")
    st.write(f"Predicted RSSI: {prediction:.2f} dBm")

    # GRAPH
    st.subheader("📉 Distance vs RSSI")
    fig, ax = plt.subplots()
    ax.scatter(df['distance'], df['rssi'], alpha=0.4)
    ax.set_xlabel("Distance")
    ax.set_ylabel("RSSI")
    st.pyplot(fig)

    # HEATMAP
    st.subheader("🗺️ Coverage Heatmap")
    grid = 40
    x = np.linspace(1,100,grid)
    y = np.linspace(0,5,grid)
    Z = np.zeros((grid,grid))

    for i,d in enumerate(x):
        for j,o in enumerate(y):
            Z[j,i] = model.predict([[d,o]])[0]

    fig2, ax2 = plt.subplots()
    c = ax2.imshow(Z, extent=[1,100,0,5], origin='lower', aspect='auto')
    fig2.colorbar(c)
    st.pyplot(fig2)

    # -----------------------------
    # PDF GENERATION
    # -----------------------------
    def create_pdf(distance, obstacles, prediction):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        c = canvas.Canvas(temp_file.name, pagesize=letter)

        c.drawString(100, 750, "Wireless Signal Lab Report")
        c.drawString(100, 700, f"Distance: {distance} m")
        c.drawString(100, 680, f"Obstacles: {obstacles}")
        c.drawString(100, 660, f"Predicted RSSI: {prediction:.2f} dBm")

        c.save()
        return temp_file.name

    if st.button("📄 Download PDF Report"):
        pdf_path = create_pdf(distance, obstacles, prediction)
        with open(pdf_path, "rb") as f:
            st.download_button("Download", f, "report.pdf")

# -----------------------------
# QUIZ
# -----------------------------
elif menu == "Quiz":
    st.title("🧠 Quiz")

    q1 = st.radio("1. What affects signal strength most?",
                  ["Distance", "Color", "Sound"])

    if st.button("Submit Quiz"):
        if q1 == "Distance":
            st.success("Correct")
        else:
            st.error("Wrong")

# -----------------------------
# FEEDBACK
# -----------------------------
elif menu == "Feedback":
    st.title("💬 Feedback")

    name = st.text_input("Name")
    feedback = st.text_area("Your Feedback")

    if st.button("Submit"):
        st.success("Feedback submitted successfully!")
