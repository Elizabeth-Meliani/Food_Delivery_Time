import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load("xgb_model.pkl")

# Load dataset (untuk visualisasi distribusi)
df = pd.read_csv("Food_Delivery_Times.csv")

# Judul Dashboard
st.title("📦 Food Delivery Time")
st.header("🛠️ Input Operasional")

# Input User
distance = st.slider("Jarak Pengiriman (km)", 0.5, 20.0, 5.0)
prep_time = st.slider("Waktu Persiapan (menit)", 5, 60, 20)
experience = st.slider("Pengalaman Kurir (tahun)", 0.0, 10.0, 2.0)
weather = st.selectbox("Cuaca", ["Clear", "Rainy", "Foggy", "Snowy", "Windy"])
traffic = st.selectbox("Lalu Lintas", ["Low", "Medium", "High"])
time_of_day = st.selectbox("Waktu Pengantaran", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Jenis Kendaraan", ["Bike", "Car", "Scooter"])

# Prediksi
input_df = pd.DataFrame([{
    "Distance_km": distance,
    "Preparation_Time_min": prep_time,
    "Courier_Experience_yrs": experience,
    "Weather": weather,
    "Traffic_Level": traffic,
    "Time_of_Day": time_of_day,
    "Vehicle_Type": vehicle
}])

predicted_time = model.predict(input_df)[0]
st.subheader(f"⏱️ Estimasi Durasi Pengantaran: {int(predicted_time)} menit")

# Distribusi Waktu Pengantaran
st.header("📈 Distribusi Waktu Pengantaran")
fig, ax = plt.subplots()
sns.histplot(df["Delivery_Time_min"], bins=30, kde=True, color="salmon", edgecolor="black", ax=ax)
ax.axvline(predicted_time, color="blue", linestyle="--", label="Prediksi")
ax.legend()
st.pyplot(fig)

# Sidebar Profil
st.sidebar.header("Profile")
st.sidebar.markdown("**Name :** Elizabeth Meliani")
st.sidebar.markdown("**Email :** melzyunho@gmail.com")
st.sidebar.markdown("**Bio :** Data Scientist Learner")
