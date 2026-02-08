import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
import joblib
import numpy as np
from groq import Groq
import os
import altair as alt
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Plant Growth System", layout="wide")

# -----effect and color--------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2e7d32, #66bb6a);
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white;
}

/* Title */
h1, h2, h3 {
    color: #2e7d32 !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#43a047,#66bb6a);
    color: lightgray;
    border-radius: 12px;
    border: none;
    padding: 0.6em 1.2em;
    transition: 0.3s ease;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}

.stButton>button:hover {
    background: linear-gradient(90deg,#2e7d32,#43a047);
    transform: scale(1.05);
}

/* Inputs */
input, textarea {
    border-radius: 10px !important;
}

/* Cards / dataframe */
.css-1d391kg, .stDataFrame {
    background: rgba(255,255,255,0.75);
    border-radius: 15px;
    padding: 10px;
}

/* Success box */
.stAlert {
    border-radius: 12px;
}

/* Webcam frame */
video {
            
    border-radius: 20px;
    box-shadow: 0 0 15px rgba(0,150,0,0.4);
}

/* Sliders */
.stSlider > div {
    color: green;
}

/* Calm animation */
@keyframes float {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-4px);}
    100% {transform: translateY(0px);}
}

h1 {
    animation: float 4s ease-in-out infinite;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="position:fixed;bottom:20px;right:20px;font-size:30px;">🌱</div>
""", unsafe_allow_html=True)


# ---------------- DATABASE ----------------
conn = sqlite3.connect("plants.db", timeout=30, check_same_thread=False)
cursor = conn.cursor()

# USERS
cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT UNIQUE,
password BLOB,
role TEXT
)
""")

# PLANTS        
cursor.execute("""
CREATE TABLE IF NOT EXISTS plants(
plant TEXT PRIMARY KEY,
info TEXT
)
""")

# ENTRIES
cursor.execute("""
CREATE TABLE IF NOT EXISTS entries(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT,
plant TEXT,
temperature REAL,
humidity REAL,
moisture REAL,
prediction REAL
)
""")


conn.commit()

# Insert plant info ONCE
cursor.execute("SELECT COUNT(*) FROM plants")
if cursor.fetchone()[0] == 0:
    plants_data = [
        ("Tomato", "Needs full sun, moderate watering."),
        ("Potato", "Cool weather crop, loose soil."),
        ("Rice", "High water requirement, warm climate."),
        ("Wheat", "Dry climate, well-drained soil."),
        ("Corn", "Warm weather, fertile soil."),
    ]

    cursor.executemany(
        "INSERT INTO plants VALUES (?,?)",
        plants_data
    )
    conn.commit()



# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------- HELPERS ----------------
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), bytes(hashed))

def signup_user(username, password, role):
    hashed = hash_password(password)
    try:
        cursor.execute(
            "INSERT INTO users(username,password,role) VALUES (?,?,?)",
            (username, hashed, role)
        )
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    cursor.execute("SELECT password, role FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    if row and verify_password(password, row[0]):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = row[1]
        return True
    return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = None
    st.rerun()


# ---------------- LOGIN ----------------
if not st.session_state.logged_in:

    st.title("🔐 Smart Plant System")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(u, p):
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        su = st.text_input("New Username")
        sp = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["user", "admin"])

        if st.button("Signup"):
            if su and sp:
                if signup_user(su, sp, role):
                    st.success("Signup successful. Login now.")
                else:
                    st.error("Username exists")
            else:
                st.warning("Fill all fields")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.success(f"{st.session_state.username} ({st.session_state.role})")
st.sidebar.button("Logout", on_click=logout)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))

# ---------------- GROQ ----------------
api_key = os.getenv("api_key")
client = Groq(api_key=api_key)

# ---------------- YOLO VIDEO PROCESSOR ----------------
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = yolo_model(img, conf=0.4)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_model.names[cls]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- MAIN ----------------
st.title("🌱 Smart Plant Growth Prediction System")

plant = st.text_input("Plant Name")
temp = st.number_input("Temperature", value=25.0)
humidity = st.number_input("Humidity", value=60.0)
moisture = st.number_input("Soil Moisture", value=40.0)
if st.button("Predict Growth"):

    X = np.array([[temp, humidity, moisture]])
    pred = model.predict(X)[0]

    cursor.execute("SELECT info FROM plants WHERE plant=?", (plant,))
    info = cursor.fetchone()
    info = info[0] if info else "General care required"

    cursor.execute("""
    INSERT INTO entries(username,plant,temperature,humidity,moisture,prediction)
    VALUES (?,?,?,?,?,?)
    """,(st.session_state.username,plant,temp,humidity,moisture,float(pred)))

    conn.commit()

    st.success(f"Predicted Growth: {round(pred,2)}")
    st.info(info)

# ---------------- TABLE ----------------
st.subheader("📋 All Predictions")

df = pd.read_sql("SELECT * FROM entries ORDER BY id DESC", conn)
st.dataframe(df,use_container_width=True)

# ---------------- CHARTS ----------------
if not df.empty:

    # -------- BAR CHART (Average Growth per Plant) --------
    avg_df = df.groupby("plant", as_index=False)["prediction"].mean()

    bar = alt.Chart(avg_df).mark_bar(
        cornerRadiusTopLeft=10,
        cornerRadiusTopRight=10
    ).encode(
        x=alt.X("plant:N", title="Plant"),
        y=alt.Y("prediction:Q", title="Average Growth"),
        color=alt.value("#43a047")  # calm green
    ).properties(
        height=350,
        title="🌱 Average Growth by Plant"
    )

    st.altair_chart(bar, use_container_width=True)

    # -------- LINE CHART (Prediction Trend) --------
    line = alt.Chart(df).mark_line(
        interpolate="monotone",
        strokeWidth=3
    ).encode(
        x=alt.X("id:Q", title="Entry ID"),
        y=alt.Y("prediction:Q", title="Growth Prediction"),
        color=alt.value("#2e7d32")  # darker green
    ).properties(
        height=300,
        title="📈 Growth Prediction Trend"
    )

    points = alt.Chart(df).mark_circle(size=60, color="#66bb6a").encode(
        x="id:Q",
        y="prediction:Q"
    )

    st.altair_chart(line + points, use_container_width=True)

    # ---------------- WEBCAM ----------------
st.subheader("📷 Plant Detection (Webcam)")

webrtc_streamer(
    key="plant",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)


# ---------------- AI ----------------
st.subheader("🤖 Ask Plant AI")

# store answer
if "ai_answer" not in st.session_state:
    st.session_state.ai_answer = ""

q = st.text_input("Ask question")

if st.button("Ask AI"):

    if not q:
        st.warning("Please enter a question")
    else:
        try:
            with st.spinner("Thinking... 🌱"):

                chat = client.chat.completions.create(
                    model="groq/compound",
                    messages=[
                        {"role": "system", "content": "You are an agriculture and plant growth expert."},
                        {"role": "user", "content": q}
                    ]
                )

                st.session_state.ai_answer = chat.choices[0].message.content

        except Exception as e:
            st.error(e)

# show answer AFTER rerun
if st.session_state.ai_answer:
    st.success(st.session_state.ai_answer)


# ---------------- ADMIN ----------------
if st.session_state.role == "admin":

    st.subheader("👑 Admin Panel")

    users = pd.read_sql("SELECT id,username,role FROM users", conn)
    st.table(users)

    del_user = st.text_input("Delete Username")

    if st.button("Delete User"):
        cursor.execute("DELETE FROM users WHERE username=?", (del_user,))
        conn.commit()
        st.rerun()

      




