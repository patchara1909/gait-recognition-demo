import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="AI ระบบตรวจสอบท่าทางการเดินและจดจำบุคคล",
    layout="wide"
)

st.title("🚶‍♂️ AI ระบบตรวจสอบท่าทางการเดินและจดจำบุคคล")
st.write("ระบบเดโมสำหรับวิเคราะห์ท่าทางโครงกระดูกและบันทึกพฤติกรรมการเคลื่อนไหวแบบเรียลไทม์")

# โหลดโมเดล MediaPipe แบบปลอดภัย รองรับทั้งเวอร์ชันเก่าและใหม่
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    USE_LEGACY_SOLUTIONS = True
except AttributeError:
    USE_LEGACY_SOLUTIONS = False
    # Fallback กรณีที่ MediaPipe เวอร์ชันใหม่ไม่มี mp.solutions
    import mediapipe.python.solutions.pose as legacy_pose
    import mediapipe.python.solutions.drawing_utils as legacy_drawing
    mp_pose = legacy_pose
    mp_drawing = legacy_drawing

st.markdown("### 📷 กล้องสำหรับสาธิต")
st.caption("กด Take Photo เพื่อเปิดกล้องและถ่ายภาพ หรือกด Retake เพื่อถ่ายใหม่")
captured_image = st.camera_input("เปิดกล้อง")

if captured_image is not None:
    image_bytes = np.frombuffer(captured_image.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    image = cv2.flip(image, 1)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.success("กล้องทำงานแล้ว พร้อมใช้สาธิต")