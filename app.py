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

def process_image(image):
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose_detector:
        results = pose_detector.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), bool(results.pose_landmarks)


st.markdown("### 📷 ถ่ายภาพเพื่อเริ่มการตรวจสอบ")
captured_image = st.camera_input("เปิดกล้อง")

if captured_image is not None:
    image_bytes = np.frombuffer(captured_image.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    processed_image, detected = process_image(image)
    st.image(processed_image, use_container_width=True)
    if detected:
        st.success("ตรวจพบโครงกระดูกแล้ว")
    else:
        st.warning("ยังไม่พบโครงกระดูก กรุณาถ่ายภาพใหม่")