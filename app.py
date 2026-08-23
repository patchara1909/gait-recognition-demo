import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av

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

class VideoProcessor:
    def __init__(self):
        self.pose_detector = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


def get_rtc_configuration():
    try:
        turn_url = str(st.secrets["TURN_URL"]).strip()
        turn_username = str(st.secrets["TURN_USERNAME"]).strip()
        turn_credential = str(st.secrets["TURN_CREDENTIAL"]).strip()
    except Exception:
        turn_url = ""
        turn_username = ""
        turn_credential = ""

    sample_values = ("your-server", "ชื่อผู้ใช้", "รหัสผ่าน")
    if not turn_url or any(value in turn_url + turn_username + turn_credential for value in sample_values):
        st.warning("กรุณาใส่ค่า TURN Server จริงใน Manage app > Settings > Secrets ก่อนเปิดกล้อง")
        return {"iceServers": []}

    return {
        "iceServers": [
            {
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_credential,
            }
        ]
    }


st.markdown("### 📹 กล้องตรวจสอบแบบเรียลไทม์")
st.caption("กด START เพื่อเปิดกล้อง และกด STOP เพื่อปิดกล้อง")
webrtc_streamer(
    key="gait-recognition",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    rtc_configuration=get_rtc_configuration(),
    media_stream_constraints={"video": True, "audio": False},
)