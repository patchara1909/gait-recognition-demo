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
        return {
            "iceServers": [
                {
                    "urls": [st.secrets["TURN_URL"]],
                    "username": st.secrets["TURN_USERNAME"],
                    "credential": st.secrets["TURN_CREDENTIAL"],
                }
            ]
        }
    except Exception:
        return {
            "iceServers": [
                {
                    "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
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