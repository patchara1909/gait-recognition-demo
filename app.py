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

# ฟังก์ชันสำหรับประมวลผลภาพจากเว็บแคม
class VideoProcessor:
    def __init__(self):
        # กำหนดค่าเริ่มต้นตัวตรวจจับท่าทาง
        self.pose_detector = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # กลับด้านภาพเหมือนกระจก เพื่อความ 
        img = cv2.flip(img, 1)
        
        # แปลงสีเป็น RGB สำหรับประมวลผลด้วย MediaPipe
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose_detector.process(image_rgb)
        
        # แปลงกลับเป็น BGR สำหรับแสดงผลบน OpenCV
        image_rgb.flags.writeable = True
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # วาดโครงกระดูก (Pose Landmarks) ลงบนภาพ
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ส่วนแสดงผล WebRTC สำหรับเปิดกล้องสตรีมมิ่งเรียลไทม์
st.markdown("### 📹 เปิดกล้องเพื่อเริ่มการตรวจสอบ")
webrtc_streamer(
    key="gait-recognition",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)