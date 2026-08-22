import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Gait & Face Recognition Demo", layout="centered")

st.title("🚶‍♂️ AI ระบบตรวจสอบท่าทางการเดินและจดจำบุคคล")
st.markdown("ระบบเดโม่สำหรับวิเคราะห์ท่าทางโครงกระดูกและบันทึกพฤติกรรมการเคลื่อนไหวแบบเรียลไทม์")

# โหลด MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# คลาสประมวลผลวิดีโอผ่านเว็บแคม
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # กลับด้านกระจก
        h, w, _ = img.shape

        # แปลงเป็น RGB เพื่อให้ MediaPipe ประมวลผล
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        gait_status = "Detecting..."
        color = (255, 255, 255)

        if results.pose_landmarks:
            # วาดโครงกระดูกลงบนภาพ
            mp_drawing.draw_landmarks(
                img, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            gait_status = "Normal Gait (Mock)"
            color = (0, 255, 0)

        # แสดงข้อความผลลัพธ์บนวิดีโอ
        cv2.putText(img, f"Status: {gait_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, "Name: Thonthan (Demo)", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# เปิดใช้งานกล้องผ่าน Streamlit WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="gait-detection",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

st.markdown("---")
st.info("💡 **วิธีทดสอบ:** กดปุ่ม **START** ด้านบนเพื่อเปิดกล้อง อนุญาตการเข้าถึงกล้องในเบราว์เซอร์ แล้วยืนหน้ากล้องเพื่อให้ AI จับโครงกระดูกได้ทันทีครับ")