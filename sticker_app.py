import streamlit as st
import cv2
import numpy as np
import os
from openvino.runtime import Core
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- 1. OpenVINO 모델 로드 (캐싱) ---
@st.cache_resource
def load_face_detector(model_path):
    ie = Core()
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    return compiled_model

# --- 2. 스티커 합성 함수 (알파 채널 대응 및 예외 처리) ---
def apply_sticker(img, sticker_orig, x1, y1, w, h, offset_x, offset_y, scale):
    try:
        # 스티커 크기 결정 (얼굴 너비 기준 + 스케일 적용)
        sw = int(w * scale)
        if sw <= 0: return img
        sh = int(sticker_orig.shape[0] * (sw / sticker_orig.shape[1]))
        
        # 위치 계산 (중앙 정렬 + 오프셋 적용)
        target_x = x1 + (w - sw) // 2 + offset_x
        target_y = y1 - (sh // 2) + offset_y 
        
        # 스티커 리사이즈
        sticker = cv2.resize(sticker_orig, (sw, sh))
        img_h, img_w = img.shape[:2]
        
        # 알파 채널 유무 확인 (RGBA vs RGB)
        has_alpha = sticker.shape[2] == 4

        for i in range(sh):
            for j in range(sw):
                cur_y, cur_x = target_y + i, target_x + j
                if 0 <= cur_y < img_h and 0 <= cur_x < img_w:
                    if has_alpha:
                        alpha = sticker[i, j, 3] / 255.0
                        if alpha > 0:
                            for c in range(3):
                                img[cur_y, cur_x, c] = \
                                    sticker[i, j, c] * alpha + img[cur_y, cur_x, c] * (1 - alpha)
                    else:
                        img[cur_y, cur_x, :] = sticker[i, j, :3]
    except Exception as e:
        pass # 리사이즈 오류 등 방지
    return img

# --- 3. 실시간 영상 처리 클래스 ---
class FaceStickerTransformer(VideoTransformerBase):
    def __init__(self):
        # 모델 경로 설정 (본인 환경에 맞게 수정)
        self.model_path = "./models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
        self.detector = load_face_detector(self.model_path)
        
        # 실시간 제어용 파라미터 초기화
        self.sticker_img = None
        self.conf = 0.4
        self.off_x = 0
        self.off_y = 0
        self.scale = 1.0

    def update_params(self, sticker_img, conf, off_x, off_y, scale):
        # 사이드바의 값을 실시간으로 클래스 멤버 변수에 업데이트
        self.sticker_img = sticker_img
        self.conf = conf
        self.off_x = off_x
        self.off_y = off_y
        self.scale = scale

    def transform(self, frame):
        # 프레임을 넘파이 배열(BGR)로 변환
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # OpenVINO 추론 (672x384 입력 사이즈 고정)
        input_tensor = cv2.resize(img, (672, 384)).transpose((2, 0, 1))[np.newaxis, ...]
        out = self.detector([input_tensor])[self.detector.output(0)]
        detections = out[0][0]

        for detection in detections:
            confidence = detection[2]
            if confidence > self.conf:
                # 좌표 복원
                x1 = int(detection[3] * w)
                y1 = int(detection[4] * h)
                x2 = int(detection[5] * w)
                y2 = int(detection[6] * h)
                fw, fh = x2 - x1, y2 - y1

                # 스티커 적용
                if self.sticker_img is not None:
                    img = apply_sticker(img, self.sticker_img, x1, y1, fw, fh, 
                                        self.off_x, self.off_y, self.scale)
        
        return img

# --- 4. Streamlit UI 구성 ---
st.set_page_config(page_title="얼굴 스티커 앱", layout="wide")
st.title("🎥 실시간 얼굴 스티커 카메라")

# 사이드바 컨트롤러
st.sidebar.header("⚙️ 스티커 컨트롤러")
sticker_choice = st.sidebar.radio("스티커 선택", ["선글라스", "모자", "수염", "없음"])
off_x = st.sidebar.slider("X 이동 (좌우)", -100, 100, 0)
off_y = st.sidebar.slider("Y 이동 (상하)", -100, 100, 0)
scale = st.sidebar.slider("크기 비율", 0.1, 3.0, 1.2)
conf_threshold = st.sidebar.slider("인식 민감도", 0.1, 0.9, 0.4)

# 스티커 이미지 로드
STICKER_FILES = {
    "선글라스": "sticker_glasses.png",
    "모자": "sticker_hat.png",
    "수염": "sticker_beard.png"
}

current_sticker = None
if sticker_choice != "없음":
    s_path = STICKER_FILES[sticker_choice]
    if os.path.exists(s_path):
        current_sticker = cv2.imread(s_path, cv2.IMREAD_UNCHANGED)
    else:
        st.sidebar.warning(f"⚠️ {s_path} 파일이 없습니다.")

# --- 5. WebRTC 스트리머 실행 및 실시간 파라미터 주입 ---
ctx = webrtc_streamer(
    key="face-sticker-live",
    video_transformer_factory=FaceStickerTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# 카메라가 켜져 있고 트랜스포머가 생성된 상태라면 값을 실시간으로 밀어넣음
if ctx.video_transformer:
    ctx.video_transformer.update_params(
        sticker_img=current_sticker,
        conf=conf_threshold,
        off_x=off_x,
        off_y=off_y,
        scale=scale
    )

st.info("💡 사이드바의 슬라이더를 움직여보세요. 실시간으로 스티커가 조절됩니다!")