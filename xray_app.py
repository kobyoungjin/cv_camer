import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image
import cv2
import io

# --- 페이지 설정 ---
st.set_page_config(page_title="X-Ray 폐렴 분류기", layout="wide")


# --- 모델 로드 (캐싱 처리) ---
@st.cache_resource
def load_pneumonia_model():
    # compile=False로 로드하면 속도가 빠르고 에러가 줄어듭니다. (시각화 전용이므로)
    model = load_model("pneumonia_model.h5", compile=False)
    # 레이어 이름이 정확한지 확인
    last_conv_name = [l.name for l in model.layers if "conv" in l.name.lower()][-1]
    return model, last_conv_name


def get_gradcam(model, img_array, last_conv_name):
    # 1. 마지막 Conv 레이어 찾기
    last_conv_layer = model.get_layer(last_conv_name)

    # 2. 모델의 입력부터 마지막 레이어까지 연결된 새로운 Functional 모델 생성
    # (Sequential 모델의 연결 끊김 문제를 원천 차단)
    inputs = tf.keras.Input(shape=(150, 150, 1))
    x = inputs
    # 모델의 모든 레이어를 순회하며 입력을 통과시킴
    for layer in model.layers:
        x = layer(x)
        # 만약 현재 레이어가 우리가 찾는 마지막 Conv 레이어라면 그 출력을 저장
        if layer.name == last_conv_name:
            conv_output = x

    # 새 모델: 입력 -> [마지막 Conv 출력, 최종 출력]
    grad_model = tf.keras.Model(inputs, [conv_output, x])

    # 3. 그레이디언트 계산
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # 이진 분류(Sigmoid) 기준
        class_channel = preds[:, 0]

    # 4. 특성 맵에 대한 그레이디언트 추출 (이제 None이 나오지 않습니다)
    grads = tape.gradient(class_channel, last_conv_layer_output)

    if grads is None:
        return np.zeros((7, 7))  # 안전장치 (데이터가 이상할 경우)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. 가중치 곱하기 및 히트맵 생성
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU 및 정규화
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


# --- UI 구성 ---
st.title("🩻 X-Ray 폐렴 분류 + Grad-CAM 시각화")
st.markdown(
    "이미지를 업로드하면 AI가 폐렴 여부를 판단하고 집중 분석 영역을 표시합니다."
)

# 사이드바 설정
st.sidebar.header("설정")
threshold = st.sidebar.slider("진단 임계값 (Threshold)", 0.3, 0.7, 0.5, 0.05)
st.sidebar.info(f"임계값 {threshold} 이상일 때 폐렴으로 진단합니다.")

uploaded_file = st.sidebar.file_uploader(
    "X-Ray 이미지 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. 이미지 로드 및 전처리
    raw_img = Image.open(uploaded_file).convert("L")  # 회색조 변환
    img_resized = raw_img.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=(0, -1))  # (1, 150, 150, 1)

    # 2. 예측 및 Grad-CAM 생성
    model, last_conv = load_pneumonia_model()
    prediction = model.predict(img_input)[0][0]
    heatmap = get_gradcam(model, img_input, last_conv)

    # 3. 레이아웃 (3열)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("원본 X-Ray")
        st.image(raw_img, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM 분석")
        # 히트맵 오버레이 생성
        heatmap_resized = cv2.resize(heatmap, (raw_img.size))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # 원본(RGB 변환 후)과 합성
        raw_rgb = np.array(raw_img.convert("RGB"))
        superimposed_img = cv2.addWeighted(raw_rgb, 0.6, heatmap_color, 0.4, 0)
        st.image(superimposed_img, use_container_width=True)

    with col3:
        st.subheader("예측 결과")
        is_pneumonia = prediction >= threshold
        label = "⚠️ 폐렴 (Pneumonia)" if is_pneumonia else "✅ 정상 (Normal)"
        color = "inverse" if is_pneumonia else "normal"

        st.metric(label="진단 결과", value=label)
        st.write(f"확률: {prediction*100:.2f}%")
        st.progress(float(prediction))

        # 상세 정보
        if is_pneumonia:
            st.error("폐렴 가능성이 높습니다. 전문의의 진찰이 권장됩니다.")
        else:
            st.success("정상 소견으로 보입니다.")

    # 4. 다운로드 버튼
    result_pil = Image.fromarray(superimposed_img)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button(
        label="분석 결과 이미지 다운로드",
        data=buf.getvalue(),
        file_name="xray_analysis_result.png",
        mime="image/png",
    )

else:
    st.info("왼쪽 사이드바에서 X-Ray 이미지를 업로드해주세요.")

# --- 푸터 ---
st.divider()
st.caption("본 앱은 AI 연구용이며 의료진의 최종 판단을 대체할 수 없습니다.")
