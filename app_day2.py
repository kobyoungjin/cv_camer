import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kiwipiepy import Kiwi
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
import joblib  # 모델 파일을 불러오기 위한 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리 (np라는 별칭 사용)

# 한글 폰트 설정 (Windows의 경우 맑은 고딕)
import platform

# 운영체제별 폰트 설정
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # Mac
    plt.rc("font", family="AppleGothic")
else:  # Linux (Colab 등)
    plt.rc("font", family="NanumBarunGothic")

# 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

# 1. 초기 설정 및 모델 로드 (캐싱)
kiwi = Kiwi()


@st.cache_resource
def load_models():
    # TF-IDF 모델 (가상의 학습 데이터로 간단히 구현 예시)
    # 실제 프로젝트에서는 joblib.load('model.pkl') 형식을 권장합니다.
    vectorizer = TfidfVectorizer()
    model_tfidf = LogisticRegression()

    # DL 모델 (HuggingFace의 한국어 감성 분석 모델)
    pipe_dl = pipeline(
        "text-classification", model="daekeun-ml/koelectra-small-v3-nsmc"
    )
    return vectorizer, model_tfidf, pipe_dl


vectorizer, model_tfidf, pipe_dl = load_models()


def preprocess_korean(text):
    tokens = kiwi.tokenize(text)
    return " ".join(
        [t.form for t in tokens if t.tag.startswith("N") or t.tag.startswith("V")]
    )


# 2. UI 구성
st.set_page_config(page_title="영화 리뷰 감성 분석기", layout="wide")
st.title("🎬 한국어 영화 리뷰 감성 분석")
st.markdown("리뷰를 입력하면 AI가 긍정/부정을 분석해 드립니다.")

# 사이드바: 모델 선택
with st.sidebar:
    st.header("⚙️ 설정")
    model_choice = st.radio(
        "분석 모델 선택",
        ("TF-IDF + Logistic Regression", "Deep Learning (Transformer)"),
    )

# 텍스트 입력란
user_input = st.text_area(
    "리뷰 내용을 입력하세요:", placeholder="이 영화 정말 감동적이었어요!", height=150
)

# 3. 분석 로직 및 결과 표시
if st.button("🔍 분석 시작"):
    if user_input.strip() == "":
        st.warning("내용을 입력해 주세요.")
    else:
        col1, col2 = st.columns([1, 1])

        # 가상 분석 데이터 생성 (실제 구현 시 모델 예측값 대입)
        if "TF-IDF" in model_choice:
            # 예시를 위한 더미 점수 (실제 구현 시 model.predict_proba 사용)
            prob = 0.85
            label = "긍정"

            with col1:
                st.subheader("📊 분석 결과")
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": f"예측 결과: {label}"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {
                                "color": "#2ecc71" if label == "긍정" else "#e74c3c"
                            },
                            "steps": [{"range": [0, 50], "color": "lightgray"}],
                        },
                    )
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                st.subheader("💡 판단 근거 (Feature Importance)")
                # 시각화 데이터 예시
                data = {
                    "단어": ["감동", "최고", "연기력", "스토리", "아쉬움"],
                    "기여도": [0.45, 0.32, 0.21, 0.15, -0.12],
                }
                df_imp = pd.DataFrame(data).sort_values(by="기여도", ascending=True)

                fig_bar, ax = plt.subplots()
                colors = ["#e74c3c" if x < 0 else "#2ecc71" for x in df_imp["기여도"]]
                ax.barh(df_imp["단어"], df_imp["기여도"], color=colors)
                ax.set_yticklabels(
                    df_imp["단어"], fontproperties="Malgun Gothic"
                )  # 여기서 다시 지정
                st.pyplot(fig_bar)

        else:
            # Transformer 모델 결과
            result = pipe_dl(user_input)[0]
            label = "긍정" if result["label"] == "LABEL_1" else "부정"
            score = result["score"]

            with col1:
                st.subheader("📊 분석 결과")
                st.info(
                    f"딥러닝 모델 분석 결과 **{label}**일 확률이 **{score:.2%}**입니다."
                )

# 4. 하단 예시 리뷰 버튼
st.divider()
st.subheader("📝 예시 리뷰로 테스트해 보세요")
examples = [
    "간만에 본 최고의 명작입니다. 연출이 미쳤어요!",
    "돈 주고 보기 아까운 영화.. 시간 낭비했습니다.",
    "배우들 연기는 좋은데 개연성이 너무 떨어지네요.",
    "중간에 좀 지루하긴 한데 결말은 나쁘지 않음.",
    "아이들이랑 보기 딱 좋은 따뜻한 가족 영화예요.",
]

cols = st.columns(5)
for i, ex in enumerate(examples):
    if cols[i].button(f"예시 {i+1}"):
        # 실제 운영 환경에서는 세션 스테이트를 사용하여 input 값을 변경합니다.
        st.info(f"선택된 리뷰: {ex}")
