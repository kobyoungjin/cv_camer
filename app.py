import streamlit as st
import pandas as pd
from kiwipiepy import Kiwi
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os
import platform

# 1. Kiwi 설정 및 폰트 체크
kiwi = Kiwi()


def get_font_path():
    if platform.system() == "Windows":
        return "C:/Windows/Fonts/malgun.ttf"
    else:  # Colab/Linux
        return "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"


FONT_PATH = get_font_path()


# 2. 데이터 로드
@st.cache_data
def load_data():
    if os.path.exists("news_data.csv"):
        return pd.read_csv("news_data.csv", encoding="utf-8-sig")
    return pd.DataFrame()


# 3. 키워드 추출 로직 (필터링 대폭 완화)
def extract_forced_keywords(df):
    # 컬럼 자동 감지
    cols = df.columns.tolist()
    content_candidates = [
        c for c in cols if any(k in c.lower() for k in ["cont", "본문", "내용", "text"])
    ]
    content_col = content_candidates[0] if content_candidates else cols[-1]

    full_text = " ".join(df[content_col].astype(str).tolist())

    # [핵심] 형태소 분석 태그를 거의 모든 실질형태소(명사, 동사, 형용사, 부사, 외국어)로 확대
    tokens = kiwi.tokenize(full_text)

    # N(명사), V(용언 어근), S(외국어/숫자), M(부사) 등 거의 다 포함
    allowed_tags = ["NNG", "NNP", "NNB", "VV", "VA", "SL", "SN", "MAG"]

    words = []
    for t in tokens:
        # 1글자 이상 모든 단어 추출 (단, 너무 흔한 조사는 제외)
        if t.tag in allowed_tags and len(t.form) >= 1:
            words.append(t.form)

    return words


# --- UI ---
st.title("🚀 키워드 무조건 10개 뽑기 프로젝트")

df = load_data()

if not df.empty:
    all_words = extract_forced_keywords(df)
    word_counts = Counter(all_words)

    # 상위 10개 추출
    top_10 = word_counts.most_common(10)

    if len(top_10) > 0:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("☁️ 워드클라우드 (분산 모드)")
            # 단어 크기 차이를 줄여서 골고루 보이게 함 (relative_scaling=0)
            wc = WordCloud(
                font_path=FONT_PATH,
                background_color="white",
                width=800,
                height=500,
                max_words=30,
                relative_scaling=0,  # 빈도 차이가 커도 단어 크기를 비슷하게 유지
            ).generate_from_frequencies(word_counts)

            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.subheader("📊 추출된 키워드 순위")
            st.table(pd.DataFrame(top_10, columns=["단어", "빈도"]))

        # 데이터가 너무 적을 경우 원문 확인용
        with st.expander("데이터 원문 확인"):
            st.write(df)
    else:
        st.error(
            "데이터에서 단어를 하나도 찾지 못했습니다. CSV 파일의 내용을 확인해주세요."
        )
else:
    st.error("news_data.csv 파일을 찾을 수 없습니다.")
