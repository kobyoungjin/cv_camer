import streamlit as st
import pandas as pd
from kiwipiepy import Kiwi
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os

# 1. 환경 설정
kiwi = Kiwi()
# 한글 폰트 경로 (Colab 나눔바른고딕)
# app.py 상단 수정
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

# 2. 확장된 불용어 리스트
KOREAN_STOPWORDS = {
    "것",
    "등",
    "및",
    "약",
    "또",
    "를",
    "수",
    "이",
    "그",
    "저",
    "더",
    "때",
    "중",
    "위",
    "뿐",
    "즉",
    "한",
    "할",
    "감",
    "곳",
    "기자",
    "뉴스",
    "보도",
    "지난",
    "올해",
    "관련",
    "대한",
    "통해",
    "이번",
    "현재",
    "최근",
    "오늘",
    "내년",
    "에서",
    "으로",
    "하는",
    "있는",
    "하고",
    "라고",
    "밝혔다",
    "전했다",
    "말했다",
    "무단",
    "전재",
    "배포",
    "금지",
    "저작권",
    "제보",
    "강조했다",
    "설명했다",
    "덧붙였다",
    "지적했다",
    "주장했다",
    "확인됐다",
    "예상된다",
    "전망된다",
    "알려졌다",
    "나타났다",
    "한편",
    "또한",
    "특히",
    "다만",
    "이외에도",
    "반면",
    "경우",
    "사실",
    "때문",
    "정도",
    "위해",
    "과정",
}


# 3. 데이터 로드 및 컬럼 체크 함수
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        # 인코딩 에러 방지를 위해 utf-8-sig 사용
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        return df
    else:
        # 파일이 없을 경우를 대비한 샘플 데이터
        data = {
            "category": ["정치", "경제", "IT", "정치", "경제", "IT"],
            "content": [
                "국회 법안 통과 소식입니다. 의원들이 강조했다.",
                "삼성전자 주가가 상승했습니다. 경제 지표가 나타났다.",
                "새로운 AI 모델이 공개되었습니다. 기술력을 강조했다.",
                "선거 결과가 발표되었습니다. 정치권이 설명했다.",
                "금리 인상이 전망된다. 시장 반응이 확인됐다.",
                "스마트폰 신제품 출시 소식입니다. IT 업계 소식입니다.",
            ],
        }
        return pd.DataFrame(data)


# 4. Kiwi 형태소 분석 함수
def extract_nouns(text):
    if not isinstance(text, str):
        return []
    tokens = kiwi.tokenize(text)
    # 명사(NNG, NNP) 추출 + 불용어 제거 + 2글자 이상
    return [
        t.form
        for t in tokens
        if t.tag in ["NNG", "NNP"]
        and t.form not in KOREAN_STOPWORDS
        and len(t.form) > 1
    ]


# --- UI 레이아웃 ---
st.set_page_config(page_title="뉴스 키워드 대시보드", layout="wide")
st.title("📊 뉴스 주제별 키워드 분석기")

df = load_data("news_data.csv")

if not df.empty:
    # 🔍 컬럼명 자동 매칭 (에러 방지 핵심 로직)
    # 'category'라는 단어가 포함된 컬럼을 찾고, 없으면 첫 번째 컬럼 선택
    cat_cols = [
        c for c in df.columns if "cat" in c.lower() or "주제" in c or "분류" in c
    ]
    category_col = cat_cols[0] if cat_cols else df.columns[0]

    # 'content'라는 단어가 포함된 컬럼을 찾고, 없으면 마지막 컬럼 선택
    cont_cols = [
        c for c in df.columns if "cont" in c.lower() or "본문" in c or "내용" in c
    ]
    content_col = cont_cols[0] if cont_cols else df.columns[-1]

    st.sidebar.success(f"선택된 카테고리 컬럼: {category_col}")
    st.sidebar.success(f"선택된 본문 컬럼: {content_col}")

    # 1) 주제 선택 드롭다운
    target_categories = sorted(df[category_col].unique())
    selected_cat = st.selectbox(
        "🎯 분석하고 싶은 뉴스 주제를 선택하세요", target_categories
    )

    # 데이터 필터링 및 분석
    filtered_df = df[df[category_col] == selected_cat]
    all_text = " ".join(filtered_df[content_col].astype(str).tolist())

    with st.spinner("키워드 분석 중..."):
        nouns = extract_nouns(all_text)
        word_counts = Counter(nouns)

    # 결과 시각화
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"💡 '{selected_cat}' 주요 키워드 시각화")
        if nouns:
            wc = WordCloud(
                font_path=FONT_PATH,
                background_color="white",
                width=800,
                height=500,
                colormap="viridis",
            ).generate_from_frequencies(word_counts)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("분석할 텍스트가 없습니다.")

    with col2:
        st.subheader("🔝 TOP 10 키워드")
        top_10 = pd.DataFrame(word_counts.most_common(10), columns=["키워드", "빈도수"])
        st.table(top_10)

    st.write("---")
    st.caption("데이터 전처리: KiwiPiePy | 시각화: WordCloud & Streamlit")
else:
    st.error("데이터프레임이 비어있습니다. CSV 파일을 확인해주세요.")
