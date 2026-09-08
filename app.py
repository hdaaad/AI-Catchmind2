import random
import time
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from streamlit_drawable_canvas import st_canvas

# 태블릿 화면은 데스크톱보다 좁고 세로로 긴 경우가 많아서,
# "wide"(가로로 넓게 쓰는) 레이아웃 대신 "centered"(가운데 정렬, 적당한 너비)를 써요.
# 그래야 내용이 화면 양 끝까지 늘어나지 않고 태블릿에서도 보기 좋게 나와요.
st.set_page_config(page_title="AI 캐치마인드", page_icon="🎨", layout="centered")

ROUND_LIMIT_SEC = 60           # 한 문제당 제한시간(초)
TOTAL_ROUNDS = 5               # 총 문제 수
CANVAS_WIDTH = 480             # 태블릿 화면 폭을 고려한 그림판 크기
CANVAS_HEIGHT = 360

# Gemini 모델 이름은 구글이 꽤 자주 바꿔요(새 모델 출시 -> 이전 모델 서비스 종료).
# 만약 나중에 또 "모델을 찾을 수 없다"는 에러가 나오면, 에러 메시지에 적힌
# 최신 모델 이름으로 이 한 줄만 바꿔주면 돼요.
GEMINI_MODEL = "gemini-3.6-flash"

CATEGORIES = {
    "동물": "🐶",
    "과일": "🍎",
    "채소": "🥕",
    "사물": "📎",
    "교통수단": "🚗",
}

# -----------------------------------------------------
# session_state 초기화
# -----------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "start"          # start / game / result
if "category" not in st.session_state:
    st.session_state.category = None
if "questions" not in st.session_state:
    st.session_state.questions = []          # 이번 게임에서 뽑힌 제시어 5개
if "round_idx" not in st.session_state:
    st.session_state.round_idx = 0
if "rounds" not in st.session_state:
    st.session_state.rounds = []             # 라운드별 결과 저장
if "round_start_time" not in st.session_state:
    st.session_state.round_start_time = None
if "_last_canvas_data" not in st.session_state:
    st.session_state._last_canvas_data = None


@st.cache_data
def load_keywords():
    """keyword.csv를 불러옵니다. (카테고리, 키워드) 두 개의 열이 있어야 해요."""
    return pd.read_csv("keyword.csv")


# -----------------------------------------------------
# Gemini(AI) 호출 관련
# -----------------------------------------------------
@st.cache_resource
def get_gemini_client():
    """Gemini API 클라이언트를 한 번만 만들어서 재사용합니다.
    st.secrets에 GEMINI_API_KEY가 없으면 None을 돌려줘서,
    앱 전체가 에러로 멈추지 않고 대신 안내 문구를 보여줄 수 있게 해요."""
    try:
        from google import genai
        api_key = st.secrets["GEMINI_API_KEY"]
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def ask_ai_guess(image: Image.Image, category: str) -> str:
    """카테고리와 그림을 Gemini에게 보내서, 한 단어로 된 정답 추측을 받아옵니다."""
    client = get_gemini_client()
    if client is None:
        return "(API 키 설정 필요)"

    prompt = (
        f"이 그림은 초등학생이 그린 캐치마인드(그림 퀴즈) 그림입니다. "
        f"이 그림의 카테고리는 반드시 '{category}'입니다. "
        f"어린이 그림이라 정교하지 않으니, 전체적인 '형태'와 '윤곽'에 집중해서 추론해주세요. "
        f"카테고리 '{category}'에 속하는 것들 중, 이 그림이 무엇을 그린 것인지 "
        f"설명이나 문장 없이 오직 '한 단어'로만 답해주세요. "
        f"예시 답변 형식: 사과 / 원숭이 / 연필"
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, image],
        )
        answer = (response.text or "").strip()
        if not answer:
            return "(답을 찾지 못했어요)"
        # 혹시 여러 단어나 문장으로 답이 와도, 첫 단어만 안전하게 뽑아내요.
        first_word = answer.split()[0]
        return first_word.strip(".,!?'\"()[]")
    except Exception as e:
        return f"(오류: {e})"


def canvas_array_to_pil(image_data) -> Image.Image:
    """캔버스가 돌려주는 (세로, 가로, 4) 형태의 RGBA 숫자 배열을
    흰 배경 위에 합성한 보통의 PIL 이미지로 바꿔줍니다."""
    rgba = Image.fromarray(image_data.astype("uint8"), mode="RGBA")
    white_bg = Image.new("RGB", rgba.size, "white")
    white_bg.paste(rgba, mask=rgba.split()[3])
    return white_bg


def blank_canvas_image() -> Image.Image:
    return Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")


def start_new_game(category: str):
    keywords_df = load_keywords()
    pool = keywords_df.loc[keywords_df["카테고리"] == category, "키워드"].tolist()
    n = min(TOTAL_ROUNDS, len(pool))
    st.session_state.category = category
    st.session_state.questions = random.sample(pool, n)
    st.session_state.round_idx = 0
    st.session_state.rounds = []
    st.session_state.round_start_time = None
    st.session_state._last_canvas_data = None
    st.session_state.page = "game"


# =======================================================
# 화면 1) 시작 화면 (카테고리 선택)
# =======================================================
def start_screen():
    st.title("🎨 AI 캐치마인드")
    st.write("카테고리를 골라서 시작해요! 그림을 그리면 AI가 맞혀볼 거예요.")
    st.write("")

    names = list(CATEGORIES.keys())
    for i in range(0, len(names), 3):
        row_names = names[i:i + 3]
        cols = st.columns(len(row_names))
        for col, name in zip(cols, row_names):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='text-align:center;font-size:44px;'>{CATEGORIES[name]}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='text-align:center;font-size:19px;font-weight:600;margin-bottom:8px;'>{name}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("시작하기", key=f"start_{name}", width="stretch"):
                        start_new_game(name)
                        st.rerun()


# =======================================================
# 화면 2) 게임 화면 (그림 그리기 + AI 정답 맞히기)
# =======================================================
def process_submission(image_data):
    """제출된 그림을 저장하고 AI에게 정답을 물어본 뒤, 다음 라운드로 넘어갑니다."""
    idx = st.session_state.round_idx
    keyword = st.session_state.questions[idx]

    snapshot = canvas_array_to_pil(image_data) if image_data is not None else blank_canvas_image()

    # 캔버스 자리에 방금 그린 그림을 그대로 고정해서 보여줘요.
    st.image(snapshot, width="stretch", caption="제출한 그림")

    with st.spinner("🤖 AI가 생각 중입니다..."):
        ai_answer = ask_ai_guess(snapshot, st.session_state.category)

    buf = BytesIO()
    snapshot.save(buf, format="PNG")
    st.session_state.rounds.append({
        "keyword": keyword,
        "image_bytes": buf.getvalue(),
        "ai_answer": ai_answer,
    })

    st.session_state.round_idx += 1
    st.session_state.round_start_time = None
    st.session_state._last_canvas_data = None
    st.rerun()


def game_screen():
    idx = st.session_state.round_idx

    if idx >= len(st.session_state.questions):
        st.session_state.page = "result"
        st.rerun()
        return

    keyword = st.session_state.questions[idx]

    if st.session_state.round_start_time is None:
        st.session_state.round_start_time = time.time()

    elapsed = time.time() - st.session_state.round_start_time
    remaining = max(0, ROUND_LIMIT_SEC - int(elapsed))
    time_is_up = remaining <= 0

    # ---- 상단 정보: 진행 상황 / 카테고리 / 제시어 / 남은 시간 ----
    st.markdown(
        f"<div style='text-align:center;font-size:15px;color:#888;'>"
        f"{st.session_state.category} · 문제 {idx + 1} / {len(st.session_state.questions)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='text-align:center;font-size:34px;font-weight:700;margin:4px 0 8px;'>"
        f"✏️ {keyword}</div>",
        unsafe_allow_html=True,
    )

    timer_color = "#D32F2F" if remaining <= 10 else "#333333"
    st.markdown(
        f"<div style='text-align:center;font-size:22px;font-weight:600;color:{timer_color};margin-bottom:10px;'>"
        f"⏱️ 남은 시간: {remaining}초</div>",
        unsafe_allow_html=True,
    )

    if not time_is_up:
        # 아직 시간이 남았으면: 1초마다 화면을 새로고침해서 시간이 흐르는 걸 반영해요.
        st_autorefresh(interval=1000, limit=ROUND_LIMIT_SEC + 5, key=f"timer_{idx}")

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=8,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            drawing_mode="freedraw",
            update_streamlit=True,
            display_toolbar=True,
            key=f"canvas_{idx}",
        )

        # 지금까지 그린 그림을 계속 저장해둬요.
        # -> 시간이 갑자기 끝나도, 마지막으로 그려진 그림을 바로 꺼내 쓸 수 있어요.
        if canvas_result.image_data is not None:
            st.session_state._last_canvas_data = canvas_result.image_data

        submit_clicked = st.button("제출하기 ✅", type="primary", width="stretch")
    else:
        st.warning("⏰ 시간이 다 됐어요! 마지막 그림으로 자동 제출할게요.")
        submit_clicked = False

    if submit_clicked or time_is_up:
        process_submission(st.session_state._last_canvas_data)


# =======================================================
# 화면 3) 결과 화면
# =======================================================
def result_screen():
    st.title("🏆 게임 결과")
    st.subheader(f"카테고리: {st.session_state.category}")
    st.write("")

    correct_count = 0
    for i, r in enumerate(st.session_state.rounds, start=1):
        is_correct = r["ai_answer"].strip() == r["keyword"].strip()
        if is_correct:
            correct_count += 1

        with st.container(border=True):
            c1, c2 = st.columns([1, 1.4])
            with c1:
                st.image(r["image_bytes"], width="stretch")
            with c2:
                st.markdown(f"**{i}번 문제**")
                st.write(f"정답: **{r['keyword']}**")
                st.write(f"AI의 대답: **{r['ai_answer']}**")
                st.write("✅ 정답이에요!" if is_correct else "❌ 아쉬워요")

    st.write("")
    st.metric("맞힌 개수", f"{correct_count} / {len(st.session_state.rounds)}")

    st.write("")
    if st.button("처음으로 돌아가기", width="stretch"):
        st.session_state.page = "start"
        st.session_state.category = None
        st.session_state.questions = []
        st.session_state.round_idx = 0
        st.session_state.rounds = []
        st.session_state.round_start_time = None
        st.session_state._last_canvas_data = None
        st.rerun()


# =======================================================
# 라우팅
# =======================================================
if st.session_state.page == "start":
    start_screen()
elif st.session_state.page == "game":
    game_screen()
elif st.session_state.page == "result":
    result_screen()
