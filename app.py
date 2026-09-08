import random
import time
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from streamlit_drawable_canvas import st_canvas

# 태블릿 화면은 데스크톱보다 좁은 경우가 많아서,
# "wide" 대신 "centered"(가운데 정렬, 적당한 너비)를 써요.
st.set_page_config(page_title="AI 캐치마인드", page_icon="🎨", layout="centered")

ROUND_LIMIT_SEC = 60            # 한 문제당 제한시간(초)
TOTAL_ROUNDS = 5                # 총 문제 수 (패스는 이 개수에 포함되지 않음)
CANVAS_WIDTH = 620              # 그림판 가로 길이 (더 넓게)
CANVAS_HEIGHT = 360
DISPLAY_IMG_WIDTH = 380         # 그려진 그림을 보여줄 때의 고정 크기 (확대되어 보이지 않도록)
THUMB_IMG_WIDTH = 150           # 결과 화면 목록에서 쓰는 작은 썸네일 크기

# Gemini는 버전 이름(2.5-flash, 3.6-flash 등)을 붙이면 언젠가 서비스 종료가 돼요.
# "gemini-flash-latest"는 특정 버전이 아니라 "지금 시점의 최신 flash 모델"을 항상
# 자동으로 가리키는 별명이라서, 구글이 모델을 바꿔도 이 이름 자체는 계속 유효해요.
# 일단 안정적으로 돌아가는 게 우선이니 이 하나만 쓰도록 단순화했어요.
GEMINI_MODEL = "gemini-flash-latest"

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
    st.session_state.page = "start"          # start / game / grading / result
if "category" not in st.session_state:
    st.session_state.category = None
if "keyword_pool" not in st.session_state:
    st.session_state.keyword_pool = []       # 이번 카테고리 단어를 섞어둔 목록
if "keyword_pointer" not in st.session_state:
    st.session_state.keyword_pointer = 0
if "current_item" not in st.session_state:
    st.session_state.current_item = None     # 지금 라운드의 {카테고리, 키워드, 유사정답}
if "round_idx" not in st.session_state:
    st.session_state.round_idx = 0           # 실제로 "제출"한 문제 수 (패스는 안 셈)
if "rounds" not in st.session_state:
    st.session_state.rounds = []
if "round_start_time" not in st.session_state:
    st.session_state.round_start_time = None
if "_last_canvas_data" not in st.session_state:
    st.session_state._last_canvas_data = None
if "draw_seq" not in st.session_state:
    st.session_state.draw_seq = 0  # 그림판을 새로 만들 때마다 하나씩 늘어나는 고유 번호


@st.cache_data
def load_keywords():
    """keyword.csv를 불러옵니다. (카테고리, 키워드, 유사정답) 세 개의 열이 있어야 해요."""
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
    """카테고리와 그림을 Gemini에게 보내서, 한 단어로 된 정답 추측을 받아옵니다.
    구글 서버가 일시적으로 붐빌 때(503)는 잠깐 쉬었다가 한 번 더 시도해봐요."""
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

    max_attempts = 2  # 503(서버 과부하)일 때 한 번 더 시도해보기 위한 횟수
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, image],
            )
            answer = (response.text or "").strip()
            if not answer:
                return "(답을 찾지 못했어요)"
            first_word = answer.split()[0]
            return first_word.strip(".,!?'\"()[]")
        except Exception as e:
            msg = str(e)
            is_overloaded = "UNAVAILABLE" in msg or "503" in msg
            if is_overloaded and attempt < max_attempts - 1:
                time.sleep(2)  # 서버가 붐비는 건 보통 금방 풀려요. 2초만 쉬었다가 재시도
                continue
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                return "(AI가 너무 바빠요! 1분 후 다시 시도해주세요)"
            if is_overloaded:
                return "(AI 서버가 지금 너무 붐벼요! 잠시 후 다시 시도해주세요)"
            if "NOT_FOUND" in msg or "404" in msg:
                return "(AI 모델을 찾을 수 없어요. 선생님께 알려주세요)"
            return f"(오류: {e})"


def is_correct(ai_answer: str, item: dict) -> bool:
    """AI의 답이 정답이거나, 정답의 유사정답(| 로 구분) 중 하나와 같으면 정답 처리합니다."""
    valid_answers = {str(item["키워드"]).strip()}
    similar = item.get("유사정답")
    if isinstance(similar, str) and similar.strip():
        valid_answers |= {s.strip() for s in similar.split("|") if s.strip()}
    return str(ai_answer).strip() in valid_answers


def canvas_array_to_pil(image_data) -> Image.Image:
    """캔버스가 돌려주는 (세로, 가로, 4) 형태의 RGBA 숫자 배열을
    흰 배경 위에 합성한 보통의 PIL 이미지로 바꿔줍니다."""
    rgba = Image.fromarray(image_data.astype("uint8"), mode="RGBA")
    white_bg = Image.new("RGB", rgba.size, "white")
    white_bg.paste(rgba, mask=rgba.split()[3])
    return white_bg


def blank_canvas_image() -> Image.Image:
    return Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")


def draw_next_keyword():
    """카테고리 단어 더미에서 다음 단어를 하나 꺼내와, 지금 라운드의 제시어로 설정합니다.
    더미를 다 썼으면 다시 섞어서 재사용하되, 방금 나온 단어가 바로 또 나오지 않게 해요."""
    pool = st.session_state.keyword_pool
    pointer = st.session_state.keyword_pointer
    prev_word = st.session_state.current_item["키워드"] if st.session_state.current_item else None

    if pointer >= len(pool):
        reshuffled = pool[:]
        random.shuffle(reshuffled)
        if prev_word is not None and len(reshuffled) > 1 and reshuffled[0]["키워드"] == prev_word:
            reshuffled[0], reshuffled[1] = reshuffled[1], reshuffled[0]
        pool = reshuffled
        pointer = 0
        st.session_state.keyword_pool = pool

    st.session_state.current_item = pool[pointer]
    st.session_state.keyword_pointer = pointer + 1
    st.session_state.round_start_time = None   # 새 제시어는 60초를 새로 시작해요
    st.session_state._last_canvas_data = None
    st.session_state.draw_seq += 1


def start_new_game(category: str):
    keywords_df = load_keywords()
    subset = keywords_df.loc[keywords_df["카테고리"] == category].to_dict("records")
    random.shuffle(subset)

    st.session_state.category = category
    st.session_state.keyword_pool = subset
    st.session_state.keyword_pointer = 0
    st.session_state.current_item = None
    st.session_state.round_idx = 0
    st.session_state.rounds = []
    draw_next_keyword()
    st.session_state.page = "game"


# =======================================================
# 화면 1) 시작 화면 (카테고리 선택 + 게임 안내)
# =======================================================
def start_screen():
    st.title("🎨 AI 캐치마인드")
    st.caption("카테고리를 고르고, 제시어를 그림으로 그려서 AI가 맞히게 해보세요!")

    with st.expander("🕹️ 게임 방법"):
        st.markdown(
            "- 카테고리를 하나 고르면 바로 시작해요\n"
            "- 제시어를 보고 **60초 안에** 그림을 그려요\n"
            "- '제출하기'를 누르면 AI가 그림을 보고 정답을 맞혀봐요\n"
            "- 그리기 어려우면 '패스'로 다른 제시어를 받을 수 있어요 (패스는 문제 수에 안 들어가요)\n"
            f"- 정답 {TOTAL_ROUNDS}개를 다 풀면 결과를 확인해요"
        )

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
    """제출된 그림을 저장하고 AI에게 정답을 물어본 뒤, 채점 화면으로 넘어갑니다."""
    # 안전장치: 이번 라운드(draw_seq)에 대해 이미 제출 처리를 했다면 다시 하지 않아요.
    # (버튼을 여러 번 눌러도 AI를 두 번 부르지 않도록 막아주는 역할)
    if st.session_state.get("_last_submitted_seq") == st.session_state.draw_seq:
        return
    st.session_state._last_submitted_seq = st.session_state.draw_seq

    item = st.session_state.current_item
    keyword = item["키워드"]

    snapshot = canvas_array_to_pil(image_data) if image_data is not None else blank_canvas_image()

    # 캔버스 자리에 방금 그린 그림을 그대로 고정해서 보여줘요. (원래 크기 그대로, 확대 없이)
    st.image(snapshot, width=CANVAS_WIDTH, caption="제출한 그림")

    with st.spinner("🤖 AI가 생각 중입니다..."):
        ai_answer = ask_ai_guess(snapshot, st.session_state.category)

    buf = BytesIO()
    snapshot.save(buf, format="PNG")
    st.session_state.rounds.append({
        "keyword": keyword,
        "image_bytes": buf.getvalue(),
        "ai_answer": ai_answer,
        "correct": is_correct(ai_answer, item),
    })

    st.session_state.round_idx += 1
    st.session_state.page = "grading"
    st.rerun()


def game_screen():
    item = st.session_state.current_item
    keyword = item["키워드"]

    if st.session_state.round_start_time is None:
        st.session_state.round_start_time = time.time()

    elapsed = time.time() - st.session_state.round_start_time
    remaining = max(0, ROUND_LIMIT_SEC - int(elapsed))
    time_is_up = remaining <= 0

    # ---- 상단 정보: 한 줄로 압축해서 (진행 상황 · 카테고리 · 남은 시간) ----
    timer_color = "#D32F2F" if remaining <= 10 else "#888888"
    st.markdown(
        f"<div style='text-align:center;font-size:14px;color:{timer_color};'>"
        f"{st.session_state.category} · {st.session_state.round_idx + 1}/{TOTAL_ROUNDS}"
        f" · ⏱ {remaining}초</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h2 style='text-align:center;margin:6px 0 14px;'>✏️ {keyword}</h2>",
        unsafe_allow_html=True,
    )

    pass_clicked = False
    submit_clicked = False

    if not time_is_up:
        st_autorefresh(interval=1000, limit=ROUND_LIMIT_SEC + 5, key=f"timer_{st.session_state.draw_seq}")

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
            key=f"canvas_{st.session_state.draw_seq}",
        )

        if canvas_result.image_data is not None:
            st.session_state._last_canvas_data = canvas_result.image_data

        # 캔버스는 그대로 두고, 버튼만 별도의 st.form으로 묶었어요.
        # 이렇게 하면 "그림 그리는 중 계속 전송되는 캔버스 신호"랑
        # "제출 버튼 클릭"이 서로 꼬여서 클릭이씹히는 문제가 줄어들어요.
        with st.form(key=f"action_form_{st.session_state.draw_seq}", border=False):
            col1, col2 = st.columns([1, 1.4])
            with col1:
                pass_clicked = st.form_submit_button("🙅 패스", width="stretch")
            with col2:
                submit_clicked = st.form_submit_button("제출하기 ✅", type="primary", width="stretch")
    else:
        st.warning("⏰ 시간이 다 됐어요! 마지막 그림으로 자동 제출할게요.")

    if pass_clicked:
        draw_next_keyword()
        st.rerun()

    if submit_clicked or time_is_up:
        process_submission(st.session_state._last_canvas_data)


# =======================================================
# 화면 2.5) 라운드 채점 화면 (정답/오답 바로 보여주기)
# =======================================================
def grading_screen():
    r = st.session_state.rounds[-1]

    if r["correct"]:
        st.markdown(
            "<h2 style='text-align:center;color:#2E7D32;'>✅ 정답이에요!</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h2 style='text-align:center;color:#D32F2F;'>❌ 아쉬워요</h2>",
            unsafe_allow_html=True,
        )

    st.image(r["image_bytes"], width=DISPLAY_IMG_WIDTH)
    st.markdown(
        f"<div style='text-align:center;font-size:17px;margin-top:8px;'>"
        f"정답: <b>{r['keyword']}</b> &nbsp;·&nbsp; AI의 대답: <b>{r['ai_answer']}</b></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    is_last = st.session_state.round_idx >= TOTAL_ROUNDS
    label = "결과 보기 →" if is_last else "다음 문제 →"
    if st.button(label, type="primary", width="stretch"):
        if is_last:
            st.session_state.page = "result"
        else:
            draw_next_keyword()
            st.session_state.page = "game"
        st.rerun()


# =======================================================
# 화면 3) 결과 화면
# =======================================================
def result_screen():
    st.title("🏆 게임 결과")
    st.subheader(f"카테고리: {st.session_state.category}")
    st.write("")

    correct_count = sum(1 for r in st.session_state.rounds if r["correct"])

    for i, r in enumerate(st.session_state.rounds, start=1):
        with st.container(border=True):
            c1, c2 = st.columns([1, 1.6])
            with c1:
                st.image(r["image_bytes"], width=THUMB_IMG_WIDTH)
            with c2:
                st.markdown(f"**{i}번 문제**")
                st.write(f"정답: **{r['keyword']}**")
                st.write(f"AI의 대답: **{r['ai_answer']}**")
                st.write("✅ 정답이에요!" if r["correct"] else "❌ 아쉬워요")

    st.write("")
    st.metric("맞힌 개수", f"{correct_count} / {len(st.session_state.rounds)}")

    st.write("")
    if st.button("처음으로 돌아가기", width="stretch"):
        st.session_state.page = "start"
        st.session_state.category = None
        st.session_state.keyword_pool = []
        st.session_state.keyword_pointer = 0
        st.session_state.current_item = None
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
elif st.session_state.page == "grading":
    grading_screen()
elif st.session_state.page == "result":
    result_screen()
