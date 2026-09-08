import random
import re
import time
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from streamlit_drawable_canvas import st_canvas


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="centered",
)

ROUND_LIMIT_SEC = 60
TOTAL_ROUNDS = 5

CANVAS_WIDTH = 620
CANVAS_HEIGHT = 360

DISPLAY_IMG_WIDTH = 380
THUMB_IMG_WIDTH = 150

# ------------------------------------------------------------
# Gemini 모델
# ------------------------------------------------------------
# 그림 한 장을 보고 한 단어만 답하는 단순 작업이므로
# 현재 안정적인 경량 모델을 사용합니다.
#
# gemini-flash-latest를 쓰지 않는 이유:
# latest 별칭은 향후 더 무거운 모델로 바뀔 수 있기 때문입니다.
# ------------------------------------------------------------

GEMINI_MODEL = "gemini-3.5-flash-lite"


CATEGORIES = {
    "동물": "🐶",
    "과일": "🍎",
    "채소": "🥕",
    "사물": "📎",
    "교통수단": "🚗",
}


# ============================================================
# Session State 초기화
# ============================================================

DEFAULTS = {
    "page": "start",                 # start / game / processing / grading / ai_error / result
    "category": None,
    "keyword_pool": [],
    "keyword_pointer": 0,
    "current_item": None,

    # 실제 채점 완료 문제 수
    "round_idx": 0,

    "rounds": [],

    "round_start_time": None,

    # 캔버스 재생성용 고유 번호
    "draw_seq": 0,

    # 마지막 캔버스 그림
    "_last_canvas_data": None,

    # 중복 제출 방지
    "_submitted_seq": None,

    # 시간초과 자동 제출 방지
    "_timeout_handled_seq": None,

    # 제출 대기 중인 그림
    "pending_image_bytes": None,
    "pending_item": None,
    "pending_timed_out": False,

    # AI 오류
    "ai_error_type": None,
    "ai_error_message": None,
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# keyword.csv
# ============================================================

@st.cache_data
def load_keywords():

    df = pd.read_csv(
        "keyword.csv",
        encoding="utf-8-sig",
    )

    required = {
        "카테고리",
        "키워드",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "keyword.csv에는 '카테고리', '키워드' 열이 필요합니다."
        )

    # 유사정답은 없어도 실행 가능
    if "유사정답" not in df.columns:
        df["유사정답"] = ""

    df = df[
        ["카테고리", "키워드", "유사정답"]
    ].copy()

    df["카테고리"] = (
        df["카테고리"]
        .astype(str)
        .str.strip()
    )

    df["키워드"] = (
        df["키워드"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# Gemini Client
# ============================================================

@st.cache_resource
def get_gemini_client():

    try:
        api_key = str(
            st.secrets["GEMINI_API_KEY"]
        ).strip()

        if not api_key:
            return None

        return genai.Client(
            api_key=api_key
        )

    except Exception as exc:

        print(
            "[Gemini Client 오류]",
            type(exc).__name__,
            str(exc),
        )

        return None


# ============================================================
# 이미지 처리
# ============================================================

def canvas_array_to_pil(image_data):

    rgba = Image.fromarray(
        image_data.astype("uint8"),
        mode="RGBA",
    )

    white_bg = Image.new(
        "RGB",
        rgba.size,
        "white",
    )

    white_bg.paste(
        rgba,
        mask=rgba.split()[3],
    )

    return white_bg


def blank_canvas_image():

    return Image.new(
        "RGB",
        (
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
        ),
        "white",
    )


def optimize_image_for_ai(image: Image.Image):

    """
    화면에는 620x360 그림을 그대로 유지하지만
    Gemini 전송용 이미지만 축소합니다.
    """

    optimized = image.copy()

    optimized.thumbnail(
        (384, 384)
    )

    if optimized.mode != "RGB":
        optimized = optimized.convert("RGB")

    return optimized


def pil_to_png_bytes(image: Image.Image):

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# ============================================================
# AI 응답 정리
# ============================================================

def clean_ai_answer(text):

    if not text:
        return ""

    text = str(text).strip()

    # 혹시 "정답: 사과"처럼 나온 경우 처리
    text = re.sub(
        r"^(정답|답|추측)\s*[:：\-]?\s*",
        "",
        text,
    )

    match = re.search(
        r"[가-힣A-Za-z0-9]+",
        text,
    )

    if match:
        return match.group(0)

    return ""


# ============================================================
# Gemini AI 호출
# ============================================================

def ask_ai_guess(
    image: Image.Image,
    category: str,
):

    client = get_gemini_client()

    if client is None:

        return {
            "success": False,
            "type": "configuration",
            "message": "API 키 설정을 확인해 주세요.",
        }

    # --------------------------------------------------------
    # 무료 사용량 절약:
    # AI에 보내는 이미지만 축소
    # --------------------------------------------------------

    ai_image = optimize_image_for_ai(
        image
    )

    image_bytes = pil_to_png_bytes(
        ai_image
    )

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png",
    )

    # 프롬프트도 가능한 짧게 유지
    prompt = (
        f"초등학생이 그린 그림이다. "
        f"카테고리는 '{category}'이다. "
        f"형태와 윤곽을 중심으로 무엇인지 추론하라. "
        f"반드시 '{category}'에 속하는 대상 하나를 "
        f"설명 없이 한 단어로만 답하라."
    )

    # 503 서버 혼잡일 때만
    # 최대 1회 추가 재시도
    max_attempts = 2

    for attempt in range(
        max_attempts
    ):

        try:

            response = (
                client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        prompt,
                        image_part,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=10,
                    ),
                )
            )

            answer = clean_ai_answer(
                response.text
            )

            if not answer:

                return {
                    "success": False,
                    "type": "empty",
                    "message": "AI가 답을 생성하지 못했습니다.",
                }

            return {
                "success": True,
                "answer": answer,
            }

        except Exception as exc:

            message = str(exc)

            print(
                "[Gemini API 오류]",
                type(exc).__name__,
                message,
            )

            # =================================================
            # 429
            # 무료 사용량 / RPM 초과
            #
            # 중요:
            # 여기서는 자동 재시도하지 않음.
            # 괜히 API 호출 횟수만 더 소비하기 때문.
            # =================================================

            if (
                "429" in message
                or
                "RESOURCE_EXHAUSTED" in message
            ):

                return {
                    "success": False,
                    "type": "quota",
                    "message": (
                        "AI 무료 사용량 제한에 도달했습니다. "
                        "약 1분 후 다시 시도해 주세요."
                    ),
                }

            # =================================================
            # 503
            # 서버 혼잡은 1회만 재시도
            # =================================================

            if (
                "503" in message
                or
                "UNAVAILABLE" in message
            ):

                if attempt == 0:

                    time.sleep(2)

                    continue

                return {
                    "success": False,
                    "type": "server",
                    "message": (
                        "AI 서버가 현재 혼잡합니다. "
                        "잠시 후 다시 시도해 주세요."
                    ),
                }

            # =================================================
            # 404
            # 모델 종료 / 모델명 문제
            # =================================================

            if (
                "404" in message
                or
                "NOT_FOUND" in message
            ):

                return {
                    "success": False,
                    "type": "model",
                    "message": (
                        "현재 AI 모델을 사용할 수 없습니다. "
                        "모델 설정을 확인해 주세요."
                    ),
                }

            # =================================================
            # 401 / 403
            # API 키 / 권한
            # =================================================

            if (
                "401" in message
                or
                "403" in message
            ):

                return {
                    "success": False,
                    "type": "configuration",
                    "message": (
                        "Gemini API 키 또는 권한 설정을 "
                        "확인해 주세요."
                    ),
                }

            return {
                "success": False,
                "type": "unknown",
                "message": "AI 호출 중 오류가 발생했습니다.",
            }

    return {
        "success": False,
        "type": "unknown",
        "message": "AI 호출에 실패했습니다.",
    }


# ============================================================
# 정답 판정
# ============================================================

def is_correct(
    ai_answer,
    item,
):

    valid_answers = {
        str(
            item["키워드"]
        ).strip()
    }

    similar = item.get(
        "유사정답"
    )

    if (
        isinstance(similar, str)
        and
        similar.strip()
    ):

        valid_answers |= {
            answer.strip()
            for answer
            in similar.split("|")
            if answer.strip()
        }

    return (
        str(ai_answer).strip()
        in
        valid_answers
    )


# ============================================================
# 다음 제시어
# ============================================================

def draw_next_keyword():

    pool = (
        st.session_state.keyword_pool
    )

    pointer = (
        st.session_state.keyword_pointer
    )

    prev_word = None

    if st.session_state.current_item:

        prev_word = (
            st.session_state.current_item[
                "키워드"
            ]
        )

    # --------------------------------------------------------
    # 모든 단어를 사용했으면 다시 섞음
    # --------------------------------------------------------

    if pointer >= len(pool):

        reshuffled = pool[:]

        random.shuffle(
            reshuffled
        )

        if (
            prev_word is not None
            and
            len(reshuffled) > 1
            and
            reshuffled[0]["키워드"]
            ==
            prev_word
        ):

            (
                reshuffled[0],
                reshuffled[1],
            ) = (
                reshuffled[1],
                reshuffled[0],
            )

        pool = reshuffled

        pointer = 0

        st.session_state.keyword_pool = (
            pool
        )

    # --------------------------------------------------------
    # 다음 문제
    # --------------------------------------------------------

    st.session_state.current_item = (
        pool[pointer]
    )

    st.session_state.keyword_pointer = (
        pointer + 1
    )

    st.session_state.round_start_time = (
        time.time()
    )

    st.session_state._last_canvas_data = (
        None
    )

    st.session_state.draw_seq += 1

    # 새로운 라운드이므로 중복 제출 정보 초기화
    st.session_state._submitted_seq = None

    st.session_state._timeout_handled_seq = None


# ============================================================
# 게임 시작
# ============================================================

def start_new_game(
    category,
):

    df = load_keywords()

    subset = (
        df.loc[
            df["카테고리"]
            ==
            category
        ]
        .to_dict(
            "records"
        )
    )

    if len(subset) == 0:

        st.error(
            "해당 카테고리에 제시어가 없습니다."
        )

        return

    random.shuffle(
        subset
    )

    st.session_state.category = (
        category
    )

    st.session_state.keyword_pool = (
        subset
    )

    st.session_state.keyword_pointer = 0

    st.session_state.current_item = None

    st.session_state.round_idx = 0

    st.session_state.rounds = []

    st.session_state.pending_image_bytes = None

    st.session_state.pending_item = None

    st.session_state.ai_error_type = None

    st.session_state.ai_error_message = None

    draw_next_keyword()

    st.session_state.page = "game"


# ============================================================
# 제출 준비
# ============================================================

def prepare_submission(
    image_data,
    timed_out=False,
):

    # --------------------------------------------------------
    # 같은 라운드에서 두 번 제출되는 것 방지
    #
    # 중요한 점:
    # 첫 클릭에서는 항상 통과함.
    # --------------------------------------------------------

    current_seq = (
        st.session_state.draw_seq
    )

    if (
        st.session_state._submitted_seq
        ==
        current_seq
    ):

        return

    st.session_state._submitted_seq = (
        current_seq
    )

    # --------------------------------------------------------
    # 현재 그림을 즉시 스냅샷으로 변환
    # --------------------------------------------------------

    if image_data is not None:

        snapshot = (
            canvas_array_to_pil(
                image_data
            )
        )

    else:

        snapshot = (
            blank_canvas_image()
        )

    st.session_state.pending_image_bytes = (
        pil_to_png_bytes(
            snapshot
        )
    )

    st.session_state.pending_item = (
        dict(
            st.session_state.current_item
        )
    )

    st.session_state.pending_timed_out = (
        timed_out
    )

    # API 호출은 다음 화면에서 수행
    # → 버튼 클릭과 AI 호출을 분리해서 안정적으로 처리
    st.session_state.page = "processing"

    st.rerun()


# ============================================================
# 시작 화면
# ============================================================

def start_screen():

    st.title(
        "🎨 AI 캐치마인드"
    )

    st.caption(
        "카테고리를 고르고, "
        "제시어를 그림으로 표현해 보세요!"
    )

    with st.expander(
        "🕹️ 게임 방법"
    ):

        st.markdown(
            "- 카테고리를 하나 선택해요.\n"
            "- 제시어를 보고 **60초 안에** 그림을 그려요.\n"
            "- **제출하기**를 누르면 AI가 그림을 맞혀요.\n"
            "- 어려운 문제는 **패스**할 수 있어요.\n"
            "- 패스는 문제 수에 포함되지 않아요.\n"
            f"- 총 **{TOTAL_ROUNDS}문제**를 완료하면 결과를 확인해요."
        )

    st.write("")

    names = list(
        CATEGORIES.keys()
    )

    for i in range(
        0,
        len(names),
        3
    ):

        row_names = (
            names[i:i + 3]
        )

        cols = st.columns(
            len(row_names)
        )

        for col, name in zip(
            cols,
            row_names
        ):

            with col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"""
                        <div style="
                            text-align:center;
                            font-size:44px;
                        ">
                            {CATEGORIES[name]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"""
                        <div style="
                            text-align:center;
                            font-size:19px;
                            font-weight:600;
                            margin-bottom:8px;
                        ">
                            {name}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "시작하기",
                        key=f"start_{name}",
                        use_container_width=True,
                    ):

                        start_new_game(
                            name
                        )

                        st.rerun()


# ============================================================
# 타이머
# ============================================================

@st.fragment(
    run_every="1s"
)
def timer_fragment():

    if (
        st.session_state.page
        !=
        "game"
    ):

        return

    if (
        st.session_state.round_start_time
        is None
    ):

        return

    elapsed = (
        time.time()
        -
        st.session_state.round_start_time
    )

    remaining = max(
        0,
        ROUND_LIMIT_SEC
        -
        int(elapsed),
    )

    color = (
        "#D32F2F"
        if remaining <= 10
        else "#666666"
    )

    st.markdown(
        f"""
        <div style="
            text-align:center;
            color:{color};
            font-weight:700;
            font-size:16px;
        ">
            ⏱ 남은 시간 {remaining}초
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        remaining
        /
        ROUND_LIMIT_SEC
    )

    # --------------------------------------------------------
    # 시간 종료
    # --------------------------------------------------------

    if remaining <= 0:

        current_seq = (
            st.session_state.draw_seq
        )

        if (
            st.session_state._timeout_handled_seq
            !=
            current_seq
        ):

            st.session_state._timeout_handled_seq = (
                current_seq
            )

            st.rerun()


# ============================================================
# 게임 화면
# ============================================================

def game_screen():

    item = (
        st.session_state.current_item
    )

    keyword = (
        item["키워드"]
    )

    if (
        st.session_state.round_start_time
        is None
    ):

        st.session_state.round_start_time = (
            time.time()
        )

    elapsed = (
        time.time()
        -
        st.session_state.round_start_time
    )

    remaining = max(
        0,
        ROUND_LIMIT_SEC
        -
        int(elapsed),
    )

    time_is_up = (
        remaining <= 0
    )

    # --------------------------------------------------------
    # 상단 정보
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:14px;
            color:#777;
        ">
            {st.session_state.category}
            ·
            {st.session_state.round_idx + 1}
            /
            {TOTAL_ROUNDS}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <h2 style="
            text-align:center;
            margin:6px 0 10px;
        ">
            ✏️ {keyword}
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # 타이머
    # --------------------------------------------------------

    timer_fragment()

    # ========================================================
    # 시간 종료
    # ========================================================

    if time_is_up:

        st.warning(
            "⏰ 시간이 다 됐어요! "
            "마지막 그림으로 자동 제출할게요."
        )

        if (
            st.session_state._submitted_seq
            !=
            st.session_state.draw_seq
        ):

            prepare_submission(
                st.session_state._last_canvas_data,
                timed_out=True,
            )

        return

    # ========================================================
    # 그림판
    # ========================================================

    canvas_result = st_canvas(

        fill_color="rgba(0, 0, 0, 0)",

        stroke_width=8,

        stroke_color="#000000",

        background_color="#FFFFFF",

        height=CANVAS_HEIGHT,

        width=CANVAS_WIDTH,

        drawing_mode="freedraw",

        update_streamlit=True,

        # drawable-canvas 0.13.0
        return_image_data=True,

        key=(
            f"canvas_"
            f"{st.session_state.draw_seq}"
        ),
    )

    # --------------------------------------------------------
    # 현재 캔버스 데이터를 로컬 변수에 즉시 저장
    #
    # 제출 버튼을 한 번 눌렀을 때
    # 바로 이 값을 사용하기 위한 핵심 부분
    # --------------------------------------------------------

    current_canvas_data = None

    if (
        canvas_result is not None
        and
        canvas_result.image_data
        is not None
    ):

        current_canvas_data = (
            canvas_result.image_data
        )

        st.session_state._last_canvas_data = (
            current_canvas_data
        )

    else:

        current_canvas_data = (
            st.session_state._last_canvas_data
        )

    # ========================================================
    # 버튼
    # ========================================================

    col1, col2 = st.columns(
        [1, 1.4]
    )

    with col1:

        pass_clicked = st.button(
            "🙅 패스",
            use_container_width=True,
        )

    with col2:

        submit_clicked = st.button(
            "제출하기 ✅",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # 패스
    # --------------------------------------------------------

    if pass_clicked:

        draw_next_keyword()

        st.rerun()

    # --------------------------------------------------------
    # 제출
    #
    # ★ session_state 갱신을 기다리지 않고
    # 현재 canvas_result의 값을 바로 넘김
    # --------------------------------------------------------

    if submit_clicked:

        prepare_submission(
            current_canvas_data,
            timed_out=False,
        )


# ============================================================
# AI 처리 화면
# ============================================================

def processing_screen():

    image_bytes = (
        st.session_state.pending_image_bytes
    )

    item = (
        st.session_state.pending_item
    )

    if (
        image_bytes is None
        or
        item is None
    ):

        st.session_state.page = "game"

        st.rerun()

    st.subheader(
        "🤖 AI가 생각 중입니다..."
    )

    st.image(
        image_bytes,
        width=CANVAS_WIDTH,
        caption="제출한 그림",
    )

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    with st.spinner(
        "그림을 분석하고 있어요..."
    ):

        result = ask_ai_guess(
            image,
            st.session_state.category,
        )

    # ========================================================
    # AI 오류
    # ========================================================

    if not result["success"]:

        st.session_state.ai_error_type = (
            result["type"]
        )

        st.session_state.ai_error_message = (
            result["message"]
        )

        st.session_state.page = (
            "ai_error"
        )

        st.rerun()

    # ========================================================
    # 정상 응답
    # ========================================================

    ai_answer = (
        result["answer"]
    )

    correct = is_correct(
        ai_answer,
        item,
    )

    st.session_state.rounds.append(
        {
            "keyword": item["키워드"],
            "image_bytes": image_bytes,
            "ai_answer": ai_answer,
            "correct": correct,
            "timed_out": (
                st.session_state.pending_timed_out
            ),
        }
    )

    st.session_state.round_idx += 1

    st.session_state.page = (
        "grading"
    )

    st.rerun()


# ============================================================
# AI 오류 화면
# ============================================================

def ai_error_screen():

    st.title(
        "⚠️ AI 연결 안내"
    )

    if (
        st.session_state.pending_image_bytes
        is not None
    ):

        st.image(
            st.session_state.pending_image_bytes,
            width=DISPLAY_IMG_WIDTH,
            caption="제출한 그림",
        )

    error_type = (
        st.session_state.ai_error_type
    )

    message = (
        st.session_state.ai_error_message
    )

    # --------------------------------------------------------
    # 429
    # --------------------------------------------------------

    if error_type == "quota":

        st.warning(
            "⏳ AI 무료 사용량 제한에 도달했어요."
        )

        st.info(
            message
        )

    # --------------------------------------------------------
    # 503
    # --------------------------------------------------------

    elif error_type == "server":

        st.warning(
            "🌐 AI 서버가 혼잡해요."
        )

        st.info(
            message
        )

    # --------------------------------------------------------
    # 404
    # --------------------------------------------------------

    elif error_type == "model":

        st.error(
            "AI 모델 설정을 확인해 주세요."
        )

        st.write(
            message
        )

    # --------------------------------------------------------
    # API 설정
    # --------------------------------------------------------

    elif error_type == "configuration":

        st.error(
            "Gemini API 설정 오류입니다."
        )

        st.write(
            message
        )

    else:

        st.error(
            message
        )

    st.write("")

    # --------------------------------------------------------
    # 다시 호출
    #
    # 기존 제출 그림 그대로 재사용
    # 문제 수 증가 X
    # 오답 처리 X
    # --------------------------------------------------------

    if st.button(
        "🔄 AI 다시 호출",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.page = (
            "processing"
        )

        st.rerun()

    # --------------------------------------------------------
    # 문제로 돌아가기
    # --------------------------------------------------------

    if st.button(
        "↩️ 그림 다시 그리기",
        use_container_width=True,
    ):

        # 다시 제출할 수 있게 중복 제출 플래그 해제
        st.session_state._submitted_seq = (
            None
        )

        st.session_state.round_start_time = (
            time.time()
        )

        st.session_state.page = (
            "game"
        )

        st.rerun()


# ============================================================
# 채점 화면
# ============================================================

def grading_screen():

    result = (
        st.session_state.rounds[-1]
    )

    if result["correct"]:

        st.markdown(
            """
            <h2 style="
                text-align:center;
                color:#2E7D32;
            ">
                ✅ 정답이에요!
            </h2>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <h2 style="
                text-align:center;
                color:#D32F2F;
            ">
                ❌ 아쉬워요
            </h2>
            """,
            unsafe_allow_html=True,
        )

    st.image(
        result["image_bytes"],
        width=DISPLAY_IMG_WIDTH,
    )

    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:19px;
            margin-top:8px;
        ">
            정답:
            <b>{result['keyword']}</b>
            &nbsp;·&nbsp;
            AI의 대답:
            <b>{result['ai_answer']}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.get(
        "timed_out"
    ):

        st.caption(
            "⏰ 제한시간 종료 후 자동 제출된 그림입니다."
        )

    st.write("")

    is_last = (
        st.session_state.round_idx
        >=
        TOTAL_ROUNDS
    )

    label = (
        "결과 보기 →"
        if is_last
        else
        "다음 문제 →"
    )

    if st.button(
        label,
        type="primary",
        use_container_width=True,
    ):

        if is_last:

            st.session_state.page = (
                "result"
            )

        else:

            draw_next_keyword()

            st.session_state.page = (
                "game"
            )

        st.rerun()


# ============================================================
# 결과 화면
# ============================================================

def result_screen():

    st.title(
        "🏆 게임 결과"
    )

    st.subheader(
        f"카테고리: "
        f"{st.session_state.category}"
    )

    correct_count = sum(
        1
        for result
        in st.session_state.rounds
        if result["correct"]
    )

    for index, result in enumerate(
        st.session_state.rounds,
        start=1,
    ):

        with st.container(
            border=True
        ):

            col1, col2 = st.columns(
                [1, 1.6]
            )

            with col1:

                st.image(
                    result["image_bytes"],
                    width=THUMB_IMG_WIDTH,
                )

            with col2:

                st.markdown(
                    f"### {index}번 문제"
                )

                st.markdown(
                    f"🎯 정답: "
                    f"**{result['keyword']}**"
                )

                st.markdown(
                    f"🤖 AI의 대답: "
                    f"**{result['ai_answer']}**"
                )

                if result["correct"]:

                    st.success(
                        "✅ 정답이에요!"
                    )

                else:

                    st.error(
                        "❌ 아쉬워요"
                    )

    st.write("")

    st.metric(
        "맞힌 개수",
        f"{correct_count} / "
        f"{len(st.session_state.rounds)}",
    )

    st.write("")

    if st.button(
        "처음으로 돌아가기",
        use_container_width=True,
    ):

        for key in list(
            st.session_state.keys()
        ):

            del st.session_state[key]

        st.rerun()


# ============================================================
# 라우팅
# ============================================================

if st.session_state.page == "start":

    start_screen()

elif st.session_state.page == "game":

    game_screen()

elif st.session_state.page == "processing":

    processing_screen()

elif st.session_state.page == "ai_error":

    ai_error_screen()

elif st.session_state.page == "grading":

    grading_screen()

elif st.session_state.page == "result":

    result_screen()
