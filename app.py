import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import base64
import struct
import json
import os
import html as html_lib
from scipy.optimize import nnls
from scipy.ndimage import uniform_filter1d
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="Ohyoung UV-Vis",
    page_icon="logo.png" if os.path.exists("logo.png") else "🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)
# 그래프(plotly) 기본 글꼴: 영어·숫자 Outfit, 한글 나눔스퀘어 네오 (글꼴 파일은 static/fonts, 표 글꼴은 .streamlit/config.toml)
try:
    import plotly.io as _oy_pio
    _oy_pio.templates[_oy_pio.templates.default].layout.font.family = "OY Outfit, OY NanumSquare Neo, Pretendard, Malgun Gothic, sans-serif"
except Exception:
    pass

# ==========================================
# ⚙️ 매칭 신뢰도 판정 기준 (자기검증 결과를 보고 조정하세요)
# ==========================================
MATCH_MIN_SCORE = 0.95    # 1위 형태 점수가 이보다 낮으면 '일치 후보 약함'
MATCH_MIN_GAP = 0.02      # 1·2위 점수 차이가 이보다 작으면 '구분 어려움'
MATCH_MAX_DLMAX = 10.0    # 1위와 타겟의 λmax 차이(nm)가 이보다 크면 경고
BASELINE_WIDTH = 30.0     # 분석 구간 끝에서 베이스라인으로 쓸 폭(nm)
DERIV_SMOOTH = 5          # 미분 스펙트럼 평활화 창 크기(포인트 수)
MIX_GOOD_R2 = 0.99        # 혼합 피팅 '잘 맞음' 기준
MIX_OK_R2 = 0.95          # 혼합 피팅 '대체로 맞음' 기준

# ==========================================
# 🎨 디자인 토큰
# ==========================================
NAVY = "#1b489d"
NAVY_DARK = "#123373"
NAVY_SOFT = "#e9eef8"
INK = "#1d2433"
MUTED = "#6b7487"
LINE = "#e3e7ef"
OK, OK_BG = "#167a4a", "#e6f4ec"
WARN, WARN_BG = "#a65f00", "#fff4e0"
BAD, BAD_BG = "#b42318", "#fdecea"
# 그래프 시리즈 색상 (고정) — 1 검정, 2 빨강, 3 파랑, 4 보라, 이후 구분 잘 되는 색
# ※ 기존에 보던 색이므로 순서를 바꾸지 마세요
SERIES_COLORS = ["#000000", "#e01e1e", "#1f4fd8", "#8e3ccf", "#f28c00", "#0f9d74", "#d6457a", "#7a5230"]

def hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ==========================================
# 상태
# ==========================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "SPEC"
if "dye_type" not in st.session_state:
    st.session_state.dye_type = "Reactive"

def set_app_mode(mode):
    st.session_state.app_mode = mode

def change_dye_type(dye_name):
    st.session_state.dye_type = dye_name

def open_mix_with(cands):
    """매칭 후보들을 DB 선택 목록에 넣고 혼합 비율 예측 모드로 전환"""
    st.session_state[f"ms_db_{st.session_state.dye_type}"] = list(cands)[:6]
    st.session_state.app_mode = "MIX"

# ==========================================
# 🌟 전역 스타일
# ==========================================
try:
    with open("logo.png", "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
except Exception:
    logo_base64 = ""

# 상단 바 로고: 네이비 바탕에 맞춘 logo1.png (없으면 logo.png) — Spectra50 · SpectraMatch와 같은 방식
try:
    with open("logo1.png", "rb") as image_file:
        header_logo_base64 = base64.b64encode(image_file.read()).decode()
except Exception:
    header_logo_base64 = logo_base64

CSS = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
/* 글꼴: 영어·숫자 = Outfit, 한글 = 나눔스퀘어 네오 (저장소 static/fonts 폴더의 파일) */
@font-face { font-family: 'OY Outfit'; font-style: normal; font-weight: 100 900; font-display: swap; src: url('app/static/fonts/outfit-latin-wght-normal.woff2') format('woff2-variations'), url('app/static/fonts/outfit-latin-wght-normal.woff2') format('woff2'); unicode-range: U+0000-001F,U+0021-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD; }
@font-face { font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 100 349; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-aLt.woff2') format('woff2'); }
@font-face { font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 350 549; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-bRg.woff2') format('woff2'); }
@font-face { font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 550 749; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-cBd.woff2') format('woff2'); }
@font-face { font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 750 849; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-dEb.woff2') format('woff2'); }
@font-face { font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 850 900; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-eHv.woff2') format('woff2'); }
:root { --navy: NAVY_; --navy-dark: NAVYD_; --navy-soft: NAVYS_; --ink: INK_; --muted: MUTED_; --line: LINE_; }
html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'OY Outfit', 'OY NanumSquare Neo', 'Pretendard', -apple-system, 'Malgun Gothic', sans-serif !important;
}
.material-symbols-outlined { font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important; font-weight: normal; font-style: normal; letter-spacing: normal; text-transform: none; white-space: nowrap; word-wrap: normal; direction: ltr; -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; line-height: 1; vertical-align: middle; }
.stApp { background: #f5f7fb; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarHeader"] { display: none !important; }
.block-container { padding-top: 0 !important; padding-bottom: 3rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: none !important; }
[data-testid="stAppViewContainer"] { overflow-y: scroll !important; }

/* ---------- 상단 바: Spectra50 · SpectraMatch와 같은 네이비 고정 바 ---------- */
.st-key-topbar {
    position: fixed !important; top: 0; left: 0; right: 0; height: 60px; z-index: 999990;
    background: linear-gradient(100deg, var(--navy) 0%, #2557b8 100%);
    margin: 0 !important; padding: 0 20px !important; border-radius: 0;
    box-shadow: 0 4px 18px rgba(19,55,119,.28);
    display: flex; flex-direction: column; justify-content: center;
}
.st-key-topbar [data-testid="stHorizontalBlock"] { align-items: center; gap: 6px !important; }
.brand { display: flex; align-items: center; gap: 12px; color: #fff; white-space: nowrap; transform: translateY(-8px); }
.brand img { height: 38px; width: auto; max-width: 160px; object-fit: contain; }
.brand .mark { width: 36px; height: 36px; border-radius: 9px; background: rgba(255,255,255,.15);
               display: flex; align-items: center; justify-content: center; }
.brand .t1 { font-size: 21px; font-weight: 800; letter-spacing: -.3px; line-height: 1; color: #fff; }
.brand .hdr-mode { font-size: 12.5px; font-weight: 600; color: #fff; background: rgba(255,255,255,.16);
    border: 1px solid rgba(255,255,255,.22); padding: 3px 10px; border-radius: 999px; }

/* 상단 바 버튼: 투명 탭, 선택된 것만 흰색 */
.st-key-topbar button {
    background: transparent !important; color: rgba(255,255,255,.80) !important;
    border: 1px solid transparent !important; border-radius: 9px !important;
    font-weight: 600 !important; height: 36px !important; min-height: 36px !important; box-shadow: none !important;
}
.st-key-topbar button p { font-size: 13.5px !important; }
.st-key-topbar button:hover { background: rgba(255,255,255,.12) !important; color: #fff !important; }
.st-key-topbar button[kind="primary"], .st-key-topbar button[data-testid="stBaseButton-primary"] {
    background: #fff !important; color: var(--navy) !important; font-weight: 700 !important; box-shadow: 0 2px 6px rgba(0,0,0,.12) !important;
}
/* 사이드바 맨 위 DB 전환: 회색 바탕 위 선택된 쪽만 네이비 */
.st-key-dbswitch { background: #eef2f9; border: 1px solid var(--line); border-radius: 12px; padding: 4px !important; margin-bottom: 14px; }
.st-key-dbswitch [data-testid="stHorizontalBlock"] { gap: 4px !important; }
.st-key-dbswitch button { height: 34px !important; min-height: 34px !important; border-radius: 9px !important; font-weight: 700 !important; box-shadow: none !important; }
.st-key-dbswitch button[kind="secondary"], .st-key-dbswitch button[data-testid="stBaseButton-secondary"] { background: transparent !important; border-color: transparent !important; color: var(--muted) !important; }
.st-key-dbswitch button[kind="secondary"]:hover { color: var(--navy) !important; background: #fff !important; }
.st-key-dbswitch button[kind="primary"], .st-key-dbswitch button[data-testid="stBaseButton-primary"] { background: var(--navy) !important; border-color: var(--navy) !important; color: #fff !important; }
.block-container { padding-top: 84px !important; }

/* ---------- 사이드바 ---------- */
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] { z-index: 999980; }
[data-testid="stSidebarUserContent"] { padding-top: 80px !important; }
.step { display: flex; align-items: center; gap: 9px; margin: 6px 0 8px 0; font-weight: 700; color: var(--ink); font-size: 15px; }
.step .n { width: 22px; height: 22px; border-radius: 50%; background: var(--navy); color: #fff;
           font-size: 12px; display: flex; align-items: center; justify-content: center; font-weight: 700; }
.step .opt { font-weight: 500; color: var(--muted); font-size: 12px; }
.side-foot { color: var(--muted); font-size: 11.5px; margin-top: 18px; }

/* ---------- 카드 · 공통 ---------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius: 14px; }
div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-card"],
[class*="st-key-card"] {
    background: #fff; border: 1px solid var(--line) !important; border-radius: 14px;
    padding: 18px 20px !important; box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.sec { font-size: 13px; font-weight: 700; color: var(--muted); letter-spacing: .3px; margin: 0 0 10px 0;
       display: flex; align-items: center; gap: 6px; }
.sec .material-symbols-outlined { font-size: 18px; color: var(--navy); }
.page-title { font-size: 24px; font-weight: 800; color: var(--ink); letter-spacing: -.5px; margin: 0; }
.page-sub { color: var(--muted); font-size: 13.5px; margin: 2px 0 16px 0; }

.pill { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 999px;
        font-size: 12.5px; font-weight: 700; }
.pill .material-symbols-outlined { font-size: 16px; }

/* 매칭 결과 히어로 */
.hero { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
.hero .sw { width: 64px; height: 64px; border-radius: 14px; border: 1px solid rgba(0,0,0,.08); flex: none; }
.hero .lbl { font-size: 12.5px; color: var(--muted); font-weight: 600; }
.hero .name { font-size: 26px; font-weight: 800; color: var(--ink); letter-spacing: -.5px; line-height: 1.2; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin-top: 16px; }
.kpi { background: #f7f9fc; border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px; }
.kpi .k { font-size: 12px; color: var(--muted); font-weight: 600; }
.kpi .v { font-size: 20px; font-weight: 800; color: var(--ink); margin-top: 2px; font-variant-numeric: tabular-nums; }
.kpi .v small { font-size: 12px; font-weight: 600; color: var(--muted); margin-left: 2px; }
.kpi.accent { background: var(--navy); border-color: var(--navy); }
.kpi.accent .k { color: rgba(255,255,255,.75); } .kpi.accent .v { color: #fff; }
.issues { margin: 14px 0 0 0; padding: 12px 14px 12px 30px; border-radius: 10px; background: WARNBG_; color: #6b3d00; font-size: 13.5px; }
.issues li { margin: 2px 0; }

/* 색상 칩 */
.chip { display: flex; gap: 12px; align-items: center; padding: 8px 0; border-bottom: 1px dashed var(--line); }
.chip:last-child { border-bottom: none; }
.chip .sw { width: 46px; height: 46px; border-radius: 10px; border: 1px solid rgba(0,0,0,.08); flex: none; }
.chip .nm { font-weight: 700; color: var(--ink); font-size: 14px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.chip .lab { color: var(--muted); font-size: 12.5px; font-variant-numeric: tabular-nums; margin-top: 2px; }
.tag { font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 6px; }
.tag.db { background: #eef0f4; color: #4a5263; }
.tag.ai { background: var(--navy-soft); color: var(--navy); }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }

/* 농도 콜아웃 */
.callout { background: var(--navy-soft); border-radius: 12px; padding: 14px 16px; color: var(--ink); }
.callout .big { font-size: 22px; font-weight: 800; color: var(--navy); }
.callout .s { font-size: 12.5px; color: var(--muted); margin-top: 2px; }

/* 빈 화면 안내 */
.empty { text-align: center; padding: 56px 20px 40px; }
.empty .ic { width: 64px; height: 64px; border-radius: 18px; background: var(--navy-soft); color: var(--navy);
             display: inline-flex; align-items: center; justify-content: center; }
.empty .ic .material-symbols-outlined { font-size: 34px; }
.empty h3 { margin: 14px 0 4px; color: var(--ink); font-weight: 800; }
.empty p { color: var(--muted); margin: 0; }
.steps3 { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; max-width: 760px; margin: 26px auto 0; text-align: left; }
.steps3 div { background: #f7f9fc; border: 1px solid var(--line); border-radius: 12px; padding: 14px; font-size: 13.5px; color: var(--ink); }
.steps3 b { color: var(--navy); display: block; margin-bottom: 4px; }

/* 비율 바 */
.ratio-row { display: grid; grid-template-columns: 1fr 64px; gap: 10px; align-items: center; margin: 9px 0; }
.ratio-row .nm { font-size: 13.5px; font-weight: 600; color: var(--ink); margin-bottom: 4px; display: flex; align-items: center; gap: 6px; }
.ratio-row .bar { height: 10px; background: #edf0f5; border-radius: 99px; overflow: hidden; }
.ratio-row .bar > i { display: block; height: 100%; background: var(--navy); border-radius: 99px; }
.ratio-row .pct { text-align: right; font-weight: 800; color: var(--ink); font-variant-numeric: tabular-nums; }

/* 위젯 미세조정 */
[data-testid="stExpander"] details { border-radius: 12px !important; border-color: var(--line) !important; background: #fff; }
[data-testid="stFileUploaderDropzone"] { background: #f7f9fc; border: 1.5px dashed #c6cfdf; border-radius: 12px; }
</style>
"""
for k, v in {"NAVY_": NAVY, "NAVYD_": NAVY_DARK, "NAVYS_": NAVY_SOFT, "INK_": INK, "MUTED_": MUTED,
             "LINE_": LINE, "WARNBG_": WARN_BG}.items():
    CSS = CSS.replace(k, v)
import re as _re
CSS = _re.sub(r"/\*.*?\*/", "", CSS, flags=_re.S)            # 주석 제거
CSS = "\n".join(l for l in CSS.splitlines() if l.strip())    # 빈 줄이 있으면 마크다운이 CSS를 글자로 출력함
st.markdown(CSS, unsafe_allow_html=True)


def icon(name, size=None, color=None):
    style = ""
    if size: style += f"font-size:{size}px;"
    if color: style += f"color:{color};"
    return f"<span class='material-symbols-outlined' style='{style}'>{name}</span>"

def sec(title, ic):
    st.markdown(f"<div class='sec'>{icon(ic)}{title}</div>", unsafe_allow_html=True)

def pill(text, kind="ok", ic=None):
    fg, bg = {"ok": (OK, OK_BG), "warn": (WARN, WARN_BG), "bad": (BAD, BAD_BG), "info": (NAVY, NAVY_SOFT)}[kind]
    return f"<span class='pill' style='color:{fg};background:{bg};'>{icon(ic) if ic else ''}{text}</span>"


# ==========================================
# 🧭 상단 바 (브랜드 · 모드 · DB)
# ==========================================
app_mode = st.session_state.app_mode
with st.container(key="topbar"):
    c_brand, c_nav, c_gap = st.columns([1.5, 1.33, 3.47], vertical_alignment="center")
    with c_brand:
        logo_html = (f"<img src='data:image/png;base64,{header_logo_base64}'>" if header_logo_base64
                     else f"<div class='mark'>{icon('science', 22, '#fff')}</div>")
        st.markdown(f"<div class='brand'>{logo_html}<div class='t1'>Ohyoung UV-Vis</div>"
                    f"<span class='hdr-mode'>{st.session_state.dye_type}</span></div>", unsafe_allow_html=True)
    with c_nav:
        n1, n2 = st.columns(2)
        n1.button("스펙트럼 비교", icon=":material/show_chart:", use_container_width=True,
                  type="primary" if app_mode == "SPEC" else "secondary", on_click=set_app_mode, args=("SPEC",), key="nav_spec")
        n2.button("혼합 비율 예측", icon=":material/science:", use_container_width=True,
                  type="primary" if app_mode == "MIX" else "secondary", on_click=set_app_mode, args=("MIX",), key="nav_mix")

# ==========================================
# 🎨 색상 변환 및 AI 예측 내장 로직
# ==========================================
WX = [0.0, 0.0001, 0.0011, 0.0118, 0.0828, 0.3802, 1.1752, 2.446, 3.4341, 3.5389, 2.9807, 2.2911, 1.7201, 1.3816, 1.2508, 1.3059, 1.5808, 2.1201, 2.9672, 4.1138, 5.3765, 6.6217, 7.6312, 8.2071, 8.2286, 7.6217, 6.3587, 4.7782, 3.2339, 1.9712, 1.0866, 0.5395, 0.2413, 0.0972, 0.0352]
WY = [0.0002, 0.0005, 0.0015, 0.004, 0.0098, 0.0224, 0.0486, 0.1001, 0.196, 0.3656, 0.6055, 0.9604, 1.4703, 2.2178, 3.3709, 5.0962, 7.1315, 8.6799, 9.4352, 9.735, 9.4426, 8.8026, 7.8507, 6.6638, 5.3745, 4.1467, 3.0271, 2.0866, 1.356, 0.8298, 0.4799, 0.2611, 0.1336, 0.0643, 0.0291]
WZ = [0.002, 0.0099, 0.0398, 0.1381, 0.4583, 1.6659, 5.7292, 13.0408, 17.3943, 18.979, 17.5754, 13.4388, 8.2954, 4.7101, 2.7039, 1.5433, 0.8334, 0.4182, 0.1943, 0.0836, 0.0327, 0.0119, 0.004, 0.0013, 0.0004, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def calculate_lab_from_reflectance(ref_dict):
    X, Y, Z = 0.0, 0.0, 0.0
    for i, w in enumerate(range(360, 710, 10)):
        R = ref_dict.get(str(w), 0.0)
        X += R * WX[i]; Y += R * WY[i]; Z += R * WZ[i]
    def f(t): return t**(1/3) if t > 0.008856 else 7.787 * t + 16/116
    L = 116 * f(Y/100.0) - 16
    a = 500 * (f(X/94.811) - f(Y/100.0))
    b = 200 * (f(Y/100.0) - f(Z/107.304))
    return round(L, 2), round(a, 2), round(b, 2)

def lab_to_rgb(L, a, b):
    def finv(t): return t**3 if t**3 > 0.008856 else (t - 16/116) / 7.787
    fY = (L + 16) / 116
    X = 94.811 * finv(a / 500 + fY) / 100.0
    Y = 100.0 * finv(fY) / 100.0
    Z = 107.304 * finv(fY - b / 200) / 100.0
    r =  3.2404542*X - 1.5371385*Y - 0.4985314*Z
    g = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
    bl=  0.0556434*X - 0.2040259*Y + 1.0572252*Z
    def gamma(c): return 1.055 * (max(0.0, min(1.0, c))**(1/2.4)) - 0.055 if max(0.0, min(1.0, c)) > 0.0031308 else 12.92 * max(0.0, min(1.0, c))
    return f"rgb({max(0, min(255, int(round(gamma(r) * 255))))},{max(0, min(255, int(round(gamma(g) * 255))))},{max(0, min(255, int(round(gamma(bl) * 255))))})"

@st.cache_resource
def load_and_train_ai(df_uv):
    dye_json = {}
    name_map = {}
    try:
        with open("dye_data.json", "r", encoding="utf-8") as f: dye_json = json.load(f)
        if os.path.exists("dye_list.xlsx"):
            df_map = pd.read_excel("dye_list.xlsx", usecols="B:C", names=["Actual", "Modified"]).dropna()
            for _, row in df_map.iterrows():
                name_map[str(row["Modified"]).strip()] = str(row["Actual"]).strip()
                name_map[str(row["Actual"]).strip()] = str(row["Modified"]).strip()
    except Exception: pass

    X_train, Y_train = [], []
    for dye_name in df_uv.index:
        t_name = name_map.get(dye_name, dye_name)
        t_data = dye_json.get(t_name)
        if not t_data:
            for k in dye_json.keys():
                if k.lower().replace(" ", "") == t_name.lower().replace(" ", ""):
                    t_data = dye_json[k]
                    break
        if t_data and "1" in t_data:
            L, a, b = calculate_lab_from_reflectance(t_data["1"])
            X_train.append(df_uv.loc[dye_name].values)
            Y_train.append([L, a, b])

    model = None
    if len(X_train) > 0:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(np.array(X_train), np.array(Y_train))
    return model, dye_json, name_map


# ==========================================
# 🔬 스펙트럼 전처리 · 매칭 · 피팅 로직
# ==========================================
def fit_wavelengths(index_a, index_b, lo, hi):
    """두 스펙트럼에 공통으로 있는 파장 중 분석 구간(lo~hi)에 해당하는 것"""
    common = pd.Index(index_a).intersection(pd.Index(index_b))
    common = common[(common >= lo) & (common <= hi)]
    return common.sort_values()

def prep_matrix(mat, wl, use_baseline, use_deriv):
    """행 = 스펙트럼. 베이스라인 보정(구간 끝 평균 차감) 및 1차 미분(+평활화)"""
    m = np.asarray(mat, dtype=float)
    if m.ndim == 1:
        m = m[None, :]
    wl = np.asarray(wl, dtype=float)
    if use_baseline:
        tail = wl >= (wl.max() - BASELINE_WIDTH)
        m = m - m[:, tail].mean(axis=1, keepdims=True)
    if use_deriv:
        m = np.gradient(m, wl, axis=1)
        m = uniform_filter1d(m, size=DERIV_SMOOTH, axis=1, mode="nearest")
    return m

def rank_matches(target, ref_df, lo, hi, use_baseline, use_deriv):
    """DB 전체를 타겟과 비교해 형태 점수 순으로 정렬한 표를 반환"""
    wl = fit_wavelengths(ref_df.columns, target.index, lo, hi)
    if len(wl) > 0:
        valid = ref_df[wl].notna().all(axis=0).values & target[wl].notna().values
        wl = wl[valid]
    if len(wl) < 10:
        return None
    wl_arr = wl.values.astype(float)

    # 농도·λmax 계산용 (미분 안 함)
    T_raw = prep_matrix(target[wl].values, wl_arr, use_baseline, False)[0]
    R_raw = prep_matrix(ref_df[wl].values, wl_arr, use_baseline, False)
    # 형태 비교용
    T = prep_matrix(target[wl].values, wl_arr, use_baseline, use_deriv)[0]
    R = prep_matrix(ref_df[wl].values, wl_arr, use_baseline, use_deriv)

    # 피어슨 상관계수 (형태 점수)
    Tc = T - T.mean()
    Rc = R - R.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(Rc, axis=1) * np.linalg.norm(Tc)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denom > 0, (Rc @ Tc) / denom, np.nan)

    # 최소제곱 농도 배율 k (타겟 ≈ k × 기준) 및 그때의 R²
    rr = (R_raw ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(rr > 0, (R_raw @ T_raw) / rr, np.nan)
    resid = T_raw[None, :] - k[:, None] * R_raw
    sst = ((T_raw - T_raw.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum(axis=1) / sst if sst > 0 else np.full(len(k), np.nan)

    lmax_t = wl_arr[np.argmax(T_raw)]
    lmax_r = wl_arr[np.argmax(R_raw, axis=1)]

    res = pd.DataFrame({
        "염료": ref_df.index,
        "형태 점수": score,
        "Δλmax(nm)": lmax_r - lmax_t,
        "타겟 농도(%)": k * 100,
        "배율 적합 R²": r2,
    })
    res = res.sort_values("형태 점수", ascending=False, na_position="last").reset_index(drop=True)
    res.index = res.index + 1
    return res

def judge_match(res):
    """매칭 결과의 신뢰도 문제점 목록과 1·2위 점수 차이 반환"""
    issues = []
    top = res.iloc[0]
    gap = (top["형태 점수"] - res.iloc[1]["형태 점수"]) if len(res) > 1 else np.inf
    if not np.isfinite(top["형태 점수"]) or top["형태 점수"] < MATCH_MIN_SCORE:
        issues.append(f"1위 형태 점수({top['형태 점수']:.4f})가 기준({MATCH_MIN_SCORE})보다 낮습니다. DB에 없는 염료이거나 혼합 제품일 수 있습니다.")
    if np.isfinite(gap) and gap < MATCH_MIN_GAP:
        issues.append(f"1위({top['염료']})와 2위({res.iloc[1]['염료']})의 점수 차이가 {gap:.4f}로 작아 구분이 어렵습니다.")
    if np.isfinite(top["Δλmax(nm)"]) and abs(top["Δλmax(nm)"]) > MATCH_MAX_DLMAX:
        issues.append(f"1위와 타겟의 λmax 차이가 {top['Δλmax(nm)']:+.0f} nm로 큽니다.")
    return issues, gap

def nnls_fit(target, comp_df, lo, hi, use_baseline):
    """분석 구간에서 NNLS 혼합 피팅. 결과 dict 또는 None"""
    wl = fit_wavelengths(comp_df.columns, target.index, lo, hi)
    if len(wl) > 0:
        valid = comp_df[wl].notna().all(axis=0).values & target[wl].notna().values
        wl = wl[valid]
    if len(wl) < 10:
        return None
    wl_arr = wl.values.astype(float)
    Y = prep_matrix(target[wl].values, wl_arr, use_baseline, False)[0]
    X = prep_matrix(comp_df[wl].values, wl_arr, use_baseline, False).T
    coeffs, _ = nnls(X, Y)
    Y_pred = X @ coeffs
    sst = ((Y - Y.mean()) ** 2).sum()
    r2 = 1 - ((Y - Y_pred) ** 2).sum() / sst if sst > 0 else np.nan
    rmse = float(np.sqrt(np.mean((Y - Y_pred) ** 2)))
    return {"wl": wl_arr, "Y": Y, "Y_pred": Y_pred, "coeffs": coeffs, "r2": r2, "rmse": rmse}

def scale_fit(ref, target, lo, hi, use_baseline):
    """타겟 ≈ k × 기준 이 되는 농도 배율 k와 R² (분석 구간 전체 기준)"""
    wl = fit_wavelengths(ref.dropna().index, target.dropna().index, lo, hi)
    if len(wl) < 10:
        return None
    wl_arr = wl.values.astype(float)
    r = prep_matrix(ref[wl].values, wl_arr, use_baseline, False)[0]
    t = prep_matrix(target[wl].values, wl_arr, use_baseline, False)[0]
    if r @ r <= 0:
        return None
    k = (r @ t) / (r @ r)
    sst = ((t - t.mean()) ** 2).sum()
    r2 = 1 - ((t - k * r) ** 2).sum() / sst if sst > 0 else np.nan
    return k, r2


# --- 데이터 불러오기 ---
@st.cache_data
def load_data(file_name):
    df = pd.read_csv(file_name, index_col=0)
    df.columns = df.columns.astype(float)
    return df


# ==========================================
# 데이터 로드
# ==========================================
db_file = "Final_UV_Data_R.csv" if st.session_state.dye_type == "Reactive" else "Final_UV_Data_D.csv"
try:
    df = load_data(db_file)
    ai_model, dye_json_data, name_mapping = load_and_train_ai(df)
except FileNotFoundError:
    st.error(f"'{db_file}' 파일을 찾을 수 없습니다. app.py와 같은 폴더에 있는지 확인하세요.", icon=":material/error:")
    st.stop()


# ==========================================
# 왼쪽 사이드바: ① 측정 파일 → ② 비교 염료 → 고급 설정
# ==========================================
def step(n, title, opt=""):
    o = f"<span class='opt'>{opt}</span>" if opt else ""
    st.sidebar.markdown(f"<div class='step'><span class='n'>{n}</span>{title}{o}</div>", unsafe_allow_html=True)

# 염료 DB 전환 (사이드바 맨 위)
with st.sidebar.container(key="dbswitch"):
    d1, d2 = st.columns(2, gap="small")
    d1.button("Reactive", use_container_width=True, key="db_r",
              type="primary" if st.session_state.dye_type == "Reactive" else "secondary", on_click=change_dye_type, args=("Reactive",))
    d2.button("Disperse", use_container_width=True, key="db_d",
              type="primary" if st.session_state.dye_type == "Disperse" else "secondary", on_click=change_dye_type, args=("Disperse",))

step(1, "측정 파일 올리기", "SD 파일")
uploaded_files = st.sidebar.file_uploader("SD 파일", type=['sd'], accept_multiple_files=True, label_visibility="collapsed")

uploaded_spectra = {}
if uploaded_files:
    wavelengths = np.array(range(190, 1101), dtype=float)
    spectrum_bytes_length = len(wavelengths) * 8
    headers = { b'\x28\x00\x41\x00\x55\x00\x29\x00': 17, b'\x28\x41\x55\x29\x00': 5 }
    for file in uploaded_files:
        file_bytes = file.getvalue()
        clean_name = file.name.replace(".sd", "").replace(".SD", "")

        for h, spacing in headers.items():
            total_matches = file_bytes.count(h)
            start_search = count = 0
            while True:
                header_idx = file_bytes.find(h, start_search)
                if header_idx == -1: break
                start_idx = header_idx + spacing
                end_idx = start_idx + spectrum_bytes_length
                if end_idx <= len(file_bytes):
                    spectrum_data = file_bytes[start_idx:end_idx]
                    try:
                        absorbances = [val for val, in struct.iter_unpack('<d', spectrum_data)]
                        series = pd.Series(absorbances, index=wavelengths)
                        series.index.name = "Wavelength"
                        count += 1
                        spec_name = f"{clean_name} ({count})" if total_matches > 1 else clean_name
                        base_name, dedup = spec_name, 1
                        while spec_name in uploaded_spectra:
                            spec_name = f"{base_name} ({dedup})"; dedup += 1
                        uploaded_spectra[spec_name] = series
                    except Exception: pass
                start_search = header_idx + 1

combined_df = pd.concat([df, pd.DataFrame(uploaded_spectra).T]) if uploaded_spectra else df
target_name = target_series_sd = None

if uploaded_spectra:
    target_name = st.sidebar.selectbox("타겟(분석할 시료)", options=list(uploaded_spectra.keys()))
    target_series_sd = uploaded_spectra[target_name]

st.sidebar.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
max_sel = 6
step(2, "비교할 염료", f"최대 {max_sel}개 · 선택사항" if uploaded_spectra else f"최대 {max_sel}개")
selected_db_dyes = st.sidebar.multiselect("DB 염료", options=df.index.tolist(), max_selections=max_sel,
                                          key=f"ms_db_{st.session_state.dye_type}",
                                          placeholder=f"{st.session_state.dye_type} DB에서 검색")

selected_upload_dyes = []
if uploaded_spectra:
    upload_choices = [k for k in uploaded_spectra.keys() if k != target_name]
    if upload_choices:
        selected_upload_dyes = st.sidebar.multiselect("업로드한 다른 시료", options=upload_choices, max_selections=max_sel,
                                                      key=f"ms_up_{st.session_state.dye_type}", placeholder="업로드 파일에서 추가")

selected_dyes = (selected_db_dyes + selected_upload_dyes)[:max_sel]
if uploaded_spectra and not selected_dyes and app_mode == "SPEC":
    st.sidebar.caption("비워 두면 DB에서 가장 비슷한 염료를 자동으로 찾아 비교합니다.")

st.sidebar.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
with st.sidebar.expander("고급 설정", icon=":material/tune:"):
    st.markdown("**매칭·혼합 분석 구간 (nm)**")
    col_f1, col_f2 = st.columns(2)
    fit_min = col_f1.number_input("시작", value=380.0, step=10.0, format="%.0f")
    fit_max = col_f2.number_input("끝", value=780.0, step=10.0, format="%.0f")
    use_baseline = st.toggle(f"베이스라인 보정", value=True, help=f"분석 구간 끝 {BASELINE_WIDTH:.0f} nm 평균을 0으로 맞춥니다.")
    use_deriv = st.toggle("1차 미분으로 매칭", value=True, help="모양 비교를 미분 스펙트럼으로 합니다. 분산염료에 권장.")
    st.markdown("**최대 피크 찾는 구간 (nm)**")
    col1, col2 = st.columns(2)
    min_wave = col1.number_input("최소", value=380.0, step=10.0, format="%.0f")
    max_wave = col2.number_input("최대", value=780.0, step=10.0, format="%.0f")
if fit_min >= fit_max:
    st.sidebar.warning("분석 구간이 올바르지 않아 380~780 nm를 사용합니다.", icon=":material/warning:")
    fit_min, fit_max = 380.0, 780.0

st.sidebar.markdown(f"<div class='side-foot'>DB: {st.session_state.dye_type} · {len(df)}종<br>Created by tskwon</div>", unsafe_allow_html=True)


# ==========================================
# 🧩 공통 표시 도우미
# ==========================================
def get_lab_rgb_for_dye(dye_name):
    if not dye_json_data: return None
    t_name = name_mapping.get(dye_name, dye_name)
    t_data = dye_json_data.get(t_name)
    if not t_data:
        for k in dye_json_data.keys():
            if k.lower().replace(" ", "") == t_name.lower().replace(" ", ""):
                t_data = dye_json_data[k]; break
    if t_data and "1" in t_data:
        L, a, b = calculate_lab_from_reflectance(t_data["1"])
        return L, a, b, lab_to_rgb(L, a, b)
    return None

def predict_unknown_lab(series_sd):
    if ai_model is None: return None
    x_input = series_sd.reindex(df.columns).fillna(0).values.reshape(1, -1)
    p_L, p_a, p_b = ai_model.predict(x_input)[0]
    return round(p_L, 2), round(p_a, 2), round(p_b, 2), lab_to_rgb(p_L, p_a, p_b)

def get_color_info(name, series):
    """(L, a, b, rgb, 출처) — DB 값 우선, 없으면 업로드 데이터에 한해 AI 예측"""
    lab = get_lab_rgb_for_dye(name)
    if lab:
        return (*lab, "DB")
    if name in uploaded_spectra:
        ai = predict_unknown_lab(series)
        if ai:
            return (*ai, "AI")
    return None

def lab_chip_html(name, series, dot_color=None):
    """화면용 색상 칩"""
    info = get_color_info(name, series)
    if not info: return ""
    L, a, b, rgb, src = info
    tag = "<span class='tag ai'>AI 예측</span>" if src == "AI" else "<span class='tag db'>DB</span>"
    dot = f"<span class='dot' style='background:{dot_color}'></span>" if dot_color else ""
    return (f"<div class='chip'><div class='sw' style='background:{rgb}'></div><div>"
            f"<div class='nm'>{dot}{html_lib.escape(str(name))}{tag}</div>"
            f"<div class='lab'>L* {L} · a* {a} · b* {b}</div></div></div>")

def lab_chip_copy_html(name, series):
    """복사용 색상 칩 (메일·문서에 붙여넣기 좋게 인라인 스타일)"""
    info = get_color_info(name, series)
    if not info: return ""
    L, a, b, rgb, src = info
    tag = "AI 예측" if src == "AI" else "DB"
    return (f"<table style='border-collapse:collapse;margin:4px 0;font-family:sans-serif;'><tr>"
            f"<td style='width:60px;height:30px;background:{rgb};border:1px solid #ccc;'></td>"
            f"<td style='padding:0 10px;font-size:13px;'><b>{html_lib.escape(str(name))}</b> ({tag})<br>"
            f"L={L}, a={a}, b={b}</td></tr></table>")

def html_table(headers, rows, first_col_colors=None):
    """first_col_colors: 각 행 첫 칸(이름) 글자색 — 그래프 색과 맞출 때 사용"""
    out = "<table border='1' style='border-collapse: collapse; text-align: center; font-family: sans-serif; font-size: 13px;'>"
    out += f"<tr style='background-color: {NAVY}; color: #fff;'>" + "".join(f"<th style='padding: 7px 10px;'>{html_lib.escape(str(h))}</th>" for h in headers) + "</tr>"
    for ri, r in enumerate(rows):
        cells = []
        for ci, c in enumerate(r):
            style = "padding: 7px 10px;"
            if ci == 0 and first_col_colors and ri < len(first_col_colors):
                style += f" color: {first_col_colors[ri]}; font-weight: bold;"
            cells.append(f"<td style='{style}'>{html_lib.escape(str(c))}</td>")
        out += "<tr>" + "".join(cells) + "</tr>"
    return out + "</table>"

def render_copy_button(html_content, label):
    payload = json.dumps(html_content)
    components.html(f"""
    <style>
    @font-face {{ font-family: 'OY Outfit'; font-style: normal; font-weight: 100 900; font-display: swap; src: url('app/static/fonts/outfit-latin-wght-normal.woff2') format('woff2-variations'), url('app/static/fonts/outfit-latin-wght-normal.woff2') format('woff2'); unicode-range: U+0000-001F,U+0021-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD; }}
    @font-face {{ font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 100 349; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-aLt.woff2') format('woff2'); }}
    @font-face {{ font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 350 549; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-bRg.woff2') format('woff2'); }}
    @font-face {{ font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 550 749; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-cBd.woff2') format('woff2'); }}
    @font-face {{ font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 750 849; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-dEb.woff2') format('woff2'); }}
    @font-face {{ font-family: 'OY NanumSquare Neo'; font-style: normal; font-weight: 850 900; font-display: swap; src: url('app/static/fonts/NanumSquareNeoTTF-eHv.woff2') format('woff2'); }}
    </style>
    <button onclick="copyToClipboard(this)" style="width: 100%; background-color: {NAVY}; color: white; border: none; padding: 11px 0; border-radius: 10px; font-size: 14.5px; cursor: pointer; font-weight: 700; display: flex; justify-content: center; align-items: center; gap: 6px; transition: 0.2s; font-family: 'OY Outfit','OY NanumSquare Neo','Pretendard','Malgun Gothic',sans-serif;"
        onmouseover="this.style.backgroundColor='{NAVY_DARK}'" onmouseout="this.style.backgroundColor='{NAVY}'">
        <svg xmlns="http://www.w3.org/2000/svg" height="19" viewBox="0 -960 960 960" width="19" fill="white"><path d="M360-240q-33 0-56.5-23.5T280-320v-480q0-33 23.5-56.5T360-880h360q33 0 56.5 23.5T800-800v480q0 33-23.5 56.5T720-240H360Zm0-80h360v-480H360v480ZM200-80q-33 0-56.5-23.5T120-160v-560h80v560h440v80H200Zm160-240v-480 480Z"/></svg>
        <span>{label}</span>
    </button>
    <script>
    function copyToClipboard(button) {{
        const htmlText = {payload};
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = htmlText;
        tempDiv.style.position = "absolute"; tempDiv.style.left = "-9999px";
        document.body.appendChild(tempDiv);
        const selection = window.getSelection(); const range = document.createRange();
        range.selectNodeContents(tempDiv); selection.removeAllRanges(); selection.addRange(range);
        try {{
            document.execCommand('copy');
            const iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" height="19" viewBox="0 -960 960 960" width="19" fill="white"><path d="M382-240 154-468l57-57 171 171 367-367 57 57-424 424Z"/></svg>';
            const originalHtml = button.innerHTML;
            button.innerHTML = iconSvg + '<span>복사 완료! (Ctrl+V)</span>';
            button.style.backgroundColor = '{OK}';
            setTimeout(() => {{ button.innerHTML = originalHtml; button.style.backgroundColor = '{NAVY}'; }}, 2500);
        }} catch(e) {{}}
        document.body.removeChild(tempDiv); selection.removeAllRanges();
    }}
    </script>
    """, height=48)

def style_fig(fig, title=None, height=470, x_range=None):
    fig.update_layout(
        title=dict(text=title or "", font=dict(size=14, color=INK), x=0.01),   # 제목이 없을 때 None을 주면 그래프 위에 'undefined'가 찍힘
        height=height, hovermode="x unified", dragmode="zoom",
        margin=dict(l=10, r=10, t=44 if title else 16, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="OY Outfit, OY NanumSquare Neo, Pretendard, Malgun Gothic, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font_size=12),
    )
    fig.update_xaxes(title_text="Wavelength (nm)", showgrid=True, gridcolor="#eef1f6", zeroline=False,
                     linecolor=LINE, ticks="outside", tickcolor=LINE, range=x_range)
    fig.update_yaxes(title_text="Absorbance (AU)", showgrid=True, gridcolor="#eef1f6",
                     zeroline=True, zerolinecolor="#d9dee8", linecolor=LINE)
    return fig

PLOT_CONFIG = {"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

def series_picker_css(colors, box_key):
    """선택 버튼(pills)을 그래프 색과 같은 색으로 칠함 — n번째 버튼 = n번째 시리즈"""
    rules = []
    for i, c in enumerate(colors, 1):
        sels = [f".st-key-{box_key} [data-testid='stButtonGroup'] button:nth-of-type({i})",
                f".st-key-{box_key} [data-testid='stButtonGroup'] > div > div:nth-child({i}) button"]
        base = ", ".join(sels)
        active = ", ".join(f"{x}[data-testid='stBaseButton-pillsActive']" for x in sels)
        dot = ", ".join(f"{x}::before" for x in sels)
        rules.append(f"{base} {{ color: {c} !important; border: 1.5px solid {hex_rgba(c, .35)} !important; background: #fff !important; opacity: .55; }}")
        rules.append(f"{active} {{ color: {c} !important; border: 1.5px solid {c} !important; background: {hex_rgba(c, .10)} !important; opacity: 1; font-weight: 700 !important; }}")
        rules.append(f"{dot} {{ content: ''; display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {c}; margin-right: 7px; flex: none; }}")
    rules.append(f".st-key-{box_key} [data-testid='stButtonGroup'] button p {{ color: inherit !important; font-weight: inherit !important; }}")
    st.markdown("<style>" + " ".join(rules) + "</style>", unsafe_allow_html=True)

def series_picker(names, key):
    """그래프에 보일 시리즈 선택 (pills, 구버전이면 multiselect)"""
    if hasattr(st, "pills"):
        sel = st.pills("표시할 스펙트럼", names, selection_mode="multi", default=names, key=key, label_visibility="collapsed")
    else:
        sel = st.multiselect("표시할 스펙트럼", names, default=names, key=key, label_visibility="collapsed")
    return sel or []

def empty_state(ic, title, desc, steps):
    cells = "".join(f"<div><b>{i}. {t}</b>{d}</div>" for i, (t, d) in enumerate(steps, 1))
    st.markdown(f"<div class='empty'><div class='ic'>{icon(ic)}</div><h3>{title}</h3><p>{desc}</p>"
                f"<div class='steps3'>{cells}</div></div>", unsafe_allow_html=True)


# ==========================================
# 모드 1: 스펙트럼 비교
# ==========================================
if app_mode == "SPEC":
    st.markdown("<div class='page-title'>스펙트럼 비교</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>측정한 시료가 어떤 염료와 가장 비슷한지 찾고, 농도와 색상을 비교합니다 · 분석 구간 {fit_min:.0f}–{fit_max:.0f} nm</div>", unsafe_allow_html=True)

    plot_items_all = []
    best_matches = []
    match_res = None
    match_issues = []
    gap = np.nan

    if target_series_sd is not None and not target_series_sd.empty:
        plot_items_all.append({"name": target_name, "data": target_series_sd, "is_target": True})
        match_res = rank_matches(target_series_sd, df, fit_min, fit_max, use_baseline, use_deriv)
        if match_res is None:
            st.warning("분석 구간에서 DB와 겹치는 파장이 부족합니다. 고급 설정의 분석 구간을 확인하세요.", icon=":material/warning:")
        else:
            match_issues, gap = judge_match(match_res)
            best_matches = [match_res.iloc[0]["염료"]] if not match_issues else match_res.head(2)["염료"].tolist()

    dyes_to_plot = selected_dyes.copy()
    if target_series_sd is not None and len(dyes_to_plot) == 0 and len(best_matches) > 0:
        dyes_to_plot = best_matches
    added_names = [item["name"] for item in plot_items_all]
    for dye in dyes_to_plot:
        if dye not in added_names:
            plot_items_all.append({"name": dye, "data": combined_df.loc[dye], "is_target": False})
            added_names.append(dye)
    for i, item in enumerate(plot_items_all):
        item["color"] = SERIES_COLORS[i % len(SERIES_COLORS)]

    if not plot_items_all:
        empty_state("show_chart", "분석할 스펙트럼을 올려 주세요",
                    "왼쪽에서 SD 파일을 올리거나, DB 염료만 골라 비교할 수도 있습니다.",
                    [("SD 파일 올리기", "측정한 시료 파일을 끌어다 놓으세요."),
                     ("자동 매칭 확인", "DB에서 가장 비슷한 염료와 농도 차이를 보여줍니다."),
                     ("결과 복사", "표와 요약을 메일·보고서에 바로 붙여넣으세요.")])
        st.stop()

    # ---------- ① 매칭 결과 카드 ----------
    if match_res is not None:
        top = match_res.iloc[0]
        top_info = get_color_info(top["염료"], df.loc[top["염료"]])
        sw = f"<div class='sw' style='background:{top_info[3]}'></div>" if top_info else ""
        if not match_issues:
            status = pill("신뢰할 수 있는 일치", "ok", "check_circle")
        elif top["형태 점수"] < MATCH_MIN_SCORE:
            status = pill("일치하는 염료 없음 / 혼합 의심", "bad", "error")
        else:
            status = pill("확인 필요", "warn", "warning")
        with st.container(key="card_match"):
            st.markdown(
                f"<div class='sec'>{icon('target')}DB 자동 매칭 · {'미분' if use_deriv else '원'} 스펙트럼</div>"
                f"<div class='hero'>{sw}<div><div class='lbl'>{html_lib.escape(str(target_name))} 와(과) 가장 비슷한 염료</div>"
                f"<div class='name'>{html_lib.escape(str(top['염료']))}</div><div style='margin-top:6px'>{status}</div></div></div>"
                + (("<ul class='issues'>" + "".join(f"<li>{html_lib.escape(str(x_))}</li>" for x_ in match_issues) + "</ul>") if match_issues else ""),
                unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ---------- ② 그래프 + 요약 ----------
    col_chart, col_side = st.columns([1.95, 1], gap="medium")

    with col_chart:
        with st.container(key="card_chart"):
            names_all = [it["name"] for it in plot_items_all]
            picker_key = "vis_" + str(abs(hash(tuple(names_all))))
            series_picker_css([it["color"] for it in plot_items_all], "serpick")
            with st.container(key="serpick"):
                visible = series_picker(names_all, picker_key)
            active_plot_items = [it for it in plot_items_all if it["name"] in visible]

            fig = go.Figure()
            fig.add_vrect(x0=fit_min, x1=fit_max, fillcolor="rgba(27,72,157,0.05)", line_width=0,
                          annotation_text="분석 구간", annotation_position="top left",
                          annotation_font=dict(size=10, color=NAVY))
            table_rows = []
            for item in active_plot_items:
                name, series, color = item["name"], item["data"], item["color"]
                fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=name,
                                         line=dict(color=color, width=1.2)))   # 모든 선 같은 굵기 (얇게)
                mask = (series.index >= min_wave) & (series.index <= max_wave)
                if np.any(mask):
                    range_series = series[mask].dropna()
                    if range_series.empty: continue
                    p_wave, p_abs = range_series.idxmax(), range_series.max()
                    fig.add_trace(go.Scatter(x=[p_wave], y=[p_abs], mode='markers+text',
                                             marker=dict(color=color, size=7, line=dict(color="white", width=1.5)),
                                             text=[f"{p_wave:.0f} nm"], textposition="top center",
                                             textfont=dict(color=color, size=11), showlegend=False, hoverinfo='skip'))
                    table_rows.append((name, color, p_wave, p_abs))
            style_fig(fig, height=500, x_range=[190, 1100])
            if active_plot_items:
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            else:
                st.info("위에서 표시할 스펙트럼을 하나 이상 선택하세요.", icon=":material/visibility:")

    # 농도 분석: 분석 구간 전체를 최소제곱으로 맞춘 배율
    plain_summary_text = ""
    conc_html = ""
    target_item = next((it for it in active_plot_items if it.get("is_target")), None)
    ref_item = next((it for it in active_plot_items if not it.get("is_target")), None)
    if target_item is not None and ref_item is not None:
        sf = scale_fit(ref_item["data"], target_item["data"], fit_min, fit_max, use_baseline)
        if sf is not None:
            k, r2_k = sf
            diff_pct = (k - 1) * 100
            direction_str = "진합니다" if diff_pct > 0 else "연합니다"
            plain_summary_text = (f"농도 분석: {target_item['name']}이 {ref_item['name']} 대비 약 {abs(diff_pct):.1f}% 더 {direction_str} "
                                  f"({fit_min:.0f}–{fit_max:.0f} nm 기준, R²={r2_k:.3f})")
            low_fit = np.isfinite(r2_k) and r2_k < 0.9
            if low_fit:
                plain_summary_text += " — 스펙트럼 모양 차이가 커서 참고용입니다"
            plain_summary_text += "."
            arrow = "▲" if diff_pct > 0 else "▼"
            note = f"<div class='s' style='color:{WARN}'>모양 차이가 커서 참고용입니다</div>" if low_fit else ""
            conc_html = (f"<div class='callout'><div class='s'>{html_lib.escape(ref_item['name'])} 대비 농도</div>"
                         f"<div class='big'>{arrow} {abs(diff_pct):.1f}% {'진함' if diff_pct > 0 else '연함'}</div>"
                         f"<div class='s'>R² {r2_k:.3f} · {fit_min:.0f}–{fit_max:.0f} nm 전체 기준</div>{note}</div>")

    with col_side:
        with st.container(key="card_side"):
            if conc_html:
                sec("농도 비교", "opacity")
                st.markdown(conc_html, unsafe_allow_html=True)
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            chips = "".join(lab_chip_html(it["name"], it["data"], it["color"]) for it in active_plot_items)
            if chips:
                sec("색상 (CIE L*a*b*)", "palette")
                st.markdown(chips, unsafe_allow_html=True)
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            if table_rows:
                sec(f"최대 피크 ({min_wave:.0f}–{max_wave:.0f} nm)", "insights")
                peak_df = pd.DataFrame([(n, f"{w:.1f}", f"{a:.4f}") for n, _, w, a in table_rows],
                                       columns=["Name", "λmax (nm)", "Abs (AU)"])
                name_colors = [c for _, c, _, _ in table_rows]
                peak_styled = peak_df.style.apply(
                    lambda col: [f"color: {c}; font-weight: 700" for c in name_colors], subset=["Name"])
                st.dataframe(peak_styled, hide_index=True, use_container_width=True)

                html_content = "<h3>[실무 분석 요약]</h3>"
                if plain_summary_text: html_content += f"<p><b>{html_lib.escape(plain_summary_text)}</b></p>"
                copy_chips = "".join(lab_chip_copy_html(it["name"], it["data"]) for it in active_plot_items)
                if copy_chips: html_content += copy_chips + "<br>"
                html_content += html_table(["Name", "Peaks (nm)", "Abs (AU)"],
                                           [(n, f"{w:.1f}", f"{a:.5f}") for n, _, w, a in table_rows],
                                           first_col_colors=[c for _, c, _, _ in table_rows])
                if match_res is not None:
                    html_content += f"<br><p><b>DB 매칭 후보 상위 5개</b> ({'미분' if use_deriv else '원'} 스펙트럼, {fit_min:.0f}–{fit_max:.0f} nm)</p>"
                    html_content += html_table(
                        ["순위", "염료", "형태 점수", "Δλmax (nm)", "타겟 농도", "배율 R²"],
                        [(i, r["염료"], f"{r['형태 점수']:.4f}", f"{r['Δλmax(nm)']:+.0f}", f"{r['타겟 농도(%)']:.0f}%", f"{r['배율 적합 R²']:.3f}")
                         for i, r in match_res.head(5).iterrows()])
                render_copy_button(html_content, "표 + 분석 요약 복사")


# ==========================================
# 모드 2: 혼합 비율 예측 (NNLS)
# ==========================================
elif app_mode == "MIX":
    st.markdown("<div class='page-title'>혼합 비율 예측</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>고른 염료들을 어떤 비율로 섞으면 타겟 스펙트럼이 되는지 계산합니다 (NNLS) · 분석 구간 {fit_min:.0f}–{fit_max:.0f} nm</div>", unsafe_allow_html=True)

    if target_series_sd is not None:
        mix_target_series, mix_target_name, comp_dyes = target_series_sd, target_name, selected_dyes
    else:
        mix_target_series = mix_target_name = None
        comp_dyes = []
        if len(selected_dyes) > 0:
            mix_target_name = selected_dyes[0]
            mix_target_series = combined_df.loc[mix_target_name]
            comp_dyes = selected_dyes[1:]

    if mix_target_series is None:
        empty_state("science", "혼합물 시료가 필요합니다",
                    "SD 파일을 올리거나, DB 염료 중 첫 번째로 고른 것을 타겟으로 씁니다.",
                    [("타겟 준비", "혼합물 SD 파일을 올리세요."),
                     ("구성 염료 선택", "섞였을 것 같은 염료를 2~6개 고르세요."),
                     ("비율 확인", "예측 비율과 재구성 스펙트럼 일치도를 봅니다.")])
    elif len(comp_dyes) == 0:
        st.info(f"'{mix_target_name}'을(를) 만들 구성 염료를 왼쪽 ②에서 골라 주세요."
                + ("" if target_series_sd is not None else " (첫 번째로 고른 염료가 타겟, 두 번째부터 구성 성분)"),
                icon=":material/add_circle:")
    else:
        fit = nnls_fit(mix_target_series, combined_df.loc[comp_dyes], fit_min, fit_max, use_baseline)
        if fit is None:
            st.warning("분석 구간에서 공통 파장이 부족합니다. 고급 설정의 분석 구간을 확인하세요.", icon=":material/warning:")
        else:
            coeffs = fit["coeffs"]
            total_coeff = np.sum(coeffs)
            pcts = (coeffs / total_coeff) * 100 if total_coeff > 0 else np.zeros_like(coeffs)
            r2, rmse = fit["r2"], fit["rmse"]

            nnls_df = pd.DataFrame({"Name": comp_dyes, "Ratio(%)": pcts, "Coefficient": coeffs}).sort_values(by="Ratio(%)", ascending=False)
            nnls_result_str = ", ".join([f"{row['Name']}({row['Ratio(%)']:.1f}%)" for _, row in nnls_df.iterrows() if row['Ratio(%)'] > 0.1]) or "-"
            plain_summary_text = f"예측 혼합비 (Target: {mix_target_name}, R²={r2:.3f}):<br>{nnls_result_str}"

            if np.isfinite(r2) and r2 >= MIX_GOOD_R2:
                verdict = pill("잘 일치", "ok", "check_circle"); vmsg = "재구성 스펙트럼이 타겟과 잘 맞습니다."
            elif np.isfinite(r2) and r2 >= MIX_OK_R2:
                verdict = pill("대체로 일치", "warn", "info"); vmsg = "일부 파장대에서 차이가 있습니다. 잔차 곡선을 확인하세요."
            else:
                verdict = pill("불일치", "bad", "error"); vmsg = "빠진 성분이 있거나 선택한 성분이 실제와 다를 수 있습니다."

            col_l, col_r = st.columns([1, 2.1], gap="medium")
            with col_l:
                with st.container(key="card_mixres"):
                    st.markdown(
                        f"<div class='sec'>{icon('target')}타겟</div>"
                        f"<div style='font-size:20px;font-weight:800;color:{INK};margin-bottom:8px'>{html_lib.escape(str(mix_target_name))}</div>"
                        f"{verdict}<div style='color:{MUTED};font-size:13px;margin:6px 0 4px'>{vmsg}</div>"
                        f"<div class='kpis' style='grid-template-columns:1fr 1fr;margin-top:12px'>"
                        f"<div class='kpi accent'><div class='k'>적합도 R²</div><div class='v'>{r2:.4f}</div></div>"
                        f"<div class='kpi'><div class='k'>RMSE</div><div class='v'>{rmse:.4f}<small>AU</small></div></div></div>",
                        unsafe_allow_html=True)

                    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
                    sec("예측 혼합 비율", "donut_small")
                    rows_html = ""
                    for _, row in nnls_df.iterrows():
                        info = get_color_info(row["Name"], combined_df.loc[row["Name"]])
                        dot = f"<span class='dot' style='background:{info[3]};width:12px;height:12px;border:1px solid rgba(0,0,0,.1)'></span>" if info else ""
                        p = max(0.0, float(row["Ratio(%)"]))
                        rows_html += (f"<div class='ratio-row' style='opacity:{1 if p >= 0.1 else 0.45}'><div><div class='nm'>{dot}{html_lib.escape(str(row['Name']))}</div>"
                                      f"<div class='bar'><i style='width:{p:.1f}%'></i></div></div>"
                                      f"<div class='pct'>{p:.1f}%</div></div>")
                    st.markdown(rows_html, unsafe_allow_html=True)
                    st.caption("비율은 각 기준 스펙트럼 배수(계수)를 합 100%로 환산한 값입니다. 기준 스펙트럼의 측정 농도가 서로 다르면 계수 × 기준 농도로 환산해서 보세요.")
                    with st.expander("계수 원본 보기"):
                        st.dataframe(nnls_df, hide_index=True, use_container_width=True,
                                     column_config={"Ratio(%)": st.column_config.NumberColumn("Ratio", format="%.1f%%"),
                                                    "Coefficient": st.column_config.NumberColumn("Coefficient", format="%.4f")})

                    html_content = f"<h3>[혼합 비율 예측 요약]</h3><p><b>{html_lib.escape(plain_summary_text).replace('&lt;br&gt;', '<br>')}</b></p>"
                    html_content += f"<p>적합도 R² = {r2:.4f}, RMSE = {rmse:.4f} AU (분석 구간 {fit_min:.0f}–{fit_max:.0f} nm)</p>"
                    copy_chips = "".join(lab_chip_copy_html(d, combined_df.loc[d]) for d in [mix_target_name] + comp_dyes)
                    if copy_chips: html_content += copy_chips + "<br>"
                    html_content += html_table(["Component Name", "Ratio (%)", "Coefficient"],
                                               [(r["Name"], f"{r['Ratio(%)']:.1f}%", f"{r['Coefficient']:.4f}") for _, r in nnls_df.iterrows()])
                    render_copy_button(html_content, "표 + 예측 결과 복사")

            with col_r:
                with st.container(key="card_mixchart"):
                    sec("타겟 vs 재구성 스펙트럼" + (" · 베이스라인 보정" if use_baseline else ""), "stacked_line_chart")
                    fig_nnls = go.Figure()
                    fig_nnls.add_trace(go.Scatter(x=fit["wl"], y=fit["Y"], mode='lines', name=f"타겟 ({mix_target_name})", line=dict(color=NAVY, width=1.2)))
                    fig_nnls.add_trace(go.Scatter(x=fit["wl"], y=fit["Y_pred"], mode='lines', name="재구성", line=dict(color="#e4572e", width=1.2, dash='dash')))
                    fig_nnls.add_trace(go.Scatter(x=fit["wl"], y=fit["Y"] - fit["Y_pred"], mode='lines', name="잔차", line=dict(color="#9aa3b2", width=1.2, dash='dot'),
                                                  fill="tozeroy", fillcolor="rgba(154,163,178,0.12)"))
                    style_fig(fig_nnls, height=470, x_range=[fit_min, fit_max])
                    st.plotly_chart(fig_nnls, use_container_width=True, config=PLOT_CONFIG)

                chips = "".join(lab_chip_html(d, combined_df.loc[d]) for d in [mix_target_name] + comp_dyes)
                if chips:
                    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                    with st.container(key="card_mixcolors"):
                        sec("색상 (CIE L*a*b*)", "palette")
                        st.markdown(f"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));column-gap:20px'>{chips}</div>",
                                    unsafe_allow_html=True)
