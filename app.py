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
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# 🌟 앱 모드 상태 관리 및 상단 버튼 🌟
# ==========================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "SPEC"
def set_app_mode(mode):
    st.session_state.app_mode = mode

def open_mix_with(cands):
    """매칭 후보들을 DB 선택 목록에 넣고 혼합 비율 예측 모드로 전환"""
    st.session_state[f"ms_db_{st.session_state.dye_type}"] = list(cands)[:6]
    st.session_state.app_mode = "MIX"

app_mode = st.session_state.app_mode

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.button("스펙트럼 비교", icon=":material/bar_chart:", use_container_width=True, type="primary" if app_mode == "SPEC" else "secondary", on_click=set_app_mode, args=("SPEC",))
with col_m2:
    st.button("혼합 비율 예측", icon=":material/science:", use_container_width=True, type="primary" if app_mode == "MIX" else "secondary", on_click=set_app_mode, args=("MIX",))

st.markdown("<div style='height: 40px; display: block;'></div>", unsafe_allow_html=True)

# ==========================================
# 🌟 상단 고정 메뉴바 및 UI 커스텀 CSS 🌟
# ==========================================
try:
    with open("logo.png", "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
except Exception:
    logo_base64 = ""

st.markdown(f'''
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
    [data-testid="stHeader"] {{ display: none !important; }}
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarHeader"] {{ display: none !important; height: 0 !important; margin: 0 !important; }}
    .fixed-header {{ position: fixed; top: 0; left: 0; width: 100vw; height: 60px; background-color: var(--background-color, #ffffff); box-shadow: 0px 2px 10px rgba(0,0,0,0.1); z-index: 999998; display: flex; align-items: center; padding-left: 20px; border-bottom: 1px solid rgba(128,128,128,0.2); }}
    .fixed-header img {{ width: 45px; margin-right: 12px; }}
    .fixed-header h2 {{ margin: 0; padding: 0; font-size: 24px; font-weight: 700; color: var(--text-color); margin-right: 30px; }}
    [data-testid="stMainBlockContainer"] > div > div:first-child {{ position: fixed !important; top: 11px !important; left: 290px !important; width: 380px !important; z-index: 999999 !important; background-color: transparent !important; }}
    .block-container {{ padding-top: 90px !important; }}
    [data-testid="stSidebar"] {{ padding-top: 60px !important; }}
    [data-testid="stSidebarUserContent"] {{ padding-top: 10px !important; padding-bottom: 10px !important; }}
    [data-testid="stSidebarUserContent"] > div {{ gap: 0.5rem !important; }}
    div.element-container {{ margin-bottom: 0 !important; }}
    .stTextInput>div, .stMultiSelect>div, .stFileUploader>div, .stSelectbox>div {{ padding-bottom: 0 !important; }}
    .material-symbols-outlined {{ line-height: 1 !important; vertical-align: middle; }}
    [data-testid="stAppViewContainer"] {{ overflow-y: scroll !important; }}
</style>
<div class="fixed-header"><img src="data:image/png;base64,{logo_base64}" onerror="this.style.display='none'"><h2>Ohyoung UV-Vis</h2></div>
''', unsafe_allow_html=True)


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

if "dye_type" not in st.session_state: st.session_state.dye_type = "Reactive"
def change_dye_type(dye_name): st.session_state.dye_type = dye_name

# ==========================================
# 왼쪽 사이드바 (조작부)
# ==========================================
st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 0 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>folder_open</span>데이터베이스 선택</h3>", unsafe_allow_html=True)
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1: st.button("Reactive", use_container_width=True, type="primary" if st.session_state.dye_type == "Reactive" else "secondary", on_click=change_dye_type, args=("Reactive",))
with col_d2: st.button("Disperse", use_container_width=True, type="primary" if st.session_state.dye_type == "Disperse" else "secondary", on_click=change_dye_type, args=("Disperse",))

db_file = "Final_UV_Data_R.csv" if st.session_state.dye_type == "Reactive" else "Final_UV_Data_D.csv"
try:
    df = load_data(db_file)
    ai_model, dye_json_data, name_mapping = load_and_train_ai(df)
except FileNotFoundError:
    st.sidebar.error(f"오류: '{db_file}' 파일을 찾을 수 없습니다.", icon=":material/error:"); st.stop()

st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 10px 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>upload_file</span>파일 업로드 (선택사항)</h3>", unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader("SD 파일을 여러 개 올릴 수 있습니다.", type=['sd'], accept_multiple_files=True)

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

# ==========================================
# 🎯 타겟 드롭다운 선택
# ==========================================
if uploaded_spectra:
    st.sidebar.markdown("<div style='margin-top: 15px; margin-bottom: 5px; color: var(--primary-color, #2e7af5); font-weight: bold; display: flex; align-items: center;'><span class='material-symbols-outlined' style='margin-right:6px;'>my_location</span>타겟(기준) 스펙트럼 선택</div>", unsafe_allow_html=True)
    target_name = st.sidebar.selectbox("타겟 선택", options=list(uploaded_spectra.keys()), label_visibility="collapsed")
    target_series_sd = uploaded_spectra[target_name]

# ==========================================
# 🎨 비교/구성 성분 선택
# ==========================================
st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 15px 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>palette</span>비교/구성 성분 선택</h3>", unsafe_allow_html=True)
max_sel = 6
selected_db_dyes = st.sidebar.multiselect(f"1. DB 염료에서 선택 (최대 {max_sel}개):", options=df.index.tolist(), max_selections=max_sel, key=f"ms_db_{st.session_state.dye_type}")

selected_upload_dyes = []
if uploaded_spectra:
    upload_choices = [k for k in uploaded_spectra.keys() if k != target_name]
    if upload_choices:
        selected_upload_dyes = st.sidebar.multiselect(f"2. 업로드 데이터에서 추가 (최대 {max_sel}개):", options=upload_choices, max_selections=max_sel, key=f"ms_up_{st.session_state.dye_type}")

selected_dyes = (selected_db_dyes + selected_upload_dyes)[:max_sel]

# ==========================================
# ⚙️ 스펙트럼 / 분석 설정
# ==========================================
st.sidebar.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 0 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>tune</span>스펙트럼 설정</h3>", unsafe_allow_html=True)

st.sidebar.markdown("<div style='margin-bottom: 5px;'><b>매칭·혼합 분석 구간 (nm)</b></div>", unsafe_allow_html=True)
col_f1, col_f2 = st.sidebar.columns(2)
with col_f1: fit_min = st.number_input("분석 최소", value=380.0, step=10.0, format="%.0f")
with col_f2: fit_max = st.number_input("분석 최대", value=780.0, step=10.0, format="%.0f")
if fit_min >= fit_max:
    st.sidebar.warning("분석 구간이 올바르지 않아 380~780 nm를 사용합니다.", icon=":material/warning:")
    fit_min, fit_max = 380.0, 780.0
use_baseline = st.sidebar.checkbox(f"베이스라인 보정 (구간 끝 {BASELINE_WIDTH:.0f} nm 평균 차감)", value=True)
use_deriv = st.sidebar.checkbox("1차 미분 스펙트럼으로 매칭 (분산염료 권장)", value=True)

st.sidebar.markdown("<div style='margin: 8px 0 5px 0;'><b>최대 피크 표시 구간 (nm)</b></div>", unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)
with col1: min_wave = st.number_input("최소 파장", value=300.0, step=10.0, format="%.0f")
with col2: max_wave = st.number_input("최대 파장", value=800.0, step=10.0, format="%.0f")

st.sidebar.caption("Created by tskwon :material/science:")
color_palette = ['black', 'red', 'blue', 'purple', 'green', 'orange', 'brown', 'pink']


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

def lab_chip_html(name, series, compact=False):
    """색상 칩 HTML (DB 값 우선, 없으면 업로드 데이터에 한해 AI 예측). 표시할 게 없으면 빈 문자열"""
    w, h, fs, small, mb = (80, 35, 13, 11, 12) if compact else (90, 40, 14, 12, 15)
    safe_name = html_lib.escape(str(name))
    lab = get_lab_rgb_for_dye(name)
    if lab:
        L, a, b, rgb = lab
        wrap = f"margin-bottom: {mb}px;"
        tag = f"<span style='font-size: {small}px; font-weight: normal; color: #666;'>(DB 추출)</span>"
    elif name in uploaded_spectra:
        ai = predict_unknown_lab(series)
        if not ai: return ""
        L, a, b, rgb = ai
        wrap = f"margin-bottom: {mb}px; padding: {10 if compact else 12}px; background-color: #f0f7ff; border-left: {3 if compact else 4}px solid #2e7af5; border-radius: {4 if compact else 6}px;"
        tag = f"<span style='font-size: {small}px; font-weight: bold; color: #2e7af5;'>(✨AI 예측)</span>"
    else:
        return ""
    return (f"<div style='{wrap}'>"
            f"<div style='width: {w}px; height: {h}px; background-color: {rgb}; border-radius: 6px; border: 1px solid #ccc; margin-bottom: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'></div>"
            f"<div style='font-size: {fs}px; font-weight: bold;'>{safe_name} {tag}</div>"
            f"<div style='font-size: {fs}px; color: #333; margin-top: 2px;'>L={L}, a={a}, b={b}</div></div>")

def html_table(headers, rows):
    out = "<table border='1' style='border-collapse: collapse; text-align: center; font-family: sans-serif;'>"
    out += "<tr style='background-color: #f2f2f2;'>" + "".join(f"<th style='padding: 8px;'>{html_lib.escape(str(h))}</th>" for h in headers) + "</tr>"
    for r in rows:
        out += "<tr>" + "".join(f"<td style='padding: 8px;'>{html_lib.escape(str(c))}</td>" for c in r) + "</tr>"
    return out + "</table>"

def render_copy_button(html_content, label):
    payload = json.dumps(html_content)
    components.html(f"""
    <button onclick="copyToClipboard(this)" style="width: 100%; background-color: var(--primary-color, #2b5ce6); color: white; border: none; padding: 10px 0; border-radius: 8px; font-size: 15px; cursor: pointer; font-weight: bold; display: flex; justify-content: center; align-items: center; gap: 6px; transition: 0.3s; margin-top: -10px;">
        <svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" width="20" fill="white"><path d="M360-240q-33 0-56.5-23.5T280-320v-480q0-33 23.5-56.5T360-880h360q33 0 56.5 23.5T800-800v480q0 33-23.5 56.5T720-240H360Zm0-80h360v-480H360v480ZM200-80q-33 0-56.5-23.5T120-160v-560h80v560h440v80H200Zm160-240v-480 480Z"/></svg>
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
            const iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" width="20" fill="white"><path d="M382-240 154-468l57-57 171 171 367-367 57 57-424 424Z"/></svg>';
            const originalHtml = button.innerHTML; const originalBg = button.style.backgroundColor;
            button.innerHTML = iconSvg + '<span>복사 완료! (Ctrl+V)</span>';
            button.style.backgroundColor = '#28a745';
            setTimeout(() => {{ button.innerHTML = originalHtml; button.style.backgroundColor = originalBg; }}, 2500);
        }} catch(e) {{}}
        document.body.removeChild(tempDiv); selection.removeAllRanges();
    }}
    </script>
    """, height=50)

PLOT_LAYOUT = dict(hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40), legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), plot_bgcolor='white', paper_bgcolor='white', dragmode="zoom", xaxis_title="Wavelength (nm)", yaxis_title="Absorbance (AU)")


# ==========================================
# 모드 1: 스펙트럼 비교 분석
# ==========================================
if app_mode == "SPEC":
    st.markdown("<h2 style='margin-top: 15px; display:flex; align-items:center;'><span class='material-symbols-outlined' style='font-size:32px; margin-right:8px;'>bar_chart</span>스펙트럼 일반 비교 분석</h2>", unsafe_allow_html=True)

    plot_items_all = []
    best_matches = []
    match_res = None
    match_issues = []

    if target_series_sd is not None and not target_series_sd.empty:
        plot_items_all.append({"name": target_name, "data": target_series_sd, "is_target": True})
        match_res = rank_matches(target_series_sd, df, fit_min, fit_max, use_baseline, use_deriv)

        if match_res is None:
            st.warning("분석 구간에서 DB와 겹치는 파장이 부족합니다. 사이드바의 분석 구간을 확인하세요.", icon=":material/warning:")
        else:
            match_issues, gap = judge_match(match_res)
            top = match_res.iloc[0]
            mode_label = "미분 스펙트럼" if use_deriv else "원 스펙트럼"
            gap_txt = f"{gap:.4f}" if np.isfinite(gap) else "-"
            if not match_issues:
                st.success(f"추천 DB 염료: **{top['염료']}** · 형태 점수 {top['형태 점수']:.4f} · 2위와 차이 {gap_txt}  ({mode_label}, {fit_min:.0f}–{fit_max:.0f} nm)", icon=":material/check_circle:")
                best_matches = [top["염료"]]
            else:
                issue_md = "\n".join(f"- {s}" for s in match_issues)
                st.warning(f"**1위: {top['염료']}** (형태 점수 {top['형태 점수']:.4f}) — 결과를 그대로 믿기 어렵습니다.\n{issue_md}", icon=":material/warning:")
                best_matches = match_res.head(2)["염료"].tolist()

            with st.expander("매칭 후보 상위 5개", expanded=bool(match_issues)):
                top5 = match_res.head(5)
                st.dataframe(
                    top5.style.format({"형태 점수": "{:.4f}", "Δλmax(nm)": "{:+.0f}", "타겟 농도(%)": "{:.0f}%", "배율 적합 R²": "{:.3f}"}, na_rep="-"),
                    use_container_width=True,
                )
                st.caption("형태 점수: 분석 구간 스펙트럼 모양의 상관계수(1에 가까울수록 유사) · Δλmax: 기준 λmax − 타겟 λmax · "
                           "타겟 농도: 기준을 100%로 볼 때 타겟의 상대 농도 · 배율 적합 R²: 농도만 맞췄을 때 겹치는 정도")

                if match_issues and len(match_res) >= 2:
                    cand_names = top5["염료"].tolist()
                    mix_fit = nnls_fit(target_series_sd, df.loc[cand_names], fit_min, fit_max, use_baseline)
                    if mix_fit is not None:
                        single_r2 = top["배율 적합 R²"]
                        total_c = mix_fit["coeffs"].sum()
                        shares = [(n, c / total_c * 100) for n, c in zip(cand_names, mix_fit["coeffs"]) if total_c > 0 and c / total_c > 0.05]
                        shares.sort(key=lambda x: -x[1])
                        st.markdown(f"**혼합 검토** — 상위 {len(cand_names)}개 후보로 혼합 피팅 시 R² = **{mix_fit['r2']:.3f}** (1위 단독 {single_r2:.3f})")
                        if shares:
                            st.markdown("주요 성분: " + ", ".join(f"{n} ({p:.0f}%)" for n, p in shares))
                        if len(shares) >= 2 and np.isfinite(single_r2) and mix_fit["r2"] - single_r2 >= 0.02:
                            st.info("여러 성분을 섞었을 때 훨씬 잘 맞습니다. 혼합 제품일 가능성이 있습니다.", icon=":material/science:")
                        st.button("이 후보들로 혼합 비율 예측 열기", icon=":material/science:", on_click=open_mix_with, args=(cand_names,))

    dyes_to_plot = selected_dyes.copy()
    if target_series_sd is not None and len(dyes_to_plot) == 0 and len(best_matches) > 0:
        dyes_to_plot = best_matches

    added_names = [item["name"] for item in plot_items_all]
    for dye in dyes_to_plot:
        if dye not in added_names:
            plot_items_all.append({"name": dye, "data": combined_df.loc[dye], "is_target": False})
            added_names.append(dye)

    col_left, col_right = st.columns([1, 2])

    active_plot_items = []
    with col_left:
        st.markdown("<h3 style='display: flex; align-items: center;'><span class='material-symbols-outlined' style='margin-right:8px;'>checklist</span>그래프 표시 선택</h3>", unsafe_allow_html=True)
        for i, item in enumerate(plot_items_all):
            is_checked = st.checkbox(f"✔️ {item['name']}", value=True, key=f"chk_{item['name']}")
            if is_checked:
                item_with_color = item.copy()
                item_with_color["color_idx"] = i
                active_plot_items.append(item_with_color)

        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='display: flex; align-items: center;'><span class='material-symbols-outlined' style='margin-right:8px;'>lightbulb</span>실무 분석 요약</h3>", unsafe_allow_html=True)

    if len(active_plot_items) > 0:
        fig = go.Figure()
        fig.add_vrect(x0=fit_min, x1=fit_max, fillcolor="rgba(46,122,245,0.06)", line_width=0,
                      annotation_text="분석 구간", annotation_position="top left", annotation_font_size=10)
        table_data = {"Name": [], "Peaks(nm)": [], "Abs(AU)": []}
        row_colors = []

        for item in active_plot_items:
            name, series = item["name"], item["data"]
            color = color_palette[item["color_idx"] % len(color_palette)]

            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=name, line=dict(color=color, width=1.5)))

            mask = (series.index >= min_wave) & (series.index <= max_wave)
            if np.any(mask):
                range_series = series[mask].dropna()
                if range_series.empty: continue
                p_wave, p_abs = range_series.idxmax(), range_series.max()
                fig.add_trace(go.Scatter(x=[p_wave], y=[p_abs], mode='markers+text', marker=dict(color=color, size=8), text=[f"{p_wave:.0f}nm<br>({p_abs:.2f})"], textposition="top right", textfont=dict(color=color, size=11), showlegend=False, hoverinfo='skip'))
                table_data["Name"].append(name); table_data["Peaks(nm)"].append(f"{p_wave:.1f}"); table_data["Abs(AU)"].append(f"{p_abs:.5f}")
                row_colors.append(color)

        fig.update_layout(title=f"Spectrum Comparison ({min_wave:.0f}nm ~ {max_wave:.0f}nm Max Peak)", xaxis=dict(showgrid=True, gridcolor='#eaeaea', range=[190, 1100]), yaxis=dict(showgrid=True, gridcolor='#eaeaea'), **PLOT_LAYOUT)

        # 농도 분석: 피크 하나가 아니라 분석 구간 전체를 최소제곱으로 맞춘 배율로 계산
        plain_summary_text = ""
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
                if np.isfinite(r2_k) and r2_k < 0.9:
                    plain_summary_text += " — 스펙트럼 모양 차이가 커서 참고용입니다"
                plain_summary_text += "."

        with col_right:
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        with col_left:
            if plain_summary_text:
                st.write(f"- **{plain_summary_text}**")

            html_lab_part = ""
            for item in active_plot_items:
                chip_html = lab_chip_html(item["name"], item["data"])
                if chip_html:
                    st.markdown(chip_html, unsafe_allow_html=True)
                    html_lab_part += chip_html

            if table_data["Name"]:
                df_summary = pd.DataFrame(table_data)
                df_summary.index = range(1, len(df_summary) + 1)
                def color_rows(row):
                    return [f'color: {row_colors[row.name - 1]}; font-weight: bold;'] * len(row)
                st.table(df_summary.style.apply(color_rows, axis=1))

                html_content = "<h3>[실무 분석 요약]</h3>"
                if plain_summary_text: html_content += f"<p><b>{html_lib.escape(plain_summary_text)}</b></p>"
                if html_lab_part: html_content += html_lab_part + "<br>"
                html_content += html_table(["Name", "Peaks (nm)", "Abs (AU)"],
                                           zip(table_data["Name"], table_data["Peaks(nm)"], table_data["Abs(AU)"]))
                if match_res is not None:
                    html_content += f"<br><p><b>DB 매칭 후보 상위 5개</b> ({'미분' if use_deriv else '원'} 스펙트럼, {fit_min:.0f}–{fit_max:.0f} nm)</p>"
                    html_content += html_table(
                        ["순위", "염료", "형태 점수", "Δλmax (nm)", "타겟 농도", "배율 R²"],
                        [(i, r["염료"], f"{r['형태 점수']:.4f}", f"{r['Δλmax(nm)']:+.0f}", f"{r['타겟 농도(%)']:.0f}%", f"{r['배율 적합 R²']:.3f}")
                         for i, r in match_res.head(5).iterrows()])
                render_copy_button(html_content, "표 + 분석 요약 복사하기")


# ==========================================
# 모드 2: 다성분 혼합 비율 예측 (NNLS)
# ==========================================
elif app_mode == "MIX":
    st.markdown("<h2 style='margin-top: 15px; display:flex; align-items:center;'><span class='material-symbols-outlined' style='font-size:32px; margin-right:8px;'>science</span>다성분 혼합 비율 예측 (NNLS)</h2>", unsafe_allow_html=True)
    st.caption(f"선택한 성분 데이터들을 어떤 비율로 섞어야 타겟 스펙트럼이 되는지 분석합니다. (분석 구간 {fit_min:.0f}–{fit_max:.0f} nm)")

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
        st.info("타겟(혼합물) 데이터가 없습니다. SD 파일을 업로드하거나 DB 염료를 1개 이상 선택하세요.", icon=":material/arrow_back:")
    elif len(comp_dyes) == 0:
        st.warning(f"'{mix_target_name}'(을)를 분석하기 위한 구성 염료(2번째 이후 선택)를 추가로 선택해주세요.", icon=":material/add_circle:")
    else:
        fit = nnls_fit(mix_target_series, combined_df.loc[comp_dyes], fit_min, fit_max, use_baseline)
        if fit is None:
            st.warning("분석 구간에서 공통 파장이 부족합니다. 사이드바의 분석 구간을 확인하세요.", icon=":material/warning:")
        else:
            coeffs = fit["coeffs"]
            total_coeff = np.sum(coeffs)
            pcts = (coeffs / total_coeff) * 100 if total_coeff > 0 else np.zeros_like(coeffs)
            r2, rmse = fit["r2"], fit["rmse"]

            nnls_df = pd.DataFrame({"Name": comp_dyes, "Ratio(%)": pcts, "Coefficient": coeffs}).sort_values(by="Ratio(%)", ascending=False)
            nnls_result_str = ", ".join([f"{row['Name']}({row['Ratio(%)']:.1f}%)" for _, row in nnls_df.iterrows() if row['Ratio(%)'] > 0.1]) or "-"
            plain_summary_text = f"예측 혼합비 (Target: {mix_target_name}, R²={r2:.3f}):<br>{nnls_result_str}"

            col_n1, col_n2 = st.columns([1, 2])
            with col_n1:
                st.markdown(f"**Target:** {mix_target_name}")
                m1, m2 = st.columns(2)
                m1.metric("적합도 R²", f"{r2:.4f}")
                m2.metric("RMSE (AU)", f"{rmse:.4f}")
                if np.isfinite(r2) and r2 >= MIX_GOOD_R2:
                    st.success("재구성 스펙트럼이 타겟과 잘 일치합니다.", icon=":material/check_circle:")
                elif np.isfinite(r2) and r2 >= MIX_OK_R2:
                    st.info("대체로 맞지만 일부 파장대에서 차이가 있습니다. 오른쪽 잔차 그래프를 확인하세요.", icon=":material/info:")
                else:
                    st.warning("잘 맞지 않습니다. 빠진 성분이 있거나 선택한 성분이 실제와 다를 수 있습니다.", icon=":material/warning:")

                html_lab_part = ""
                for d_name in [mix_target_name] + comp_dyes:
                    chip_html = lab_chip_html(d_name, combined_df.loc[d_name], compact=True)
                    if chip_html:
                        st.markdown(chip_html, unsafe_allow_html=True)
                        html_lab_part += chip_html

                st.dataframe(nnls_df.style.format({"Ratio(%)": "{:.1f}%", "Coefficient": "{:.4f}"}), use_container_width=True, hide_index=True)
                st.caption("Ratio는 각 기준 스펙트럼의 배수(Coefficient)를 합 100%로 환산한 값입니다. "
                           "기준 스펙트럼들의 측정 농도가 서로 다르면 실제 배합비와 다르므로, 계수 × 기준 농도로 환산해서 보세요.")

                html_content = f"<h3>[혼합 비율 예측 요약]</h3><p><b>{html_lib.escape(plain_summary_text).replace('&lt;br&gt;', '<br>')}</b></p>"
                html_content += f"<p>적합도 R² = {r2:.4f}, RMSE = {rmse:.4f} AU (분석 구간 {fit_min:.0f}–{fit_max:.0f} nm)</p>"
                if html_lab_part: html_content += html_lab_part + "<br>"
                html_content += html_table(["Component Name", "Ratio (%)", "Coefficient"],
                                           [(r["Name"], f"{r['Ratio(%)']:.1f}%", f"{r['Coefficient']:.4f}") for _, r in nnls_df.iterrows()])
                render_copy_button(html_content, "표 + 예측 결과 복사하기")

            with col_n2:
                fig_nnls = go.Figure()
                fig_nnls.add_trace(go.Scatter(x=fit["wl"], y=fit["Y"], mode='lines', name=f"Original ({mix_target_name})", line=dict(color='black', width=1.5)))
                fig_nnls.add_trace(go.Scatter(x=fit["wl"], y=fit["Y_pred"], mode='lines', name="Reconstructed (Simulated)", line=dict(color='red', width=1.5, dash='dash')))
                fig_nnls.add_trace(go.Scatter(x=fit["wl"], y=fit["Y"] - fit["Y_pred"], mode='lines', name="Residual (Original − Reconstructed)", line=dict(color='gray', width=1, dash='dot')))
                title_suffix = " · 베이스라인 보정" if use_baseline else ""
                fig_nnls.update_layout(title=f"Target vs Reconstructed Spectrum (R²={r2:.4f}{title_suffix})", xaxis=dict(showgrid=True, gridcolor='#eaeaea', range=[fit_min, fit_max]), yaxis=dict(showgrid=True, gridcolor='#eaeaea', zeroline=True, zerolinecolor='#cccccc'), **PLOT_LAYOUT)
                st.plotly_chart(fig_nnls, use_container_width=True, config={'scrollZoom': True})
