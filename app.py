import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import struct
from scipy.optimize import nnls
import plotly.graph_objects as go

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="Ohyoung UV-Vis", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🌟 1. 앱 모드 상태 관리 🌟
# ==========================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "SPEC"
    
def set_app_mode(mode):
    st.session_state.app_mode = mode

app_mode = st.session_state.app_mode

# 상태에 따라 버튼에 색상을 켜주는 클래스 변수
active_spec_class = "active" if app_mode == "SPEC" else "inactive"
active_mix_class = "active" if app_mode == "MIX" else "inactive"


# ==========================================
# 🌟 2. 숨겨진 Streamlit 네이티브 버튼 (상태 변경용 트리거) 🌟
# ==========================================
with st.sidebar:
    st.button("BTN_SPEC", on_click=set_app_mode, args=("SPEC",), key="btn_spec")
    st.button("BTN_MIX", on_click=set_app_mode, args=("MIX",), key="btn_mix")

hide_and_trigger_js = """
<script>
    const parent = window.parent.document;
    
    function hideButtons() {
        const buttons = parent.querySelectorAll('button');
        buttons.forEach(btn => {
            if (btn.innerText.includes('BTN_SPEC') || btn.innerText.includes('BTN_MIX')) {
                let container = btn.closest('.element-container');
                if (container) container.style.display = 'none';
            }
        });
    }
    setTimeout(hideButtons, 10);
    setTimeout(hideButtons, 100);

    window.parent.triggerClick = function(targetMode) {
        const targetText = targetMode === 'SPEC' ? 'BTN_SPEC' : 'BTN_MIX';
        const buttons = parent.querySelectorAll('button');
        for (let btn of buttons) {
            if (btn.innerText.includes(targetText)) {
                btn.click();
                break;
            }
        }
    }
</script>
"""
st.components.v1.html(hide_and_trigger_js, height=0)


# ==========================================
# 🌟 3. 진짜 상단 고정 메뉴바 (Top Navbar) 및 UI 커스텀 CSS 🌟
# ==========================================
try:
    with open("logo.png", "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
except Exception:
    logo_base64 = ""

header_html = r'''
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
    /* 기본 헤더 숨기기 */
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarHeader"] { display: none !important; height: 0 !important; margin: 0 !important; }
    
    /* 상단 고정 바 디자인 */
    .fixed-header {
        position: fixed; top: 0; left: 0; width: 100vw; height: 60px;
        background-color: var(--background-color, #ffffff); box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        z-index: 999998; display: flex; align-items: center; padding-left: 20px; border-bottom: 1px solid rgba(128,128,128,0.2);
    }
    .fixed-header img { width: 45px; margin-right: 12px; }
    .fixed-header h2 { margin: 0; padding: 0; font-size: 24px; font-weight: 700; color: var(--text-color); margin-right: 30px; }
    
    /* HTML 버튼 디자인 */
    .top-menu-btn {
        background-color: transparent;
        color: var(--text-color);
        border: 1px solid rgba(128,128,128,0.3);
        padding: 8px 16px;
        margin-right: 12px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    .top-menu-btn:hover {
        border-color: var(--primary-color);
        color: var(--primary-color);
    }
    .top-menu-btn.active {
        background-color: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    .top-menu-btn.active:hover { color: white; }
    
    /* 본문 영역 밀림 방지 */
    .block-container { padding-top: 80px !important; }
    [data-testid="stSidebar"] { padding-top: 60px !important; }
    [data-testid="stSidebarUserContent"] { padding-top: 10px !important; padding-bottom: 10px !important; }
    [data-testid="stSidebarUserContent"] > div { gap: 0.5rem !important; }
    div.element-container { margin-bottom: 0 !important; }
    .stTextInput>div, .stMultiSelect>div, .stFileUploader>div { padding-bottom: 0 !important; }
    .material-symbols-outlined { line-height: 1 !important; vertical-align: middle; }
    [data-testid="stAppViewContainer"] { overflow-y: scroll !important; }
</style>

<div class="fixed-header">
    <img src="data:image/png;base64,LOGO_BASE64" onerror="this.style.display='none'">
    <h2>Ohyoung UV-Vis</h2>
    
    <button class="top-menu-btn CLASS_SPEC" onclick="window.parent.triggerClick('SPEC')">
        <span class="material-symbols-outlined" style="margin-right: 6px; font-size: 18px;">bar_chart</span>스펙트럼 비교
    </button>
    <button class="top-menu-btn CLASS_MIX" onclick="window.parent.triggerClick('MIX')">
        <span class="material-symbols-outlined" style="margin-right: 6px; font-size: 18px;">science</span>혼합 비율 예측
    </button>
</div>
'''

header_html = header_html.replace("LOGO_BASE64", logo_base64)
header_html = header_html.replace("CLASS_SPEC", active_spec_class)
header_html = header_html.replace("CLASS_MIX", active_mix_class)

st.markdown(header_html, unsafe_allow_html=True)


# --- 데이터 불러오기 ---
@st.cache_data
def load_data(file_name):
    df = pd.read_csv(file_name, index_col=0)
    df.columns = df.columns.astype(float)
    return df

if "dye_type" not in st.session_state:
    st.session_state.dye_type = "Reactive"

def change_dye_type(dye_name):
    st.session_state.dye_type = dye_name

# ==========================================
# 왼쪽 사이드바 (조작부)
# ==========================================
st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 0 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>folder_open</span>데이터베이스 선택</h3>", unsafe_allow_html=True)
col_d1, col_d2 = st.sidebar.columns(2)

with col_d1:
    st.button("Reactive", use_container_width=True, type="primary" if st.session_state.dye_type == "Reactive" else "secondary", on_click=change_dye_type, args=("Reactive",))
with col_d2:
    st.button("Disperse", use_container_width=True, type="primary" if st.session_state.dye_type == "Disperse" else "secondary", on_click=change_dye_type, args=("Disperse",))

db_file = "Final_UV_Data_R.csv" if st.session_state.dye_type == "Reactive" else "Final_UV_Data_D.csv"

try:
    df = load_data(db_file)
except FileNotFoundError:
    st.sidebar.error(f"오류: '{db_file}' 파일을 찾을 수 없습니다.", icon=":material/error:")
    st.stop()

st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 10px 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>upload_file</span>파일 업로드 (선택사항)</h3>", unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader("SD 파일을 여러 개 올릴 수 있습니다.", type=['sd'], accept_multiple_files=True)

uploaded_spectra = {}

if uploaded_files:
    wavelengths = np.array(range(190, 1101), dtype=float)
    spectrum_bytes_length = len(wavelengths) * 8
    headers = { b'\x28\x00\x41\x00\x55\x00\x29\x00': 17, b'\x28\x41\x55\x29\x00': 5 }
    
    for file in uploaded_files:
        file_bytes = file.getvalue()
        filename = file.name
        
        for h, spacing in headers.items():
            total_matches = file_bytes.count(h)
            start_search = 0
            count = 1
            while True:
                header_idx = file_bytes.find(h, start_search)
                if header_idx == -1: 
                    break
                
                start_idx = header_idx + spacing
                end_idx = start_idx + spectrum_bytes_length
                
                if end_idx <= len(file_bytes):
                    spectrum_data = file_bytes[start_idx:end_idx]
                    try:
                        absorbances = [val for val, in struct.iter_unpack('<d', spectrum_data)]
                        series = pd.Series(absorbances, index=wavelengths)
                        series.index.name = "Wavelength"
                        
                        if total_matches > 1:
                            spec_name = f"{filename} ({count})"
                        else:
                            spec_name = filename
                            
                        base_name = spec_name
                        dedup = 1
                        while spec_name in uploaded_spectra:
                            spec_name = f"{base_name} ({dedup})"
                            dedup += 1
                            
                        uploaded_spectra[spec_name] = series
                        count += 1
                    except Exception:
                        pass
                
                start_search = header_idx + 1

    if not uploaded_spectra:
        st.sidebar.error("업로드된 파일에서 유효한 스펙트럼을 찾을 수 없습니다.", icon=":material/error:")


# 업로드된 데이터와 기존 DB 병합
if uploaded_spectra:
    uploaded_df = pd.DataFrame(uploaded_spectra).T
    uploaded_df.columns = uploaded_df.columns.astype(float)
    combined_df = pd.concat([df, uploaded_df])
else:
    combined_df = df

target_name = None
target_series_sd = None

# ==========================================
# 🎯 타겟 드롭다운 선택
# ==========================================
if uploaded_spectra:
    st.sidebar.markdown("<div style='margin-top: 15px; margin-bottom: 5px; color: #2e7af5; font-weight: bold;'>🎯 타겟(기준) 스펙트럼 선택</div>", unsafe_allow_html=True)
    target_name = st.sidebar.selectbox("타겟 선택", options=list(uploaded_spectra.keys()), label_visibility="collapsed")
    target_series_sd = uploaded_spectra[target_name]

# ==========================================
# 🎨 1. DB 염료 선택 & 2. 다른 업로드 성분 선택 분리
# ==========================================
st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 15px 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>palette</span>비교/구성 성분 선택</h3>", unsafe_allow_html=True)

max_sel = 6

# 1. DB 염료에서 선택
selected_db_dyes = st.sidebar.multiselect(
    f"1. DB 염료에서 선택 (최대 {max_sel}개):", 
    options=df.index.tolist(),
    max_selections=max_sel,
    key=f"ms_db_{st.session_state.dye_type}"
)

# 2. 업로드 데이터에서 선택 (타겟 본인 제외)
selected_upload_dyes = []
if uploaded_spectra:
    upload_choices = [k for k in uploaded_spectra.keys() if k != target_name]
    if upload_choices:
        selected_upload_dyes = st.sidebar.multiselect(
            f"2. 업로드 데이터에서 추가 (최대 {max_sel}개):", 
            options=upload_choices,
            max_selections=max_sel,
            key=f"ms_up_{st.session_state.dye_type}"
        )

# 최종 합산 리스트
selected_dyes = selected_db_dyes + selected_upload_dyes
if len(selected_dyes) > max_sel:
    st.sidebar.warning(f"최대 {max_sel}개까지만 선택 가능합니다. 초과분은 제외됩니다.")
    selected_dyes = selected_dyes[:max_sel]

if not uploaded_spectra and not selected_dyes:
    st.sidebar.info("타겟 파일을 올리거나 성분 염료를 선택하세요.", icon=":material/touch_app:")

st.sidebar.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 0 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>tune</span>스펙트럼 설정</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='margin-bottom: 5px;'><b>최대 피크 탐색 구간 (nm)</b></div>", unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)

with col1: min_wave_str = st.text_input("최소 파장", value="300")
with col2: max_wave_str = st.text_input("최대 파장", value="800")
    
try:
    min_wave, max_wave = float(min_wave_str), float(max_wave_str)
except ValueError:
    st.sidebar.error("숫자만 입력해 주세요.", icon=":material/error:")
    min_wave, max_wave = 300.0, 800.0

st.sidebar.caption("Created by tskwon :material/science:")

# 공용 차트 색상 팔레트
color_palette = ['black', 'red', 'blue', 'purple', 'green', 'orange', 'brown', 'pink']

# 인쇄용 공통 변수 초기화
img_base64 = ""
summary_box_html = ""
table_rows_html = ""

# ==========================================
# 모드 1: 스펙트럼 비교 분석
# ==========================================
if app_mode == "SPEC":
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("<h2 style='margin-top:0; display:flex; align-items:center;'><span class='material-symbols-outlined' style='font-size:32px; margin-right:8px;'>bar_chart</span>스펙트럼 일반 비교 분석</h2>", unsafe_allow_html=True)
        
    plot_items = []
    best_match = None

    if target_series_sd is not None and not target_series_sd.empty:
        plot_items.append({"name": target_name, "data": target_series_sd, "is_target": True})
        
        common_wavelengths = df.columns.intersection(target_series_sd.index)
        db_data = df[common_wavelengths]
        t_data = target_series_sd[common_wavelengths]
        errors = ((db_data - t_data) ** 2).mean(axis=1)
        best_match = errors.sort_values().head(3).index[0]
        st.success(f"자동 매칭 탐색 (가장 유사한 DB 염료: **{best_match}**)", icon=":material/check_circle:")

    dyes_to_plot = selected_dyes.copy()
    if target_series_sd is not None and len(dyes_to_plot) == 0 and best_match is not None:
        dyes_to_plot = [best_match]

    for dye in dyes_to_plot:
        plot_items.append({"name": dye, "data": combined_df.loc[dye], "is_target": False})

    if len(plot_items) > 0:
        fig = go.Figure()
        table_data = {"Name": [], "Peaks(nm)": [], "Abs(AU)": []}
        target_max_abs = first_match_max_abs = match_name_for_conc = None

        for i, item in enumerate(plot_items):
            name = item["name"]
            series = item["data"]
            color = color_palette[i % len(color_palette)] 
            
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, 
                mode='lines', name=name, 
                line=dict(color=color, width=1.5)
            ))
            
            mask = (series.index >= min_wave) & (series.index <= max_wave)
            if np.any(mask):
                range_series = series[mask]
                p_wave = range_series.idxmax()
                p_abs = range_series.max()
                
                fig.add_trace(go.Scatter(
                    x=[p_wave], y=[p_abs], 
                    mode='markers+text', 
                    marker=dict(color=color, size=8),
                    text=[f"{p_wave:.0f}nm<br>({p_abs:.2f})"],
                    textposition="top right",
                    textfont=dict(color=color, size=11),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                table_data["Name"].append(name)
                table_data["Peaks(nm)"].append(f"{p_wave:.1f}")
                table_data["Abs(AU)"].append(f"{p_abs:.5f}")
                
                if i == 0 and item["is_target"]: target_max_abs = p_abs
                elif i == 1 and target_max_abs is not None:
                    first_match_max_abs = p_abs
                    match_name_for_conc = name

        fig.update_layout(
            title=f"Spectrum Comparison ({min_wave:.0f}nm ~ {max_wave:.0f}nm Max Peak)",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Absorbance (AU)",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            plot_bgcolor='white',
            paper_bgcolor='white',
            dragmode="zoom",
            xaxis=dict(showgrid=True, gridcolor='#eaeaea', range=[190, 1100]), 
            yaxis=dict(showgrid=True, gridcolor='#eaeaea')
        )
        
        # 🔥 에러 방지용 안전장치 (앱 다운 방지)
        try:
            img_bytes = fig.to_image(format="png", engine="kaleido", scale=2)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        except Exception:
            img_base64 = ""
            st.warning("⚠️ 서버 환경에서 인쇄용 이미지 변환 모듈(Kaleido)을 실행할 수 없어 인쇄 기능이 비활성화되었습니다. (화면 그래프는 정상 작동합니다.)")
        
        conc_summary_web = conc_summary_print = ""
        # 타겟과 1순위 매칭(또는 사용자가 직접 1개 선택한 염료)과의 농도 차이 계산
        if target_max_abs is not None and first_match_max_abs is not None and len(plot_items) == 2:
            conc_diff_pct = ((target_max_abs - first_match_max_abs) / first_match_max_abs) * 100
            direction_str = "진합니다" if conc_diff_pct > 0 else "연합니다"
            conc_summary_web = f"- **농도 분석:** {target_name}이 {match_name_for_conc} 대비 약 **{abs(conc_diff_pct):.1f}%** 더 {direction_str}."
            conc_summary_print = f"<b>농도 분석:</b> {target_name}이 {match_name_for_conc} 대비 약 <b>{abs(conc_diff_pct):.1f}%</b> 더 {direction_str}."

        col_left, col_right = st.columns([1, 2])
        with col_right:
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
        with col_left:
            st.markdown("<h3 style='display: flex; align-items: center;'><span class='material-symbols-outlined' style='margin-right:8px;'>lightbulb</span>실무 분석 요약</h3>", unsafe_allow_html=True)
            if conc_summary_web: st.write(conc_summary_web)
            
            if table_data["Name"]:
                df_summary = pd.DataFrame(table_data)
                df_summary.index = range(1, len(df_summary) + 1)
                def color_rows(row):
                    idx = row.name - 1
                    color = color_palette[idx % len(color_palette)]
                    return [f'color: {color}; font-weight: bold;'] * len(row)
                st.table(df_summary.style.apply(color_rows, axis=1))

        # 인쇄용 포맷 세팅
        for idx in range(len(table_data["Name"])):
            c = color_palette[idx % len(color_palette)]
            table_rows_html += f"<tr style='color: {c}; font-weight: bold; border-bottom: 1px solid #ddd;'>"
            table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{table_data['Name'][idx]}</td>"
            table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{table_data['Peaks(nm)'][idx]}</td>"
            table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{table_data['Abs(AU)'][idx]}</td>"
            table_rows_html += "</tr>"
        
        if conc_summary_print:
            summary_box_html = f'''
            <div style="margin-top: 25px; margin-bottom: 10px; padding: 12px 15px; font-size: 14pt; background-color: #f8f9fa; border-left: 5px solid #2e7af5; border-radius: 4px; display: flex; align-items: center;">
                <span class="material-symbols-outlined" style="margin-right: 8px;">lightbulb</span>
                <span>{conc_summary_print}</span>
            </div>
            '''

# ==========================================
# 모드 2: 다성분 혼합 비율 예측 (NNLS)
# ==========================================
elif app_mode == "MIX":
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("<h2 style='margin-top:0; display:flex; align-items:center;'><span class='material-symbols-outlined' style='font-size:32px; margin-right:8px;'>science</span>다성분 혼합 비율 예측 (NNLS)</h2>", unsafe_allow_html=True)
        st.caption("선택한 성분 데이터들을 어떤 비율로 섞어야 타겟 스펙트럼이 되는지 분석합니다.")

    mix_target_series = None
    mix_target_name = ""
    comp_dyes = []

    if target_series_sd is not None:
        mix_target_series = target_series_sd
        mix_target_name = target_name
        comp_dyes = selected_dyes
    else:
        # SD 파일이 아예 없을 때, 첫 번째 선택 염료를 타겟으로 지정
        if len(selected_dyes) > 0:
            mix_target_name = selected_dyes[0]
            mix_target_series = combined_df.loc[mix_target_name]
            comp_dyes = selected_dyes[1:]

    if mix_target_series is None:
        st.info("타겟(혼합물) 데이터가 없습니다. SD 파일을 업로드하거나 DB 염료를 1개 이상 선택하세요.", icon=":material/arrow_back:")
    elif len(comp_dyes) == 0:
        st.warning(f"'{mix_target_name}'(을)를 분석하기 위한 구성 염료(2번째 이후 선택)를 추가로 선택해주세요.", icon=":material/add_circle:")
    else:
        common_wvl = combined_df.columns.intersection(mix_target_series.index)
        
        X_nnls = combined_df.loc[comp_dyes, common_wvl].T.values
        Y_nnls = mix_target_series[common_wvl].values
        
        coeffs, _ = nnls(X_nnls, Y_nnls)
        total_coeff = np.sum(coeffs)
        pcts = (coeffs / total_coeff) * 100 if total_coeff > 0 else np.zeros_like(coeffs)
            
        Y_pred = np.dot(X_nnls, coeffs)
        
        nnls_df = pd.DataFrame({
            "Name": comp_dyes,
            "Ratio(%)": pcts,
            "Coefficient": coeffs
        }).sort_values(by="Ratio(%)", ascending=False)
        
        nnls_result_str = ", ".join([f"{row['Name']}({row['Ratio(%)']:.1f}%)" for _, row in nnls_df.iterrows() if row['Ratio(%)'] > 0.1])
        nnls_summary_print = f"<b>예측 혼합비(NNLS) [Target: {mix_target_name}]:</b> {nnls_result_str}"
        
        col_n1, col_n2 = st.columns([1, 2])
        with col_n1:
            st.markdown(f"**Target:** {mix_target_name}")
            st.dataframe(
                nnls_df.style.format({"Ratio(%)": "{:.1f}%", "Coefficient": "{:.4f}"}),
                use_container_width=True, hide_index=True
            )
        
        with col_n2:
            fig_nnls = go.Figure()
            
            fig_nnls.add_trace(go.Scatter(
                x=common_wvl, y=Y_nnls, 
                mode='lines', name=f"Original ({mix_target_name})", 
                line=dict(color='black', width=1.5)
            ))
            
            fig_nnls.add_trace(go.Scatter(
                x=common_wvl, y=Y_pred, 
                mode='lines', name="Reconstructed (Simulated)", 
                line=dict(color='red', width=1.5, dash='dash')
            ))

            fig_nnls.update_layout(
                title=f"Target vs Reconstructed Spectrum",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Absorbance (AU)",
                hovermode="x unified",
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                plot_bgcolor='white',
                paper_bgcolor='white',
                dragmode="zoom",
                xaxis=dict(showgrid=True, gridcolor='#eaeaea', range=[190, 1100]),
                yaxis=dict(showgrid=True, gridcolor='#eaeaea')
            )
            
            # 🔥 에러 방지용 안전장치 (앱 다운 방지)
            try:
                img_bytes = fig_nnls.to_image(format="png", engine="kaleido", scale=2)
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            except Exception:
                img_base64 = ""

            st.plotly_chart(fig_nnls, use_container_width=True, config={'scrollZoom': True})

        table_rows_html = f"<tr style='background-color:#eee; font-weight:bold;'><td colspan='3'>Target: {mix_target_name}</td></tr>"
        for _, row in nnls_df.iterrows():
            table_rows_html += f"<tr style='border-bottom: 1px solid #ddd;'>"
            table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{row['Name']}</td>"
            table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{row['Ratio(%)']:.1f}%</td>"
            table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{row['Coefficient']:.4f}</td>"
            table_rows_html += "</tr>"
            
        summary_box_html = f'''
        <div style="margin-top: 25px; margin-bottom: 10px; padding: 12px 15px; font-size: 14pt; background-color: #f8f9fa; border-left: 5px solid #ff4b4b; border-radius: 4px; display: flex; align-items: center;">
            <span class="material-symbols-outlined" style="margin-right: 8px;">science</span>
            <span>{nnls_summary_print}</span>
        </div>
        '''

# ==========================================
# 🖨️ 공통 인쇄 (PDF 저장) 버튼 로직
# ==========================================
if img_base64:
    header_th = "<th>Name</th><th>Peaks(nm)</th><th>Abs(AU)</th>" if app_mode == "SPEC" else "<th>Component Name</th><th>Ratio (%)</th><th>Coefficient</th>"
    report_title = "UV-Vis 분석 보고서 (일반 비교)" if app_mode == "SPEC" else "UV-Vis 분석 보고서 (혼합 비율 예측)"
    
    print_js = f'''
    <script>
    function printReport() {{
        const parentDoc = window.parent.document;
        let iframe = parentDoc.getElementById('print-iframe');
        if (!iframe) {{
            iframe = parentDoc.createElement('iframe');
            iframe.id = 'print-iframe';
            iframe.style.position = 'absolute'; iframe.style.width = '0px'; iframe.style.height = '0px'; iframe.style.border = 'none';
            parentDoc.body.appendChild(iframe);
        }}
        const doc = iframe.contentWindow.document;
        doc.open();
        doc.write(`
            <html>
            <head>
                <style>
                    @page {{ size: A4 portrait; margin: 15mm; }}
                    body {{ font-family: sans-serif; margin: 0; padding: 0; }}
                    table {{ width: 100%; border-collapse: collapse; font-size: 14pt; text-align: center; border: 1px solid #ddd; margin-top: 20px; }}
                    th {{ background-color: #f4f4f4; padding: 12px; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h2 style="text-align: center; margin-bottom: 20px;">{report_title}</h2>
                <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto;">
                {summary_box_html}
                <table><tr>{header_th}</tr>{table_rows_html}</table>
            </body>
            </html>
        `);
        doc.close();
        setTimeout(() => {{ iframe.contentWindow.focus(); iframe.contentWindow.print(); }}, 500);
    }}
    </script>
    <div style="display:flex; justify-content:flex-end;">
        <button onclick="printReport()" style="background-color: var(--primary-color, #2e7af5); color: white; border: none; padding: 10px 20px; border-radius: 6px; font-size: 15px; cursor: pointer; font-weight: bold; margin-top: 20px; display: flex; align-items: center; gap: 6px;">
            <span class="material-symbols-outlined" style="font-size:18px;">print</span> 인쇄 (PDF 저장)
        </button>
    </div>
    '''
    with col_btn:
        st.components.v1.html(print_js, height=70)