import streamlit as st
import pandas as pd
import numpy as np
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

# ==========================================
# 🌟 2. 순정 메뉴 버튼 생성 (가장 먼저 작성!) 🌟
# ==========================================
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.button("스펙트럼 비교", icon=":material/bar_chart:", use_container_width=True, 
              type="primary" if app_mode == "SPEC" else "secondary",
              on_click=set_app_mode, args=("SPEC",))
with col_m2:
    st.button("혼합 비율 예측", icon=":material/science:", use_container_width=True, 
              type="primary" if app_mode == "MIX" else "secondary",
              on_click=set_app_mode, args=("MIX",))

# ==========================================
# 🌟 3. 본문 겹침 방지용 강력한 여백 (물리적 차단막) 🌟
# ==========================================
st.markdown("<div style='height: 40px; display: block;'></div>", unsafe_allow_html=True)

# ==========================================
# 🌟 4. 진짜 상단 고정 메뉴바 및 완벽 복구된 CSS 🌟
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
    
    .fixed-header {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 60px;
        background-color: var(--background-color, #ffffff); box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        z-index: 999998; display: flex; align-items: center; padding-left: 20px; border-bottom: 1px solid rgba(128,128,128,0.2);
    }}
    .fixed-header img {{ width: 45px; margin-right: 12px; }}
    .fixed-header h2 {{ margin: 0; padding: 0; font-size: 24px; font-weight: 700; color: var(--text-color); margin-right: 30px; }}
    
    /* 버튼만 상단바로 쏙 들어가는 핵심 CSS */
    [data-testid="stMainBlockContainer"] > div > div:first-child {{
        position: fixed !important; top: 11px !important; left: 290px !important; 
        width: 380px !important; z-index: 999999 !important; background-color: transparent !important;
    }}
    
    .block-container {{ padding-top: 90px !important; }}
    [data-testid="stSidebar"] {{ padding-top: 60px !important; }}
    [data-testid="stSidebarUserContent"] {{ padding-top: 10px !important; padding-bottom: 10px !important; }}
    [data-testid="stSidebarUserContent"] > div {{ gap: 0.5rem !important; }}
    div.element-container {{ margin-bottom: 0 !important; }}
    .stTextInput>div, .stMultiSelect>div, .stFileUploader>div, .stSelectbox>div {{ padding-bottom: 0 !important; }}
    .material-symbols-outlined {{ line-height: 1 !important; vertical-align: middle; }}
    
    /* 화면 흔들림(Jittering) 방지 */
    [data-testid="stAppViewContainer"] {{ overflow-y: scroll !important; }}
</style>

<div class="fixed-header">
    <img src="data:image/png;base64,{logo_base64}" onerror="this.style.display='none'">
    <h2>Ohyoung UV-Vis</h2>
</div>
''', unsafe_allow_html=True)


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


# ==========================================
# 🌟 [다중 파일 및 다중 데이터 추출] 업로드부 🌟
# ==========================================
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
    st.sidebar.markdown("<div style='margin-top: 15px; margin-bottom: 5px; color: var(--primary-color, #2e7af5); font-weight: bold; display: flex; align-items: center;'><span class='material-symbols-outlined' style='margin-right:6px;'>my_location</span>타겟(기준) 스펙트럼 선택</div>", unsafe_allow_html=True)
    target_name = st.sidebar.selectbox("타겟 선택", options=list(uploaded_spectra.keys()), label_visibility="collapsed")
    target_series_sd = uploaded_spectra[target_name]

# ==========================================
# 🎨 비교/구성 성분 선택
# ==========================================
st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 15px 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>palette</span>비교/구성 성분 선택</h3>", unsafe_allow_html=True)

max_sel = 6

selected_db_dyes = st.sidebar.multiselect(
    f"1. DB 염료에서 선택 (최대 {max_sel}개):", 
    options=df.index.tolist(),
    max_selections=max_sel,
    key=f"ms_db_{st.session_state.dye_type}"
)

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

color_palette = ['black', 'red', 'blue', 'purple', 'green', 'orange', 'brown', 'pink']


# ==========================================
# 모드 1: 스펙트럼 비교 분석
# ==========================================
if app_mode == "SPEC":
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("<h2 style='margin-top: 15px; display:flex; align-items:center;'><span class='material-symbols-outlined' style='font-size:32px; margin-right:8px;'>bar_chart</span>스펙트럼 일반 비교 분석</h2>", unsafe_allow_html=True)
        
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

    df_summary = pd.DataFrame()
    copy_text_js = ""

    if len(plot_items) > 0:
        fig = go.Figure()
        table_data = {"Name": [], "Peaks(nm)": [], "Abs(AU)": []}
        target_max_abs = first_match_max_abs = match_name_for_conc = None

        for i, item in enumerate(plot_items):
            name = item["name"]
            series = item["data"]
            color = color_palette[i % len(color_palette)] 
            
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, mode='lines', name=name, 
                line=dict(color=color, width=1.5)
            ))
            
            mask = (series.index >= min_wave) & (series.index <= max_wave)
            if np.any(mask):
                range_series = series[mask]
                p_wave = range_series.idxmax()
                p_abs = range_series.max()
                
                fig.add_trace(go.Scatter(
                    x=[p_wave], y=[p_abs], mode='markers+text', 
                    marker=dict(color=color, size=8),
                    text=[f"{p_wave:.0f}nm<br>({p_abs:.2f})"],
                    textposition="top right", textfont=dict(color=color, size=11),
                    showlegend=False, hoverinfo='skip'
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
            xaxis_title="Wavelength (nm)", yaxis_title="Absorbance (AU)",
            hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            plot_bgcolor='white', paper_bgcolor='white', dragmode="zoom",
            xaxis=dict(showgrid=True, gridcolor='#eaeaea', range=[190, 1100]), 
            yaxis=dict(showgrid=True, gridcolor='#eaeaea')
        )
        
        conc_summary_web = ""
        plain_summary_text = ""
        if target_max_abs is not None and first_match_max_abs is not None and len(plot_items) == 2:
            conc_diff_pct = ((target_max_abs - first_match_max_abs) / first_match_max_abs) * 100
            direction_str = "진합니다" if conc_diff_pct > 0 else "연합니다"
            conc_summary_web = f"- **농도 분석:** {target_name}이 {match_name_for_conc} 대비 약 **{abs(conc_diff_pct):.1f}%** 더 {direction_str}."
            plain_summary_text = f"농도 분석: {target_name}이 {match_name_for_conc} 대비 약 {abs(conc_diff_pct):.1f}% 더 {direction_str}."

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

                # 🌟 [HTML 표 복사 기능] 워드/엑셀에 테이블 형태로 복사되도록 HTML 생성
                html_content = f"<h3>[실무 분석 요약]</h3>"
                if plain_summary_text:
                    html_content += f"<p><b>{plain_summary_text}</b></p>"
                else:
                    html_content += "<p>농도 비교 결과 없음</p>"
                
                html_content += "<table border='1' style='border-collapse: collapse; text-align: center; font-family: sans-serif;'>"
                html_content += "<tr style='background-color: #f2f2f2;'><th>Name</th><th>Peaks (nm)</th><th>Abs (AU)</th></tr>"
                for idx in range(len(table_data["Name"])):
                    html_content += f"<tr><td style='padding: 8px;'>{table_data['Name'][idx]}</td><td style='padding: 8px;'>{table_data['Peaks(nm)'][idx]}</td><td style='padding: 8px;'>{table_data['Abs(AU)'][idx]}</td></tr>"
                html_content += "</table>"
                
                js_html_content = html_content.replace('\n', ' ').replace('"', '\\"').replace("'", "\\'")
                
                # 표 바로 아래에 위치하는 복사 버튼
                copy_btn_html = f"""
                <button onclick="copyToClipboard(this)" style="width: 100%; background-color: var(--primary-color, #2b5ce6); color: white; border: none; padding: 10px 0; border-radius: 8px; font-size: 15px; cursor: pointer; font-weight: bold; display: flex; justify-content: center; align-items: center; gap: 6px; transition: 0.3s; margin-top: -10px;">
                    <svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" width="20" fill="white"><path d="M360-240q-33 0-56.5-23.5T280-320v-480q0-33 23.5-56.5T360-880h360q33 0 56.5 23.5T800-800v480q0 33-23.5 56.5T720-240H360Zm0-80h360v-480H360v480ZM200-80q-33 0-56.5-23.5T120-160v-560h80v560h440v80H200Zm160-240v-480 480Z"/></svg>
                    <span>표 + 분석 요약 복사하기</span>
                </button>
                <script>
                function copyToClipboard(button) {{
                    const htmlText = "{js_html_content}";
                    const tempDiv = document.createElement("div");
                    tempDiv.innerHTML = htmlText;
                    tempDiv.style.position = "absolute";
                    tempDiv.style.left = "-9999px";
                    document.body.appendChild(tempDiv);
                    
                    const selection = window.getSelection();
                    const range = document.createRange();
                    range.selectNodeContents(tempDiv);
                    selection.removeAllRanges();
                    selection.addRange(range);
                    
                    try {{
                        document.execCommand('copy');
                        const iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" width="20" fill="white"><path d="M382-240 154-468l57-57 171 171 367-367 57 57-424 424Z"/></svg>';
                        const originalHtml = button.innerHTML;
                        const originalBg = button.style.backgroundColor;
                        
                        // 팝업 없이 부드럽게 색상과 글자 변경 애니메이션
                        button.innerHTML = iconSvg + '<span>복사 완료! (Ctrl+V)</span>';
                        button.style.backgroundColor = '#28a745'; // 초록색 피드백
                        
                        setTimeout(() => {{
                            button.innerHTML = originalHtml;
                            button.style.backgroundColor = originalBg;
                        }}, 2500);
                    }} catch(e) {{
                        console.error(e);
                    }}
                    
                    document.body.removeChild(tempDiv);
                    selection.removeAllRanges();
                }}
                </script>
                """
                st.components.v1.html(copy_btn_html, height=50)


# ==========================================
# 모드 2: 다성분 혼합 비율 예측 (NNLS)
# ==========================================
elif app_mode == "MIX":
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("<h2 style='margin-top: 15px; display:flex; align-items:center;'><span class='material-symbols-outlined' style='font-size:32px; margin-right:8px;'>science</span>다성분 혼합 비율 예측 (NNLS)</h2>", unsafe_allow_html=True)
        st.caption("선택한 성분 데이터들을 어떤 비율로 섞어야 타겟 스펙트럼이 되는지 분석합니다.")

    mix_target_series = None
    mix_target_name = ""
    comp_dyes = []

    if target_series_sd is not None:
        mix_target_series = target_series_sd
        mix_target_name = target_name
        comp_dyes = selected_dyes
    else:
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
        plain_summary_text = f"예측 혼합비 (Target: {mix_target_name}):<br>{nnls_result_str}"
        
        col_n1, col_n2 = st.columns([1, 2])
        with col_n1:
            st.markdown(f"**Target:** {mix_target_name}")
            st.dataframe(
                nnls_df.style.format({"Ratio(%)": "{:.1f}%", "Coefficient": "{:.4f}"}),
                use_container_width=True, hide_index=True
            )
            
            # 🌟 [HTML 표 복사 기능] 워드/엑셀 표 형태 유지
            html_content = f"<h3>[혼합 비율 예측 요약]</h3>"
            html_content += f"<p><b>{plain_summary_text}</b></p>"
            html_content += "<table border='1' style='border-collapse: collapse; text-align: center; font-family: sans-serif;'>"
            html_content += "<tr style='background-color: #f2f2f2;'><th>Component Name</th><th>Ratio (%)</th><th>Coefficient</th></tr>"
            for _, row in nnls_df.iterrows():
                html_content += f"<tr><td style='padding: 8px;'>{row['Name']}</td><td style='padding: 8px;'>{row['Ratio(%)']:.1f}%</td><td style='padding: 8px;'>{row['Coefficient']:.4f}</td></tr>"
            html_content += "</table>"
            
            js_html_content = html_content.replace('\n', ' ').replace('"', '\\"').replace("'", "\\'")
            
            # 표 바로 아래 위치하는 복사 버튼
            copy_btn_html = f"""
            <button onclick="copyToClipboard(this)" style="width: 100%; background-color: var(--primary-color, #2b5ce6); color: white; border: none; padding: 10px 0; border-radius: 8px; font-size: 15px; cursor: pointer; font-weight: bold; display: flex; justify-content: center; align-items: center; gap: 6px; transition: 0.3s; margin-top: -10px;">
                <svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" width="20" fill="white"><path d="M360-240q-33 0-56.5-23.5T280-320v-480q0-33 23.5-56.5T360-880h360q33 0 56.5 23.5T800-800v480q0 33-23.5 56.5T720-240H360Zm0-80h360v-480H360v480ZM200-80q-33 0-56.5-23.5T120-160v-560h80v560h440v80H200Zm160-240v-480 480Z"/></svg>
                <span>표 + 예측 결과 복사하기</span>
            </button>
            <script>
            function copyToClipboard(button) {{
                const htmlText = "{js_html_content}";
                const tempDiv = document.createElement("div");
                tempDiv.innerHTML = htmlText;
                tempDiv.style.position = "absolute";
                tempDiv.style.left = "-9999px";
                document.body.appendChild(tempDiv);
                
                const selection = window.getSelection();
                const range = document.createRange();
                range.selectNodeContents(tempDiv);
                selection.removeAllRanges();
                selection.addRange(range);
                
                try {{
                    document.execCommand('copy');
                    const iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" width="20" fill="white"><path d="M382-240 154-468l57-57 171 171 367-367 57 57-424 424Z"/></svg>';
                    const originalHtml = button.innerHTML;
                    const originalBg = button.style.backgroundColor;
                    
                    // 팝업 없이 부드럽게 색상과 글자 변경 애니메이션
                    button.innerHTML = iconSvg + '<span>복사 완료! (Ctrl+V)</span>';
                    button.style.backgroundColor = '#28a745'; // 초록색 피드백
                    
                    setTimeout(() => {{
                        button.innerHTML = originalHtml;
                        button.style.backgroundColor = originalBg;
                    }}, 2500);
                }} catch(e) {{
                    console.error(e);
                }}
                
                document.body.removeChild(tempDiv);
                selection.removeAllRanges();
            }}
            </script>
            """
            st.components.v1.html(copy_btn_html, height=50)
        
        with col_n2:
            fig_nnls = go.Figure()
            
            fig_nnls.add_trace(go.Scatter(
                x=common_wvl, y=Y_nnls, mode='lines', name=f"Original ({mix_target_name})", 
                line=dict(color='black', width=1.5)
            ))
            
            fig_nnls.add_trace(go.Scatter(
                x=common_wvl, y=Y_pred, mode='lines', name="Reconstructed (Simulated)", 
                line=dict(color='red', width=1.5, dash='dash')
            ))

            fig_nnls.update_layout(
                title=f"Target vs Reconstructed Spectrum",
                xaxis_title="Wavelength (nm)", yaxis_title="Absorbance (AU)",
                hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                plot_bgcolor='white', paper_bgcolor='white', dragmode="zoom",
                xaxis=dict(showgrid=True, gridcolor='#eaeaea', range=[190, 1100]),
                yaxis=dict(showgrid=True, gridcolor='#eaeaea')
            )
            
            st.plotly_chart(fig_nnls, use_container_width=True, config={'scrollZoom': True})