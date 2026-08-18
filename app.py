import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import struct
from scipy.optimize import nnls

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="Ohyoung UV-Vis", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🌟 1. 메뉴 버튼을 무조건 "가장 먼저" 렌더링 🌟
# (아래 CSS가 첫 번째 요소만 정확하게 잡아서 상단바로 이동시킵니다. 제목이 겹치는 현상 완벽 차단!)
# ==========================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "SPEC"
    
def set_app_mode(mode):
    st.session_state.app_mode = mode

app_mode = st.session_state.app_mode

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
# 🌟 2. 진짜 상단 고정 메뉴바 (Top Navbar) 및 UI 커스텀 CSS 🌟
# ==========================================
try:
    with open("logo.png", "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
except Exception:
    logo_base64 = ""

st.markdown(f'''
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
    /* 기본 헤더 숨기기 */
    [data-testid="stHeader"] {{ display: none !important; }}
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarHeader"] {{ display: none !important; height: 0 !important; margin: 0 !important; }}
    
    /* 상단 고정 바 디자인 */
    .fixed-header {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 60px;
        background-color: #ffffff; box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        z-index: 999998; display: flex; align-items: center; padding-left: 20px; border-bottom: 1px solid #eaeaea;
    }}
    .fixed-header img {{ width: 45px; margin-right: 12px; }}
    .fixed-header h2 {{ margin: 0; padding: 0; font-size: 24px; font-weight: 700; color: #31333F; }}
    
    /* 🌟 핵심 해결: 첫 번째로 렌더링된 "버튼 구역"만 콕 집어서 상단바로 이동 🌟 */
    [data-testid="stMain"] [data-testid="block-container"] > div > div.element-container:first-child {{
        position: fixed !important;
        top: 11px !important;
        left: 310px !important; 
        width: 380px !important;
        z-index: 999999 !important;
    }}
    
    /* 본문 영역 밀림 방지 */
    .block-container {{ padding-top: 80px !important; }}
    
    /* 사이드바 여백 최적화 */
    [data-testid="stSidebar"] {{ padding-top: 60px !important; }}
    [data-testid="stSidebarUserContent"] {{ padding-top: 10px !important; padding-bottom: 10px !important; }}
    [data-testid="stSidebarUserContent"] > div {{ gap: 0.5rem !important; }}
    div.element-container {{ margin-bottom: 0 !important; }}
    .stTextInput>div, .stMultiSelect>div, .stFileUploader>div {{ padding-bottom: 0 !important; }}
    .material-symbols-outlined {{ line-height: 1 !important; vertical-align: middle; }}
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

st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 10px 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>upload_file</span>파일 업로드 (선택사항)</h3>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("측정된 원본 SD 파일을 올려주세요.", type=['sd'])

target_name = "Target (Upload)"
if uploaded_file is not None:
    target_name = st.sidebar.text_input("업로드 데이터 이름 설정", value="Target (Upload)")

st.sidebar.markdown("<h3 style='display: flex; align-items: center; margin: 10px 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>palette</span>염료 선택</h3>", unsafe_allow_html=True)

max_sel = 6
if app_mode == "SPEC":
    dye_label = f"비교할 DB 염료 수동 선택 (최대 {max_sel}개):"
    dye_help = "그래프에 겹쳐서 비교할 염료를 선택하세요."
else:
    dye_label = f"DB 염료 수동 선택 (최대 {max_sel}개):"
    dye_help = "SD파일이 없으면 첫 번째 선택 염료가 '타겟(Black)'이 되고, 두 번째부터 '혼합 성분'이 됩니다."

selected_dyes = st.sidebar.multiselect(
    dye_label, 
    options=df.index.tolist(),
    max_selections=max_sel,
    help=dye_help,
    key=f"ms_{st.session_state.dye_type}"
)

if uploaded_file is None and not selected_dyes:
    st.sidebar.info("타겟 파일을 올리거나 비교할 염료를 선택하세요.", icon=":material/touch_app:")

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

# ==========================================
# SD 파일 공통 파싱 로직
# ==========================================
target_series_sd = None
if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        wavelengths = list(range(190, 1101))
        spectrum_bytes_length = len(wavelengths) * 8
        headers = { b'\x28\x00\x41\x00\x55\x00\x29\x00': 17, b'\x28\x41\x55\x29\x00': 5 }
        header_found, spacing = None, None
        for h, s in headers.items():
            if file_bytes.find(h) != -1:
                header_found, spacing = h, s
                break
                
        if not header_found:
            st.error("SD 파일에서 흡광도 데이터를 찾을 수 없습니다.", icon=":material/error:")
        else:
            header_idx = file_bytes.find(header_found)
            start_idx = header_idx + spacing
            end_idx = start_idx + spectrum_bytes_length
            spectrum_data = file_bytes[start_idx:end_idx]
            absorbances = [val for val, in struct.iter_unpack('<d', spectrum_data)]
            target_series_sd = pd.Series(absorbances, index=np.array(wavelengths, dtype=float))
            target_series_sd.index.name = "Wavelength"
    except Exception as e:
        st.error(f"파일 분석 중 오류가 발생했습니다: {e}", icon=":material/error:")

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
        st.success(f"자동 매칭 탐색 (가장 유사한 단일 염료: **{best_match}**)", icon=":material/check_circle:")

    dyes_to_plot = selected_dyes.copy()
    if target_series_sd is not None and len(dyes_to_plot) == 0 and best_match is not None:
        dyes_to_plot = [best_match]

    for dye in dyes_to_plot:
        plot_items.append({"name": dye, "data": df.loc[dye], "is_target": False})

    if len(plot_items) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        color_palette = ['black', 'red', 'blue', 'purple', 'green', 'orange']
        table_data = {"Name": [], "Peaks(nm)": [], "Abs(AU)": []}
        target_max_abs = first_match_max_abs = match_name_for_conc = None

        for i, item in enumerate(plot_items):
            name = item["name"]
            series = item["data"]
            color = color_palette[i] if i < len(color_palette) else plt.cm.tab10(i)
            
            ax.plot(series.index, series.values, label=name, color=color, linewidth=2)
            mask = (series.index >= min_wave) & (series.index <= max_wave)
            if np.any(mask):
                range_series = series[mask]
                p_wave = range_series.idxmax()
                p_abs = range_series.max()
                ax.plot(p_wave, p_abs, "o", color=color, markersize=8)
                
                peak_label = f" {p_wave:.0f}nm\n ({p_abs:.2f})"
                ax.text(p_wave, p_abs, peak_label, fontsize=9, ha='left', va='bottom', color=color, fontweight='bold')
                
                table_data["Name"].append(name)
                table_data["Peaks(nm)"].append(f"{p_wave:.1f}")
                table_data["Abs(AU)"].append(f"{p_abs:.5f}")
                
                if i == 0 and item["is_target"]: target_max_abs = p_abs
                elif i == 1 and target_max_abs is not None:
                    first_match_max_abs = p_abs
                    match_name_for_conc = name

        ax.set_title(f"Spectrum Comparison ({min_wave:.0f}nm ~ {max_wave:.0f}nm Max Peak)")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance (AU)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        conc_summary_web = conc_summary_print = ""
        if target_max_abs is not None and first_match_max_abs is not None and len(selected_dyes) == 1:
            conc_diff_pct = ((target_max_abs - first_match_max_abs) / first_match_max_abs) * 100
            direction_str = "진합니다" if conc_diff_pct > 0 else "연합니다"
            conc_summary_web = f"- **단일 농도 분석:** {target_name}이 {match_name_for_conc} 대비 약 **{abs(conc_diff_pct):.1f}%** 더 {direction_str}."
            conc_summary_print = f"<b>농도 분석:</b> {target_name}이 {match_name_for_conc} 대비 약 <b>{abs(conc_diff_pct):.1f}%</b> 더 {direction_str}."

        col_left, col_right = st.columns([1, 2])
        with col_right:
            st.pyplot(fig)
            
        with col_left:
            st.markdown("<h3 style='display: flex; align-items: center;'><span class='material-symbols-outlined' style='margin-right:8px;'>lightbulb</span>실무 분석 요약</h3>", unsafe_allow_html=True)
            if conc_summary_web: st.write(conc_summary_web)
            
            if table_data["Name"]:
                df_summary = pd.DataFrame(table_data)
                df_summary.index = range(1, len(df_summary) + 1)
                def color_rows(row):
                    idx = row.name - 1
                    color = color_palette[idx] if idx < len(color_palette) else 'black'
                    return [f'color: {color}; font-weight: bold;'] * len(row)
                st.table(df_summary.style.apply(color_rows, axis=1))

        # 인쇄용 포맷 세팅
        for idx in range(len(table_data["Name"])):
            c = color_palette[idx] if idx < len(color_palette) else 'black'
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
        st.caption("선택한 단일 염료들을 어떤 비율로 섞어야 타겟 스펙트럼이 되는지 분석합니다.")

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
            mix_target_series = df.loc[mix_target_name]
            comp_dyes = selected_dyes[1:]

    if mix_target_series is None:
        st.info("타겟(혼합물) 데이터가 없습니다. SD 파일을 업로드하거나 DB 염료를 1개 이상 선택하세요.", icon=":material/arrow_back:")
    elif len(comp_dyes) == 0:
        st.warning(f"'{mix_target_name}'(을)를 분석하기 위한 구성 염료(2번째 이후 선택)를 추가로 선택해주세요.", icon=":material/add_circle:")
    else:
        common_wvl = df.columns.intersection(mix_target_series.index)
        X_nnls = df.loc[comp_dyes, common_wvl].T.values
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
            fig_nnls, ax_nnls = plt.subplots(figsize=(8, 3.5))
            ax_nnls.plot(common_wvl, Y_nnls, label=f"Original ({mix_target_name})", color="black", linewidth=2)
            ax_nnls.plot(common_wvl, Y_pred, label="Reconstructed (Simulated)", color="red", linestyle="--", linewidth=2)
            ax_nnls.set_title(f"Target vs Reconstructed Spectrum")
            ax_nnls.set_xlabel("Wavelength (nm)")
            ax_nnls.set_ylabel("Absorbance (AU)")
            ax_nnls.legend()
            ax_nnls.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_nnls)

            buf = io.BytesIO()
            fig_nnls.savefig(buf, format="png", bbox_inches='tight', dpi=300)
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 인쇄용 포맷 세팅
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
        <button onclick="printReport()" style="background-color: #2e7af5; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-size: 15px; cursor: pointer; font-weight: bold; margin-top: 20px; display: flex; align-items: center; gap: 6px;">
            <span class="material-symbols-outlined" style="font-size:18px;">print</span> 인쇄 (PDF 저장)
        </button>
    </div>
    '''
    with col_btn:
        st.components.v1.html(print_js, height=70)