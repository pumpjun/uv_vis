import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import struct

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="Ohyoung UV-Vis", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# ⭐️ 진짜 상단 고정 메뉴바 (Top Navbar) 및 UI 커스텀 CSS
# ==========================================
# 1. 로고 이미지를 Base64로 인코딩 (T/S Colordata 프로그램과 동일한 방식 적용)
try:
    with open("logo.png", "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
except Exception:
    logo_base64 = ""

# 2. 고정 메뉴바 및 사이드바 초밀착 CSS 주입
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
    /* 1. Streamlit 기본 상단 헤더 숨기기 */
    [data-testid="stHeader"] {{
        display: none !important;
    }}
    
    /* 2. 사이드바 접기 버튼 및 숨겨진 헤더 공간 완전히 삭제 */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarHeader"] {{
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    
    /* 3. 🌟 새로운 상단 고정 메뉴바 디자인 (T/S Colordata 스타일) 🌟 */
    .fixed-header {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 60px;
        background-color: #ffffff;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        z-index: 999998;
        display: flex;
        align-items: center;
        padding-left: 20px;
        border-bottom: 1px solid #eaeaea;
    }}
    .fixed-header img {{
        width: 45px;
        margin-right: 12px;
    }}
    .fixed-header h2 {{
        margin: 0;
        padding: 0;
        font-size: 24px;
        font-weight: 700;
        color: #31333F;
    }}

    /* 4. 본문 상단 여백 설정 */
    .block-container {{
        padding-top: 80px !important; 
    }}
    
    /* 5. 🔥 사이드바 간격 강제 축소 및 맨 위로 밀착 🔥 */
    [data-testid="stSidebar"] {{
        padding-top: 60px !important; /* 커스텀 메뉴바 높이만큼만 띄움 */
    }}
    
    /* 사이드바 내부 패딩 최소화 (맨 위로 붙이기) */
    [data-testid="stSidebarUserContent"] {{
        padding-top: 10px !important; 
        padding-bottom: 10px !important;
    }}
    
    /* 스트림릿 기본 위젯 간격(gap) 강제 축소 */
    [data-testid="stSidebarUserContent"] > div {{
        gap: 0.5rem !important; 
    }}

    /* 각 컴포넌트 사이의 쓸데없는 외부 여백 차단 */
    div.element-container {{
        margin-bottom: 0 !important;
    }}
    
    /* 사이드바 내부 텍스트 인풋 등 기본 폼 패딩 줄이기 */
    .stTextInput>div, .stMultiSelect>div, .stFileUploader>div {{
        padding-bottom: 0 !important;
    }}

    /* 머티리얼 아이콘 정렬 */
    .material-symbols-outlined {{
        line-height: 1 !important;
        vertical-align: middle;
    }}
</style>

<!-- 상단 메뉴바 HTML 렌더링 -->
<div class="fixed-header">
    <img src="data:image/png;base64,{logo_base64}" onerror="this.style.display='none'">
    <h2>Ohyoung UV-Vis</h2>
</div>
""", unsafe_allow_html=True)


# --- 데이터 불러오기 ---
@st.cache_data
def load_data(file_name):
    df = pd.read_csv(file_name, index_col=0)
    df.columns = df.columns.astype(float)
    return df

# --- 상태 저장 (기본 데이터베이스 설정) ---
if "dye_type" not in st.session_state:
    st.session_state.dye_type = "Reactive"

def change_dye_type(dye_name):
    st.session_state.dye_type = dye_name


# ==========================================
# 왼쪽 사이드바 (조작부)
# ==========================================

# 1. 데이터베이스(염료 종류) 선택
st.sidebar.markdown(
    "<h3 style='display: flex; align-items: center; margin: 0 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>folder_open</span>데이터베이스 선택</h3>", 
    unsafe_allow_html=True
)
col_d1, col_d2 = st.sidebar.columns(2)

with col_d1:
    st.button(
        "Reactive", 
        use_container_width=True, 
        type="primary" if st.session_state.dye_type == "Reactive" else "secondary",
        on_click=change_dye_type, 
        args=("Reactive",)
    )
with col_d2:
    st.button(
        "Disperse", 
        use_container_width=True, 
        type="primary" if st.session_state.dye_type == "Disperse" else "secondary",
        on_click=change_dye_type, 
        args=("Disperse",)
    )

db_file = "Final_UV_Data_R.csv" if st.session_state.dye_type == "Reactive" else "Final_UV_Data_D.csv"

try:
    df = load_data(db_file)
except FileNotFoundError:
    st.sidebar.error(f"오류: '{db_file}' 파일을 찾을 수 없습니다.", icon=":material/error:")
    st.stop()

# 2. 파일 업로드 및 타겟 이름 설정 (SD 전용)
st.sidebar.markdown(
    "<h3 style='display: flex; align-items: center; margin: 10px 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>upload_file</span>파일 업로드 (선택사항)</h3>", 
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader(
    "측정된 원본 SD 파일을 올려주세요.", 
    type=['sd']
)

target_name = "Target (Upload)"
if uploaded_file is not None:
    target_name = st.sidebar.text_input("업로드 데이터 이름 설정", value="Target (Upload)")

# 3. 비교 염료 수동 선택
st.sidebar.markdown(
    "<h3 style='display: flex; align-items: center; margin: 10px 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>palette</span>비교 염료 선택</h3>", 
    unsafe_allow_html=True
)
max_sel = 3 if uploaded_file is not None else 4

selected_dyes = st.sidebar.multiselect(
    f"DB 염료 수동 선택 (최대 {max_sel}개):", 
    options=df.index.tolist(),
    max_selections=max_sel,
    key=f"ms_{st.session_state.dye_type}"
)
if uploaded_file is None and not selected_dyes:
    st.sidebar.info("타겟 파일을 올리거나 비교할 염료를 선택하세요.", icon=":material/touch_app:")

# 4. 공통 스펙트럼 설정
st.sidebar.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<h3 style='display: flex; align-items: center; margin: 0 0 5px 0;'><span class='material-symbols-outlined' style='margin-right:8px;'>tune</span>스펙트럼 설정</h3>", 
    unsafe_allow_html=True
)
st.sidebar.markdown("<div style='margin-bottom: 5px;'><b>최대 피크 탐색 구간 (nm)</b></div>", unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)

with col1:
    min_wave_str = st.text_input("최소 파장", value="300")
with col2:
    max_wave_str = st.text_input("최대 파장", value="800")
    
try:
    min_wave = float(min_wave_str)
    max_wave = float(max_wave_str)
except ValueError:
    st.sidebar.error("숫자만 입력해 주세요.", icon=":material/error:")
    min_wave = 300.0
    max_wave = 800.0

# 5. 작성자 캡션
st.sidebar.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
st.sidebar.caption("Created by tskwon :material/science:")


# ==========================================
# 오른쪽 메인 화면 (결과 출력부)
# ==========================================

# 타이틀 및 인쇄 버튼 배치
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.markdown(
        "<h1 style='display: flex; align-items: center; margin-top: 0;'><span class='material-symbols-outlined' style='font-size:36px; margin-right:12px;'>bar_chart</span>UV-Vis 스펙트럼 비교 분석</h1>", 
        unsafe_allow_html=True
    )

# 그릴 데이터 모으기
plot_items = []
target_series = None
best_match = None

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        
        # Agilent 기본 파장 범위 (190nm ~ 1100nm)
        wavelengths = list(range(190, 1101))
        spectrum_bytes_length = len(wavelengths) * 8
        
        # 바이너리 파일 내부의 흡광도 데이터 시작을 알리는 헤더 패턴
        headers = {
            b'\x28\x00\x41\x00\x55\x00\x29\x00': 17, # '( A U ) '
            b'\x28\x41\x55\x29\x00': 5             # '(AU) '
        }
        
        header_found = None
        spacing = None
        for h, s in headers.items():
            if file_bytes.find(h) != -1:
                header_found = h
                spacing = s
                break
                
        if not header_found:
            st.error("SD 파일에서 흡광도 데이터를 찾을 수 없습니다. 올바른 형식의 파일인지 확인해주세요.", icon=":material/error:")
        else:
            # 데이터 시작 위치 탐색
            header_idx = file_bytes.find(header_found)
            start_idx = header_idx + spacing
            end_idx = start_idx + spectrum_bytes_length
            
            # 순수 흡광도 데이터 추출 및 실수로 변환
            spectrum_data = file_bytes[start_idx:end_idx]
            absorbances = [val for val, in struct.iter_unpack('<d', spectrum_data)]
            
            # 그래프 및 분석을 위한 Pandas Series 생성
            target_series = pd.Series(absorbances, index=wavelengths)
            target_series.index.name = "Wavelength"

        # 데이터가 정상적으로 추출되었을 때 매칭 및 그래프 로직 실행
        if target_series is not None and not target_series.empty:
            plot_items.append({"name": target_name, "data": target_series, "is_target": True})
            
            # DB와 공통 파장 추출하여 오차율 계산
            common_wavelengths = df.columns.intersection(target_series.index)
            db_data = df[common_wavelengths]
            t_data = target_series[common_wavelengths]
            
            errors = ((db_data - t_data) ** 2).mean(axis=1)
            top3 = errors.sort_values().head(3)
            best_match = top3.index[0]
            
            st.success(f"자동 매칭 분석 완료! (1위: **{best_match}**)", icon=":material/check_circle:")
            
    except Exception as e:
        st.error(f"파일 분석 중 오류가 발생했습니다: {e}", icon=":material/error:")


# 차트에 그릴 염료 리스트 취합
dyes_to_plot = selected_dyes.copy()
if target_series is not None and len(dyes_to_plot) == 0 and best_match is not None:
    dyes_to_plot = [best_match]

for dye in dyes_to_plot:
    plot_items.append({"name": dye, "data": df.loc[dye], "is_target": False})

# 그래프 그리기 및 분석 표 출력
if len(plot_items) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    color_palette = ['black', 'red', 'blue', 'purple']
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
            ax.text(p_wave, p_abs, f" {p_wave:.0f}nm\n ({p_abs:.2f})", 
                    fontsize=9, ha='left', va='bottom', color=color, fontweight='bold')
            
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
    
    # 1. 화면에 출력하기 전에 고화질 이미지(Base64) 추출
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # 2. 농도 분석 텍스트 사전 생성
    conc_summary_web = ""
    conc_summary_print = ""
    if target_max_abs is not None and first_match_max_abs is not None:
        conc_diff_pct = ((target_max_abs - first_match_max_abs) / first_match_max_abs) * 100
        if conc_diff_pct > 0:
            conc_summary_web = f"- **농도 분석:** {target_name}이 {match_name_for_conc} 대비 약 **{conc_diff_pct:.1f}%** 더 진합니다."
            conc_summary_print = f"<b>농도 분석:</b> {target_name}이 {match_name_for_conc} 대비 약 <b>{conc_diff_pct:.1f}%</b> 더 진합니다."
        else:
            conc_summary_web = f"- **농도 분석:** {target_name}이가 {match_name_for_conc} 대비 약 **{abs(conc_diff_pct):.1f}%** 더 연합니다."
            conc_summary_print = f"<b>농도 분석:</b> {target_name}이 {match_name_for_conc} 대비 약 <b>{abs(conc_diff_pct):.1f}%</b> 더 연합니다."
    
    # 3. 웹 화면 출력
    col_left, col_right = st.columns([1, 2])
    with col_right:
        st.pyplot(fig)
        
    with col_left:
        st.markdown(
            "<h3 style='display: flex; align-items: center;'><span class='material-symbols-outlined' style='margin-right:8px;'>lightbulb</span>실무 분석 요약</h3>", 
            unsafe_allow_html=True
        )
        if conc_summary_web:
            st.write(conc_summary_web)
            st.write("") 
        
        if table_data["Name"]:
            df_summary = pd.DataFrame(table_data)
            df_summary.index = range(1, len(df_summary) + 1)
            def color_rows(row):
                idx = row.name - 1
                color = color_palette[idx] if idx < len(color_palette) else 'black'
                return [f'color: {color}; font-weight: bold;'] * len(row)
            styled_df = df_summary.style.apply(color_rows, axis=1)
            st.table(styled_df)

    # 4. 인쇄 전용 백그라운드 로직 생성
    table_rows_html = ""
    for idx in range(len(table_data["Name"])):
        c = color_palette[idx] if idx < len(color_palette) else 'black'
        table_rows_html += f"<tr style='color: {c}; font-weight: bold; border-bottom: 1px solid #ddd;'>"
        table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{table_data['Name'][idx]}</td>"
        table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{table_data['Peaks(nm)'][idx]}</td>"
        table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{table_data['Abs(AU)'][idx]}</td>"
        table_rows_html += "</tr>"

    summary_box_html = ""
    if conc_summary_print:
        summary_box_html = f"""
        <div style="margin-top: 25px; margin-bottom: 10px; padding: 12px 15px; font-size: 14pt; background-color: #f8f9fa; border-left: 5px solid #2e7af5; border-radius: 4px; display: flex; align-items: center;">
            <span class="material-symbols-outlined" style="margin-right: 8px;">lightbulb</span>
            <span>{conc_summary_print}</span>
        </div>
        """

    print_js = f"""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
    <style>
    body {{ margin: 0; padding: 0; display: flex; justify-content: flex-end; align-items: center; }}
    .print-btn {{ 
        background-color: #2e7af5; 
        color: white; 
        border: none; 
        padding: 10px 20px; 
        border-radius: 6px; 
        font-size: 15px; 
        cursor: pointer; 
        font-weight: bold; 
        margin-top: 20px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .print-btn:hover {{ background-color: #1b63d1; }}
    </style>
    
    <button onclick="printReport()" class="print-btn">
        <span class="material-symbols-outlined" style="font-size: 18px;">print</span>
        인쇄 (PDF 저장)
    </button>
    
    <script>
    function printReport() {{
        const parentDoc = window.parent.document;
        
        let iframe = parentDoc.getElementById('print-iframe');
        if (!iframe) {{
            iframe = parentDoc.createElement('iframe');
            iframe.id = 'print-iframe';
            iframe.style.position = 'absolute';
            iframe.style.width = '0px';
            iframe.style.height = '0px';
            iframe.style.border = 'none';
            parentDoc.body.appendChild(iframe);
        }}
        
        const doc = iframe.contentWindow.document;
        doc.open();
        doc.write(`
            <html>
            <head>
                <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
                <style>
                    @page {{ size: A4 portrait; margin: 15mm; }}
                    body {{ font-family: sans-serif; margin: 0; padding: 0; }}
                    table {{ width: 100%; border-collapse: collapse; font-size: 14pt; text-align: center; border: 1px solid #ddd; margin-top: 20px; }}
                    th {{ background-color: #f4f4f4; padding: 12px; border: 1px solid #ddd; }}
                    .material-symbols-outlined {{ line-height: 1 !important; vertical-align: middle; }}
                </style>
            </head>
            <body>
                <h2 style="text-align: center; margin-bottom: 20px;">UV-Vis 스펙트럼 분석 보고서</h2>
                <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto;">
                {summary_box_html}
                <table>
                    <tr><th>Name</th><th>Peaks(nm)</th><th>Abs(AU)</th></tr>
                    {table_rows_html}
                </table>
            </body>
            </html>
        `);
        doc.close();
        
        setTimeout(() => {{
            iframe.contentWindow.focus();
            iframe.contentWindow.print();
        }}, 500);
    }}
    </script>
    """
    
    # 상단 우측에 버튼 배치
    with col_btn:
        st.components.v1.html(print_js, height=70)

else:
    st.info("왼쪽 사이드바에서 측정된 파일을 업로드하거나 비교할 염료를 선택해 주세요.", icon=":material/arrow_back:")