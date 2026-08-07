import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="UV-Vis 분석기", layout="wide")

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
# 👈 왼쪽 사이드바 (조작부)
# ==========================================
st.sidebar.caption("✨ Created by tskwon")
st.sidebar.title("🧪 UV-Vis 분석기")

# 1. 데이터베이스(염료 종류) 선택
st.sidebar.subheader("📂 데이터베이스 선택")
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
    st.sidebar.error(f"오류: '{db_file}' 파일을 찾을 수 없습니다.")
    st.stop()

# 2. 파일 업로드 및 타겟 이름 설정
st.sidebar.subheader("📂 파일 업로드 (선택사항)")
uploaded_file = st.sidebar.file_uploader(
    "측정된 원본 CSV 파일을 올려주세요.", 
    type=['csv']
)

target_name = "Target (Upload)"
if uploaded_file is not None:
    target_name = st.sidebar.text_input("📝 업로드 데이터 이름 설정", value="Target (Upload)")

# 3. 비교 염료 수동 선택
st.sidebar.subheader("🎨 비교 염료 선택")
max_sel = 3 if uploaded_file is not None else 4

selected_dyes = st.sidebar.multiselect(
    f"DB 염료 수동 선택 (최대 {max_sel}개):", 
    options=df.index.tolist(),
    max_selections=max_sel,
    key=f"ms_{st.session_state.dye_type}"
)
if uploaded_file is None and not selected_dyes:
    st.sidebar.info("👆 타겟 파일을 올리거나 비교할 염료를 선택하세요.")

# 4. 공통 스펙트럼 설정
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 스펙트럼 설정")
st.sidebar.markdown("**최대 피크 탐색 구간 (nm)**")
col1, col2 = st.sidebar.columns(2)

with col1:
    min_wave_str = st.text_input("최소 파장", value="300")
with col2:
    max_wave_str = st.text_input("최대 파장", value="800")
    
try:
    min_wave = float(min_wave_str)
    max_wave = float(max_wave_str)
except ValueError:
    st.sidebar.error("숫자만 입력해 주세요.")
    min_wave = 300.0
    max_wave = 800.0


# ==========================================
# 👉 오른쪽 메인 화면 (결과 출력부)
# ==========================================

# 타이틀 및 인쇄 버튼 배치할 뼈대 미리 만들기
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.title("📊 UV-Vis 스펙트럼 비교 분석")

# 그릴 데이터 모으기
plot_items = []
target_series = None
best_match = None

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        encodings = ['utf-8', 'utf-16', 'utf-16-le', 'cp949', 'euc-kr']
        for enc in encodings:
            try:
                target_df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, usecols=[0, 1])
                target_df.columns = ["Wavelength", "Absorbance"]
                target_df["Wavelength"] = pd.to_numeric(target_df["Wavelength"], errors="coerce")
                target_df["Absorbance"] = pd.to_numeric(target_df["Absorbance"], errors="coerce")
                target_df.dropna(inplace=True)
                target_df.set_index("Wavelength", inplace=True)
                target_series = target_df["Absorbance"]
                break 
            except Exception:
                continue
        
        if target_series is not None and not target_series.empty:
            plot_items.append({"name": target_name, "data": target_series, "is_target": True})
            common_wavelengths = df.columns.intersection(target_series.index)
            db_data = df[common_wavelengths]
            t_data = target_series[common_wavelengths]
            errors = ((db_data - t_data) ** 2).mean(axis=1)
            top3 = errors.sort_values().head(3)
            best_match = top3.index[0]
            st.success(f"✅ 자동 매칭 분석 완료! (1위: **{best_match}**)")
    except Exception as e:
        st.error(f"파일 분석 오류: {e}")

dyes_to_plot = selected_dyes.copy()
if target_series is not None and len(dyes_to_plot) == 0:
    dyes_to_plot = [best_match]

for dye in dyes_to_plot:
    plot_items.append({"name": dye, "data": df.loc[dye], "is_target": False})

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
    
    # 💡 1. 화면에 출력하기 전에 고화질 이미지(Base64) 추출
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # 🔥 2. 농도 분석 텍스트 사전 생성 (웹용 & 인쇄용)
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
    
    # 💡 3. 웹 화면 출력 (좌표/우그래프 정상 배치)
    col_left, col_right = st.columns([1, 2])
    with col_right:
        st.pyplot(fig) # 웹에 렌더링
        
    with col_left:
        st.markdown("### 💡 실무 분석 요약")
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

    # 💡 4. 인쇄 전용 백그라운드 로직 생성
    table_rows_html = ""
    for idx in range(len(table_data["Name"])):
        c = color_palette[idx] if idx < len(color_palette) else 'black'
        table_rows_html += f"<tr style='color: {c}; font-weight: bold; border-bottom: 1px solid #ddd;'>"
        table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{table_data['Name'][idx]}</td>"
        table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{table_data['Peaks(nm)'][idx]}</td>"
        table_rows_html += f"<td style='padding: 12px; border: 1px solid #ddd;'>{table_data['Abs(AU)'][idx]}</td>"
        table_rows_html += "</tr>"

    # 인쇄용 농도 분석 박스 HTML (내용이 있을 때만 생성)
    summary_box_html = ""
    if conc_summary_print:
        summary_box_html = f"""
        <div style="margin-top: 25px; margin-bottom: 10px; padding: 12px 15px; font-size: 14pt; 
                    background-color: #f8f9fa; border-left: 5px solid #2e7af5; border-radius: 4px;">
            💡 {conc_summary_print}
        </div>
        """

    print_js = f"""
    <style>
    body {{ margin: 0; padding: 0; display: flex; justify-content: flex-end; align-items: center; }}
    .print-btn {{ background-color: #2e7af5; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-size: 15px; cursor: pointer; font-weight: bold; margin-top: 20px; }}
    .print-btn:hover {{ background-color: #1b63d1; }}
    </style>
    
    <button onclick="printReport()" class="print-btn">🖨️ 인쇄 (PDF 저장)</button>
    
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
                <style>
                    @page {{ size: A4 portrait; margin: 15mm; }}
                    body {{ font-family: sans-serif; margin: 0; padding: 0; }}
                    table {{ width: 100%; border-collapse: collapse; font-size: 14pt; text-align: center; border: 1px solid #ddd; margin-top: 20px; }}
                    th {{ background-color: #f4f4f4; padding: 12px; border: 1px solid #ddd; }}
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
    st.info("👈 왼쪽 사이드바에서 측정된 파일을 업로드하거나 비교할 염료를 선택해 주세요.")