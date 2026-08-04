import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

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

# 2. 파일 업로드 (선택사항)
st.sidebar.subheader("📂 파일 업로드 (선택사항)")
uploaded_file = st.sidebar.file_uploader(
    "측정된 원본 CSV 파일을 올려주세요.", 
    type=['csv']
)

# 3. 비교 염료 수동 선택
st.sidebar.subheader("🎨 비교 염료 선택")
# 타겟 파일이 있으면 DB 염료는 3개까지, 없으면 4개까지 선택 가능 (총 4개 선 제한)
max_sel = 3 if uploaded_file is not None else 4

# DB 변경 시 선택 항목이 꼬이지 않도록 key에 dye_type을 포함
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
st.title("📊 UV-Vis 스펙트럼 비교 분석")

# 그릴 데이터 모으기 (최대 4개)
plot_items = []
target_series = None
best_match = None

# 타겟 파일 처리 로직
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
            # 타겟 데이터(검정색)를 1번으로 추가
            plot_items.append({"name": "Target (Upload)", "data": target_series, "is_target": True})
            
            # 1위 매칭 찾기
            common_wavelengths = df.columns.intersection(target_series.index)
            db_data = df[common_wavelengths]
            t_data = target_series[common_wavelengths]
            
            errors = ((db_data - t_data) ** 2).mean(axis=1)
            top3 = errors.sort_values().head(3)
            best_match = top3.index[0]
            
            st.success(f"✅ 자동 매칭 분석 완료! (1위: **{best_match}**)")
            
            # 사용자에게 매칭 결과 Top 3 텍스트로 안내
            with st.expander("🎯 매칭 결과 Top 3 보기"):
                for rank, (name, error) in enumerate(top3.items(), 1):
                    st.write(f"**{rank}위:** {name} (오차율: {error:.5f})")

        else:
            st.error("파일을 읽을 수 없습니다.")
    except Exception as e:
        st.error(f"파일 분석 오류: {e}")

# 수동 선택한 염료 처리 (만약 타겟 파일이 있는데 아무것도 안 골랐다면, 자동으로 1위 염료 추가)
dyes_to_plot = selected_dyes.copy()
if target_series is not None and len(dyes_to_plot) == 0:
    dyes_to_plot = [best_match]

# 선택된 염료들을 그릴 리스트에 차례대로 추가
for dye in dyes_to_plot:
    plot_items.append({"name": dye, "data": df.loc[dye], "is_target": False})

# 본격적인 시각화 및 표 생성
if len(plot_items) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 색상 순서: 1:검정, 2:빨강, 3:파랑, 4:보라
    color_palette = ['black', 'red', 'blue', 'purple']
    
    table_data = {"Name": [], "Peaks(nm)": [], "Abs(AU)": []}
    
    # 농도 비교를 위한 1위(타겟)와 2위(비교군 첫번째) 피크 데이터 저장용 변수
    target_max_abs = None
    first_match_max_abs = None

    for i, item in enumerate(plot_items):
        name = item["name"]
        series = item["data"]
        color = color_palette[i] if i < len(color_palette) else plt.cm.tab10(i)
        
        # 선 그리기
        ax.plot(series.index, series.values, label=name, color=color, linewidth=2)
        
        # 지정된 파장 구간 내 최대 피크 찾기
        mask = (series.index >= min_wave) & (series.index <= max_wave)
        if np.any(mask):
            range_series = series[mask]
            p_wave = range_series.idxmax()
            p_abs = range_series.max()
            
            # 그래프에 점과 텍스트 표시
            ax.plot(p_wave, p_abs, "o", color=color, markersize=8)
            ax.text(p_wave, p_abs, f" {p_wave:.0f}nm\n ({p_abs:.2f})", 
                    fontsize=9, ha='left', va='bottom', color=color, fontweight='bold')
            
            # 표에 넣을 데이터 저장
            table_data["Name"].append(name)
            table_data["Peaks(nm)"].append(f"{p_wave:.1f}")
            table_data["Abs(AU)"].append(f"{p_abs:.5f}")
            
            # 농도 비교용 저장
            if i == 0 and item["is_target"]:
                target_max_abs = p_abs
            elif i == 1 and target_max_abs is not None:
                first_match_max_abs = p_abs
                match_name_for_conc = name

    ax.set_title(f"스펙트럼 비교 ({min_wave:.0f}nm ~ {max_wave:.0f}nm Max Peak)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (AU)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 화면을 반으로 나누어 왼쪽엔 결과, 오른쪽엔 그래프 배치
    col_left, col_right = st.columns([1, 2])
    
    with col_right:
        st.pyplot(fig)
        
    with col_left:
        st.markdown("### 💡 실무 분석 요약")
        
        # 농도 분석 코멘트 (타겟이 있고, 최소 1개의 비교 염료가 있을 때만 출력)
        if target_max_abs is not None and first_match_max_abs is not None:
            conc_diff_pct = ((target_max_abs - first_match_max_abs) / first_match_max_abs) * 100
            if conc_diff_pct > 0:
                st.write(f"- **농도:** [{match_name_for_conc}] 대비 약 **{conc_diff_pct:.1f}%** 더 진함")
            else:
                st.write(f"- **농도:** [{match_name_for_conc}] 대비 약 **{abs(conc_diff_pct):.1f}%** 더 연함")
            st.write("") 
        
        # 결과 요약 테이블 출력
        if table_data["Name"]:
            df_summary = pd.DataFrame(table_data)
            df_summary.index = range(1, len(df_summary) + 1)
            st.table(df_summary)
else:
    st.info("👈 왼쪽 사이드바에서 측정된 파일을 업로드하거나 비교할 염료를 선택해 주세요.")