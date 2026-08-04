import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="UV-Vis 분석기", layout="wide")

# --- 데이터 불러오기 ---
@st.cache_data
def load_data():
    df = pd.read_csv("Final_UV_Data.csv", index_col=0)
    df.columns = df.columns.astype(float)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("오류: 'Final_UV_Data.csv' 파일을 찾을 수 없습니다.")
    st.stop()

# --- 상태 저장 (초기 화면 설정) ---
if "menu" not in st.session_state:
    st.session_state.menu = "비교" 

# 메뉴 변경을 즉시 반영하기 위한 콜백 함수
def change_menu(menu_name):
    st.session_state.menu = menu_name

# ==========================================
# 👈 왼쪽 사이드바 (메뉴 및 조작부)
# ==========================================
st.sidebar.caption("✨ Created by tskwon")
st.sidebar.title("🧪 UV-Vis 분석기")

# 1. 메뉴 버튼
st.sidebar.button(
    "📊 염료 스펙트럼 비교", 
    use_container_width=True, 
    type="primary" if st.session_state.menu == "비교" else "secondary",
    on_click=change_menu, 
    args=("비교",)
)

st.sidebar.button(
    "🔍 비교 염료 매칭하기", 
    use_container_width=True, 
    type="primary" if st.session_state.menu == "매칭" else "secondary",
    on_click=change_menu, 
    args=("매칭",)
)

st.sidebar.markdown("---")

# 2. 공통 설정 (어느 메뉴에서든 파장 구간을 지정할 수 있도록 밖으로 뺌)
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

st.sidebar.markdown("---")

# 3. 선택된 메뉴별 추가 조작부
if st.session_state.menu == "비교":
    selected_dyes = st.sidebar.multiselect(
        "비교할 염료를 선택하세요:", 
        df.index.tolist()
    )

elif st.session_state.menu == "매칭":
    uploaded_file = st.sidebar.file_uploader(
        "측정된 원본 CSV 파일을 올려주세요.", 
        type=['csv']
    )


# ==========================================
# 👉 오른쪽 메인 화면 (결과 출력부)
# ==========================================
if st.session_state.menu == "비교":
    st.title("📊 염료 스펙트럼 다중 비교")
    
    if selected_dyes:
        st.subheader(f"📈 스펙트럼 그래프 ({min_wave:.0f}nm ~ {max_wave:.0f}nm 최대 피크)")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for dye in selected_dyes:
            wavelengths = df.columns.values
            absorbance = df.loc[dye].values
            
            # 그래프 선 그리기
            line = ax.plot(wavelengths, absorbance, label=dye)
            color = line[0].get_color() 
            
            # 지정한 구간 안에서만 데이터 자르기
            mask = (wavelengths >= min_wave) & (wavelengths <= max_wave)
            if np.any(mask):
                range_wave = wavelengths[mask]
                range_abs = absorbance[mask]
                
                # 가장 높은 흡광도(최대 피크) 위치 찾기
                max_idx = np.argmax(range_abs)
                peak_wave = range_wave[max_idx]
                peak_abs = range_abs[max_idx]
                
                # 최대 피크 표시
                ax.plot(peak_wave, peak_abs, "o", color=color, markersize=8)
                ax.text(peak_wave, peak_abs, f" Max: {peak_wave:.0f}nm\n ({peak_abs:.2f})", 
                        fontsize=10, ha='left', va='bottom', color=color, fontweight='bold')

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance (AU)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)

    else:
        st.info("👈 왼쪽 사이드바에서 비교할 염료를 선택해 주세요.")


elif st.session_state.menu == "매칭":
    st.title("🔍 비교 염료 매칭하기")
    
    if uploaded_file is not None:
        try:
            # 원본 CSV 읽기 (인코딩 자동 감지)
            file_bytes = uploaded_file.getvalue()
            encodings = ['utf-8', 'utf-16', 'utf-16-le', 'cp949', 'euc-kr']
            target_series = None
            
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
            
            if target_series is None or target_series.empty:
                st.error("파일을 읽을 수 없습니다. 지원하지 않는 형식이거나 데이터가 없습니다.")
            else:
                # 겹치는 구간 추출 및 오차 계산
                common_wavelengths = df.columns.intersection(target_series.index)
                db_data = df[common_wavelengths]
                target_data = target_series[common_wavelengths]
                
                errors = ((db_data - target_data) ** 2).mean(axis=1)
                top3 = errors.sort_values().head(3)
                
                st.success("✅ 매칭 분석 완료!")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### 🎯 매칭 결과 (Top 3)")
                    for rank, (name, error) in enumerate(top3.items(), 1):
                        st.write(f"**{rank}위:** {name}")
                        st.caption(f"오차율: {error:.5f}")
                
                with col2:
                    best_match = top3.index[0]
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    
                    # Target(업로드 파일)과 1st Match(DB) 그래프 선 그리기
                    wave_vals = common_wavelengths.values
                    targ_vals = target_data.values
                    match_vals = db_data.loc[best_match].values
                    
                    ax2.plot(wave_vals, targ_vals, label="Target (Upload)", linestyle='--', color='black', linewidth=2)
                    ax2.plot(wave_vals, match_vals, label=f"1st Match: {best_match}", color='#ff7f0e', alpha=0.8)
                    
                    # --- 💡 매칭 그래프에도 최대 피크 탐색 적용 ---
                    mask = (wave_vals >= min_wave) & (wave_vals <= max_wave)
                    if np.any(mask):
                        range_wave = wave_vals[mask]
                        
                        # 1. Target 데이터의 최대 피크 찾기
                        range_targ = targ_vals[mask]
                        idx_t = np.argmax(range_targ)
                        p_wave_t = range_wave[idx_t]
                        p_abs_t = range_targ[idx_t]
                        
                        ax2.plot(p_wave_t, p_abs_t, "o", color='black', markersize=8)
                        # 글씨가 겹치지 않게 Target은 살짝 왼쪽 정렬로 표시
                        ax2.text(p_wave_t, p_abs_t, f" Target Max: {p_wave_t:.0f}nm\n ({p_abs_t:.2f})", 
                                 fontsize=9, ha='right', va='bottom', color='black', fontweight='bold')
                        
                        # 2. 1st Match 데이터의 최대 피크 찾기
                        range_match = match_vals[mask]
                        idx_m = np.argmax(range_match)
                        p_wave_m = range_wave[idx_m]
                        p_abs_m = range_match[idx_m]
                        
                        ax2.plot(p_wave_m, p_abs_m, "o", color='#ff7f0e', markersize=8)
                        # 1st Match는 살짝 오른쪽 정렬로 표시
                        ax2.text(p_wave_m, p_abs_m, f" Match Max: {p_wave_m:.0f}nm\n ({p_abs_m:.2f}) ", 
                                 fontsize=9, ha='left', va='bottom', color='#ff7f0e', fontweight='bold')

                    ax2.set_title(f"Target vs Best Match ({min_wave:.0f}nm ~ {max_wave:.0f}nm Max Peak)")
                    ax2.set_xlabel("Wavelength (nm)")
                    ax2.set_ylabel("Absorbance (AU)")
                    ax2.legend()
                    ax2.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig2)
                
        except Exception as e:
            st.error(f"파일 분석 오류: {e}")
    else:
        st.info("👈 왼쪽 사이드바에서 측정된 CSV 파일을 업로드해 주세요.")