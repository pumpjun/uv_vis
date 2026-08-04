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

# --- 상태 저장 (초기 화면 및 기본 데이터베이스 설정) ---
if "menu" not in st.session_state:
    st.session_state.menu = "비교" 
if "dye_type" not in st.session_state:
    st.session_state.dye_type = "Disperse"

# 콜백 함수들
def change_menu(menu_name):
    st.session_state.menu = menu_name

def change_dye_type(dye_name):
    st.session_state.dye_type = dye_name

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

# 2. 데이터베이스(염료 종류) 선택 버튼 (가로로 2개 나란히 배치)
st.sidebar.subheader("📂 데이터베이스 선택")
col_d1, col_d2 = st.sidebar.columns(2)

with col_d1:
    st.button(
        "Disperse", 
        use_container_width=True, 
        type="primary" if st.session_state.dye_type == "Disperse" else "secondary",
        on_click=change_dye_type, 
        args=("Disperse",)
    )
with col_d2:
    st.button(
        "Reactive", 
        use_container_width=True, 
        type="primary" if st.session_state.dye_type == "Reactive" else "secondary",
        on_click=change_dye_type, 
        args=("Reactive",)
    )

# 선택된 종류에 따라 읽어올 파일 지정
db_file = "Final_UV_Data_D.csv" if st.session_state.dye_type == "Disperse" else "Final_UV_Data_R.csv"

try:
    df = load_data(db_file)
except FileNotFoundError:
    st.sidebar.error(f"오류: '{db_file}' 파일을 찾을 수 없습니다.")
    st.stop()

# 3. 메뉴별 조작부 (염료 선택 또는 파일 업로드)
if st.session_state.menu == "비교":
    st.sidebar.subheader("🎨 염료 선택")
    selected_dyes = st.sidebar.multiselect(
        "비교할 염료를 선택하세요:", 
        df.index.tolist()
    )
elif st.session_state.menu == "매칭":
    st.sidebar.subheader("📂 파일 업로드")
    uploaded_file = st.sidebar.file_uploader(
        "측정된 원본 CSV 파일을 올려주세요.", 
        type=['csv']
    )

st.sidebar.markdown("---")

# 4. 공통 스펙트럼 설정
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
if st.session_state.menu == "비교":
    st.title("📊 염료 스펙트럼 다중 비교")
    
    if selected_dyes:
        st.subheader(f"📈 스펙트럼 그래프 ({min_wave:.0f}nm ~ {max_wave:.0f}nm 최대 피크)")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for dye in selected_dyes:
            wavelengths = df.columns.values
            absorbance = df.loc[dye].values
            
            line = ax.plot(wavelengths, absorbance, label=dye)
            color = line[0].get_color() 
            
            mask = (wavelengths >= min_wave) & (wavelengths <= max_wave)
            if np.any(mask):
                range_wave = wavelengths[mask]
                range_abs = absorbance[mask]
                
                max_idx = np.argmax(range_abs)
                peak_wave = range_wave[max_idx]
                peak_abs = range_abs[max_idx]
                
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
                common_wavelengths = df.columns.intersection(target_series.index)
                db_data = df[common_wavelengths]
                target_data = target_series[common_wavelengths]
                
                errors = ((db_data - target_data) ** 2).mean(axis=1)
                top3 = errors.sort_values().head(3)
                
                st.success(f"✅ {st.session_state.dye_type} 데이터베이스로 매칭 분석 완료!")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### 🎯 매칭 결과 (Top 3)")
                    for rank, (name, error) in enumerate(top3.items(), 1):
                        st.write(f"**{rank}위:** {name}")
                        st.caption(f"오차율: {error:.5f}")
                
                with col2:
                    best_match = top3.index[0]
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    
                    wave_vals = common_wavelengths.values
                    targ_vals = target_data.values
                    match_vals = db_data.loc[best_match].values
                    
                    ax2.plot(wave_vals, targ_vals, label="Target (Upload)", linestyle='--', color='black', linewidth=2)
                    ax2.plot(wave_vals, match_vals, label=f"1st Match: {best_match}", color='#ff7f0e', alpha=0.8)
                    
                    mask = (wave_vals >= min_wave) & (wave_vals <= max_wave)
                    if np.any(mask):
                        range_wave = wave_vals[mask]
                        
                        range_targ = targ_vals[mask]
                        idx_t = np.argmax(range_targ)
                        p_wave_t = range_wave[idx_t]
                        p_abs_t = range_targ[idx_t]
                        
                        ax2.plot(p_wave_t, p_abs_t, "o", color='black', markersize=8)
                        ax2.text(p_wave_t, p_abs_t, f" Target Max: {p_wave_t:.0f}nm\n ({p_abs_t:.2f})", 
                                 fontsize=9, ha='right', va='bottom', color='black', fontweight='bold')
                        
                        range_match = match_vals[mask]
                        idx_m = np.argmax(range_match)
                        p_wave_m = range_wave[idx_m]
                        p_abs_m = range_match[idx_m]
                        
                        ax2.plot(p_wave_m, p_abs_m, "o", color='#ff7f0e', markersize=8)
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