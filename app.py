import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# 1. 메뉴 버튼 (선택된 메뉴는 primary 타입으로 색상 강조)
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

# 2. 선택된 메뉴에 따라 사이드바 아래쪽 조작부가 바뀜
if st.session_state.menu == "비교":
    st.sidebar.subheader("⚙️ 스펙트럼 설정")
    
    selected_dyes = st.sidebar.multiselect(
        "비교할 염료를 선택하세요:", 
        df.index.tolist()
    )

elif st.session_state.menu == "매칭":
    st.sidebar.subheader("📂 파일 업로드")
    
    uploaded_file = st.sidebar.file_uploader(
        "측정된 CSV 파일을 올려주세요.", 
        type=['csv']
    )


# ==========================================
# 👉 오른쪽 메인 화면 (결과 출력부)
# ==========================================
if st.session_state.menu == "비교":
    st.title("📊 염료 스펙트럼 다중 비교")
    
    if selected_dyes:
        st.subheader("📈 스펙트럼 그래프 (최대 피크 표시)")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for dye in selected_dyes:
            wavelengths = df.columns.values
            absorbance = df.loc[dye].values
            
            # 그래프 선 그리기
            line = ax.plot(wavelengths, absorbance, label=dye)
            color = line[0].get_color() 
            
            # 전체 파장 구간 중 가장 높은 흡광도(최대 피크) 위치 찾기
            max_idx = np.argmax(absorbance)
            peak_wave = wavelengths[max_idx]
            peak_abs = absorbance[max_idx]
            
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
        st.info("👈 왼쪽 메뉴에서 염료를 하나 이상 선택해 주세요.")


elif st.session_state.menu == "매칭":
    st.title("🔍 비교 염료 매칭하기")
    
    if uploaded_file is not None:
        try:
            target_df = pd.read_csv(uploaded_file, usecols=[0, 1])
            target_df.columns = ["Wavelength", "Absorbance"]
            target_df.set_index("Wavelength", inplace=True)
            target_series = target_df["Absorbance"]
            
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
                ax2.plot(common_wavelengths, target_data, label="Target (Upload)", linestyle='--', color='black', linewidth=2)
                ax2.plot(common_wavelengths, db_data.loc[best_match], label=f"1st Match: {best_match}", color='#ff7f0e', alpha=0.8)
                
                ax2.set_title("Target vs Best Match")
                ax2.set_xlabel("Wavelength (nm)")
                ax2.set_ylabel("Absorbance (AU)")
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig2)
                
        except Exception as e:
            st.error(f"파일 분석 오류: {e}")
    else:
        st.info("👈 왼쪽 사이드바에서 측정된 CSV 파일을 업로드해 주세요.")