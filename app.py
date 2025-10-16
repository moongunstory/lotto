"""
로또 VIP 분석 프로그램 - Streamlit GUI
데이터 수집, 패턴 분석, 번호 추천, 투자 시뮬레이션 통합
"""

import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetch.lotto_fetcher import LottoFetcher
from analysis.lotto_pattern_analyzer import LottoPatternAnalyzer
from analysis.lotto_recommender import LottoRecommender
from analysis.lotto_visualizer import LottoVisualizer


# 페이지 설정
st.set_page_config(
    page_title="🎯 로또 VIP 분석 프로그램",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #4A90E2;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #7B68EE;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #50C878;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fffacd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FFD700;
        margin: 1rem 0;
    }
    .number-display {
        font-size: 2rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# 세션 상태 초기화
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []


def load_data():
    """데이터 로드 및 초기화"""
    try:
        fetcher = LottoFetcher()
        analyzer = LottoPatternAnalyzer()
        recommender = LottoRecommender()
        visualizer = LottoVisualizer()
        
        st.session_state.fetcher = fetcher
        st.session_state.analyzer = analyzer
        st.session_state.recommender = recommender
        st.session_state.visualizer = visualizer
        st.session_state.data_loaded = True
        
        return True
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return False


def main():
    """메인 함수"""
    
    # 헤더
    st.markdown('<div class="main-header">🎯 로또 VIP 분석 프로그램</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/4A90E2/FFFFFF?text=LOTTO+VIP", use_container_width=True)
        st.markdown("### 📊 메뉴")
        
        tab_selection = st.radio(
            "기능 선택",
            ["🏠 홈", "📥 데이터 수집", "📊 패턴 분석", "🎲 번호 추천", "💰 투자 시뮬레이션"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ 정보")
        st.info("로또 6/45 당첨 확률:\n- 1등: 1/8,145,060\n- 2등: 1/1,357,510\n- 3등: 1/35,724")
    
    # 데이터 로드
    if not st.session_state.data_loaded:
        with st.spinner("🔄 데이터를 불러오는 중..."):
            if not load_data():
                st.stop()
    
    # 탭별 콘텐츠
    if tab_selection == "🏠 홈":
        show_home()
    elif tab_selection == "📥 데이터 수집":
        show_data_collection()
    elif tab_selection == "📊 패턴 분석":
        show_pattern_analysis()
    elif tab_selection == "🎲 번호 추천":
        show_number_recommendation()
    elif tab_selection == "💰 투자 시뮬레이션":
        show_investment_simulation()


def show_home():
    """홈 화면"""
    st.markdown('<div class="sub-header">🏠 환영합니다!</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📥 데이터 수집")
        st.markdown("동행복권 API를 통해 최신 로또 데이터를 자동으로 수집합니다.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📊 패턴 분석")
        st.markdown("출현 빈도, 홀짝 비율, 구간 분포 등 다양한 통계를 시각화합니다.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🎲 번호 추천")
        st.markdown("사용자 정의 필터를 적용하여 최적의 번호를 추천합니다.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 최신 회차 정보
    try:
        df = st.session_state.analyzer.df
        latest = df.iloc[-1]
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 최신 회차: {int(latest['draw_no'])}회")
        st.markdown(f"**추첨일**: {latest['date']}")
        
        numbers = [int(latest[f'n{i}']) for i in range(1, 7)]
        bonus = int(latest['bonus'])
        
        st.markdown(f"**당첨번호**: {' - '.join(map(str, numbers))} + 보너스 {bonus}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.warning("⚠️ 데이터를 먼저 수집해주세요.")


def show_data_collection():
    """데이터 수집 탭"""
    st.markdown('<div class="sub-header">📥 데이터 수집</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📊 데이터 현황")
        
        try:
            df = st.session_state.analyzer.df
            st.metric("총 회차 수", f"{len(df):,}회")
            st.metric("최신 회차", f"{int(df['draw_no'].max())}회")
            st.metric("데이터 기간", f"{df['date'].min()} ~ {df['date'].max()}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 최근 10회차 표시
            st.markdown("### 📋 최근 10회차 데이터")
            recent_df = df[['draw_no', 'date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'bonus']].tail(10)
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.warning("⚠️ 데이터가 없습니다. 수집을 시작해주세요.")
    
    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚙️ 작업")
        
        if st.button("🔄 데이터 갱신", use_container_width=True, type="primary"):
            with st.spinner("데이터를 갱신하는 중..."):
                try:
                    df = st.session_state.fetcher.update_data()
                    st.session_state.analyzer.load_data()
                    st.success(f"✅ 갱신 완료! 총 {len(df)}개 회차")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 갱신 실패: {e}")
        
        if st.button("📥 전체 재수집", use_container_width=True):
            with st.spinner("전체 데이터를 수집하는 중... (시간이 걸릴 수 있습니다)"):
                try:
                    df = st.session_state.fetcher.fetch_all_data()
                    st.session_state.analyzer.load_data()
                    st.success(f"✅ 수집 완료! 총 {len(df)}개 회차")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 수집 실패: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)


def show_pattern_analysis():
    """패턴 분석 탭"""
    st.markdown('<div class="sub-header">📊 패턴 분석</div>', unsafe_allow_html=True)
    
    analyzer = st.session_state.analyzer
    visualizer = st.session_state.visualizer
    
    # 강조할 번호 입력
    highlight_input = st.text_input("🎯 강조할 번호 입력 (쉼표로 구분, 예: 7,27,31)", "")
    highlight_numbers = []
    if highlight_input:
        try:
            highlight_numbers = [int(n.strip()) for n in highlight_input.split(',') if n.strip()]
        except:
            st.warning("⚠️ 올바른 번호 형식을 입력해주세요.")
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📊 출현 빈도", "📈 최근 추세", "🎯 홀짝 분석", "📍 구간 분포"])
    
    with tab1:
        st.markdown("### 📊 전체 회차 번호별 출현 빈도")
        freq_df = analyzer.get_number_frequency()
        
        fig = visualizer.plot_number_frequency(freq_df, highlight_numbers)
        st.pyplot(fig)
        plt.close()
        
        # 통계 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            most_common = freq_df.nlargest(1, '출현횟수').iloc[0]
            st.metric("최다 출현 번호", f"{int(most_common['번호'])}번", f"{int(most_common['출현횟수'])}회")
        with col2:
            least_common = freq_df.nsmallest(1, '출현횟수').iloc[0]
            st.metric("최소 출현 번호", f"{int(least_common['번호'])}번", f"{int(least_common['출현횟수'])}회")
        with col3:
            st.metric("평균 출현", f"{freq_df['출현횟수'].mean():.1f}회", "")
    
    with tab2:
        st.markdown("### 📈 최근 30회차 출현 추세")
        
        fig = visualizer.plot_recent_trend(analyzer, 30, highlight_numbers)
        st.pyplot(fig)
        plt.close()
        
        recent_freq = analyzer.get_recent_frequency(30)
        if recent_freq:
            sorted_recent = sorted(recent_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            st.markdown("**🔥 최근 30회 TOP 10 번호:**")
            
            cols = st.columns(5)
            for i, (num, count) in enumerate(sorted_recent):
                with cols[i % 5]:
                    st.metric(f"{num}번", f"{count}회")
    
    with tab3:
        st.markdown("### 🎯 홀짝 비율 분석")
        
        odd_even_data = analyzer.get_odd_even_ratio()
        
        fig = visualizer.plot_odd_even_distribution(odd_even_data)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("**📊 홀짝 비율 분포:**")
        for ratio, percentage in odd_even_data['percentage'].items():
            st.progress(percentage / 100, text=f"홀{ratio}:짝{6-ratio} - {percentage}%")
    
    with tab4:
        st.markdown("### 📍 번호 구간별 분포")
        
        range_data = analyzer.get_range_distribution()
        
        fig = visualizer.plot_range_distribution(range_data)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("**📊 구간별 출현 비율:**")
        for range_name, percentage in range_data['percentages'].items():
            st.progress(percentage / 100, text=f"{range_name} 구간 - {percentage}%")


def show_number_recommendation():
    """번호 추천 탭"""
    st.markdown('<div class="sub-header">🎲 번호 추천</div>', unsafe_allow_html=True)
    
    recommender = st.session_state.recommender
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### ⚙️ 필터 설정")
        
        # 필터 체크박스
        remove_consecutive = st.checkbox("❌ 연속번호 제거", value=False)
        if remove_consecutive:
            consecutive_level = st.radio(
                "연속번호 기준",
                [2, 3, 6],
                format_func=lambda x: f"{x}개 이상 연속" if x < 6 else "완전 연속(1~6)",
                horizontal=True
            )
        else:
            consecutive_level = 2
        
        remove_all_even = st.checkbox("❌ 전부 짝수 제거", value=False)
        remove_all_odd = st.checkbox("❌ 전부 홀수 제거", value=False)
        remove_range_cluster = st.checkbox("❌ 구간 집중 제거", value=False)
        remove_high_40s = st.checkbox("❌ 40대 번호 몰림 제거", value=False)
        balance_odd_even = st.checkbox("✅ 홀짝 밸런스 (2:4~4:2)", value=True)
        exclude_recent_10 = st.checkbox("❌ 최근 10회 번호 제외", value=False)
        
        st.markdown("---")
        st.markdown("### 🎯 포함할 번호")
        include_numbers_input = st.text_input(
            "반드시 포함할 번호 (쉼표로 구분)",
            "",
            placeholder="예: 7,27"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 추천 생성 버튼
        if st.button("🎲 추천 번호 생성", use_container_width=True, type="primary"):
            # 필터 설정
            recommender.set_filters(
                remove_consecutive=remove_consecutive,
                consecutive_level=consecutive_level,
                remove_all_even=remove_all_even,
                remove_all_odd=remove_all_odd,
                remove_range_cluster=remove_range_cluster,
                remove_high_40s=remove_high_40s,
                balance_odd_even=balance_odd_even,
                exclude_recent_10=exclude_recent_10
            )
            
            # 포함 번호 처리
            include_numbers = []
            if include_numbers_input:
                try:
                    include_numbers = [int(n.strip()) for n in include_numbers_input.split(',') if n.strip()]
                except:
                    st.error("❌ 포함 번호 형식이 올바르지 않습니다.")
                    st.stop()
            
            # 번호 생성
            with st.spinner("🔄 추천 번호를 생성하는 중..."):
                try:
                    recommendations = recommender.generate_numbers(count=5, include_numbers=include_numbers)
                    st.session_state.recommendations = recommendations
                except Exception as e:
                    st.error(f"❌ 생성 실패: {e}")
    
    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 추천 번호")
        
        if st.session_state.recommendations:
            for i, numbers in enumerate(st.session_state.recommendations, 1):
                st.markdown(
                    f'<div class="number-display">[{i}] {" - ".join(map(str, numbers))}</div>',
                    unsafe_allow_html=True
                )
            
            # 필터 영향도 계산
            st.markdown("---")
            st.markdown("### 📊 필터 영향도 분석")
            
            with st.spinner("분석 중..."):
                impact = recommender.calculate_filter_impact(sample_size=10000)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("✅ 통과율", f"{impact['pass_rate']}%")
                    st.metric("❌ 제외율", f"{impact['rejection_rate']}%")
                with col_b:
                    # 가상 확률 상승 시뮬레이션
                    improvement = min(impact['rejection_rate'] * 0.1, 15)  # 최대 15%
                    st.metric("🎯 체감 확률 상승", f"+{improvement:.1f}%", delta=f"필터 효과")
                    
                    base_prob = 8145060
                    improved_prob = int(base_prob * (1 - improvement/100))
                    st.metric("개선된 1등 확률", f"1/{improved_prob:,}", delta=f"↑{improvement:.1f}%")
            
            # 활성 필터 표시
            active_filters = recommender.get_active_filters()
            if active_filters:
                st.markdown("**🔧 활성화된 필터:**")
                for filter_name in active_filters:
                    st.markdown(f"- {filter_name}")
        
        else:
            st.info("👈 왼쪽에서 필터를 설정하고 '추천 번호 생성' 버튼을 클릭하세요.")
        
        st.markdown('</div>', unsafe_allow_html=True)


def show_investment_simulation():
    """투자 시뮬레이션 탭"""
    st.markdown('<div class="sub-header">💰 투자 시뮬레이션</div>', unsafe_allow_html=True)
    
    visualizer = st.session_state.visualizer
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 💵 투자 설정")
        
        monthly_investment = st.slider(
            "월 투자 금액 (원)",
            min_value=10000,
            max_value=500000,
            value=100000,
            step=10000
        )
        
        tickets_per_week = monthly_investment // 4000  # 주당 게임 수
        
        st.metric("주당 게임 수", f"{tickets_per_week}게임")
        st.metric("월 게임 수", f"{tickets_per_week * 4}게임")
        
        st.markdown("---")
        st.markdown("### 🎯 기대 수익 설정")
        
        improvement_rate = st.slider(
            "필터 적용 개선율 (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.5
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### 📊 시뮬레이션 결과")
        
        # 기대 수익 계산 (매우 단순화된 모델)
        base_return_rate = 0.45  # 환수율 약 45%
        expected_return = int(monthly_investment * base_return_rate)
        improved_return = int(expected_return * (1 + improvement_rate/100))
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("월 투자액", f"{monthly_investment:,}원")
        with col_b:
            st.metric("기본 기대수익", f"{expected_return:,}원")
        with col_c:
            improvement_amount = improved_return - expected_return
            st.metric("개선 기대수익", f"{improved_return:,}원", delta=f"+{improvement_amount:,}원")
        
        st.markdown("---")
        
        # 그래프 생성
        fig = visualizer.plot_investment_simulation(
            monthly_investment,
            expected_return,
            improved_return
        )
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        st.markdown("### 📝 시뮬레이션 분석")
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
