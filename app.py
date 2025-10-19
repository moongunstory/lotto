"""
로또 VIP 분석 프로그램 - Streamlit GUI
데이터 수집, 패턴 분석, 번호 추천, 투자 시뮬레이션 통합
"""

# ==================== CRITICAL FIX: PIL 이미지 크기 제한 해제 ====================
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 이미지 크기 제한 완전 해제

import warnings
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

import os
os.environ['MPLBACKEND'] = 'Agg'
# ================================================================================

import streamlit as st
import sys
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetch.lotto_fetcher import LottoFetcher
from analysis.lotto_pattern_analyzer import LottoPatternAnalyzer
from analysis.lotto_recommender import LottoRecommender
from analysis.lotto_visualizer import LottoVisualizer
from analysis.lotto_feature_engineer import LottoFeatureEngineer
from analysis.lotto_number_predictor import LottoNumberPredictor
from analysis.lotto_combo_predictor import LottoComboPredictor
from analysis.lotto_ml_visualizer import LottoMLVisualizer

# 페이지 설정
st.set_page_config(
    page_title="🎯 로또 VIP AI 분석 프로그램",
    page_icon="🤖",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .ai-box {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
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
if 'ml_models_loaded' not in st.session_state:
    st.session_state.ml_models_loaded = False
if 'number_predictor' not in st.session_state:
    st.session_state.number_predictor = None
if 'combo_predictor' not in st.session_state:
    st.session_state.combo_predictor = None
if 'predicted_probabilities' not in st.session_state:
    st.session_state.predicted_probabilities = None
if 'final_combos' not in st.session_state:
    st.session_state.final_combos = None
if 'prediction_active_filters' not in st.session_state:
    st.session_state.prediction_active_filters = []

def load_data():
    """데이터 로드 및 초기화"""
    try:
        fetcher = LottoFetcher()
        analyzer = LottoPatternAnalyzer()
        recommender = LottoRecommender()
        visualizer = LottoVisualizer()
        
        engineer = LottoFeatureEngineer()
        ml_visualizer = LottoMLVisualizer()
        
        st.session_state.fetcher = fetcher
        st.session_state.analyzer = analyzer
        st.session_state.recommender = recommender
        st.session_state.visualizer = visualizer
        st.session_state.engineer = engineer
        st.session_state.ml_visualizer = ml_visualizer
        st.session_state.data_loaded = True
        
        return True
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return False


def main():
    """메인 함수"""
    st.markdown('<div class="main-header">🎯 로또 VIP AI 분석 프로그램</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/FFFFFF?text=LOTTO+AI", use_container_width=True)
        st.markdown("### 📊 메뉴")
        tab_selection = st.radio(
            "기능 선택",
            ["🏠 홈", "📥 데이터 수집", "📊 패턴 분석", "🤖 AI 스마트 조합"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("### ℹ️ 정보")
        st.info("로또 6/45 당첨 확률:\n- 1등: 1/8,145,060\n- 2등: 1/1,357,510\n- 3등: 1/35,724")
    
    if not st.session_state.data_loaded:
        with st.spinner("🔄 데이터를 불러오는 중..."):
            if not load_data():
                st.stop()
    
    if tab_selection == "🏠 홈":
        show_home()
    elif tab_selection == "📥 데이터 수집":
        show_data_collection()
    elif tab_selection == "📊 패턴 분석":
        show_pattern_analysis()
    elif tab_selection == "🤖 AI 스마트 조합":
        show_ai_smart_combo_tab()

def show_home():
    """홈 화면"""
    st.markdown('<div class="sub-header">🏠 환영합니다!</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
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
        st.markdown('<div class="ai-box">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI 예측")
        st.markdown("머신러닝으로 번호 확률과 최적 조합을 예측합니다.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🎲 번호 추천")
        st.markdown("사용자 정의 필터를 적용하여 최적의 번호를 추천합니다.")
        st.markdown('</div>', unsafe_allow_html=True)
    
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

            st.markdown("### 📚 당첨 기록 살펴보기")
            recent_tab, all_tab = st.tabs(["최근 10회", "전체 기록"])

            columns = {
                "draw_no": st.column_config.NumberColumn("회차", format="%d"),
                "date": "추첨일", "n1": "1번", "n2": "2번", "n3": "3번",
                "n4": "4번", "n5": "5번", "n6": "6번", "bonus": "보너스"
            }

            with recent_tab:
                st.dataframe(
                    df[list(columns.keys())].tail(10),
                    use_container_width=True, hide_index=True, column_config=columns, height=360
                )

            with all_tab:
                st.dataframe(
                    df[list(columns.keys())],
                    use_container_width=True, hide_index=True, column_config=columns, height=480
                )

        except Exception as e:
            st.warning("⚠️ 데이터가 없습니다. 수집을 시작해주세요.")
    
    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚙️ 작업")
        
        if st.button("🔄 데이터 갱신", use_container_width=True, type="primary"):
            with st.spinner("데이터를 갱신하는 중..."):
                try:
                    st.session_state.fetcher.update_data()
                    st.session_state.analyzer.load_data()
                    if 'engineer' in st.session_state: st.session_state.engineer.load_data()
                    st.success("✅ 갱신 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 갱신 실패: {e}")
        
        if st.button("📥 전체 재수집", use_container_width=True):
            with st.spinner("전체 데이터를 수집하는 중..."):
                try:
                    st.session_state.fetcher.fetch_all_data()
                    st.session_state.analyzer.load_data()
                    st.success("✅ 수집 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 수집 실패: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_pattern_analysis():
    """패턴 분석 탭"""
    st.markdown('<div class="sub-header">📊 패턴 분석</div>', unsafe_allow_html=True)
    
    analyzer = st.session_state.analyzer
    visualizer = st.session_state.visualizer
    
    highlight_input = st.text_input("🎯 강조할 번호 입력 (쉼표로 구분)", "")
    highlight_numbers = [int(n.strip()) for n in highlight_input.split(',') if n.strip()] if highlight_input else []
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 출현 빈도", "📈 최근 추세", "🎯 홀짝 분석", "📍 구간 분포"])
    
    with tab1:
        st.markdown("### 📊 전체 회차 번호별 출현 빈도")
        freq_df = analyzer.get_number_frequency()
        fig = visualizer.plot_number_frequency(freq_df, highlight_numbers)
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        st.markdown("### 📈 최근 30회차 출현 추세")
        fig = visualizer.plot_recent_trend(analyzer, 30, highlight_numbers)
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.markdown("### 🎯 홀짝 비율 분석")
        odd_even_data = analyzer.get_odd_even_ratio()
        fig = visualizer.plot_odd_even_distribution(odd_even_data)
        st.pyplot(fig)
        plt.close()
    
    with tab4:
        st.markdown("### 📍 번호 구간별 분포")
        range_data = analyzer.get_range_distribution()
        fig = visualizer.plot_range_distribution(range_data)
        st.pyplot(fig)
        plt.close()

def show_ai_smart_combo_tab():
    """AI 스마트 조합 탭의 모든 UI와 로직"""
    st.markdown('<div class="sub-header">🤖 AI 스마트 조합</div>', unsafe_allow_html=True)

    recommender = st.session_state.recommender
    engineer = st.session_state.engineer
    ml_visualizer = st.session_state.ml_visualizer

    with st.expander("1️⃣ 단계: 룰 기반 필터 설정", expanded=True):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("조합을 생성하기 위한 기본 규칙을 설정합니다. 모든 조합은 이 규칙을 따릅니다.")
        
        # --- 기본 필터 ---
        st.markdown("##### ⚖️ 기본 필터")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            odd_even_options = ['6:0', '5:1', '4:2', '3:3', '2:4', '1:5', '0:6']
            odd_even_balance = st.multiselect("홀짝 밸런스", options=odd_even_options, default=['4:2', '3:3', '2:4'])
        with filter_col2:
            exclude_recent_draws = st.number_input("최근 당첨번호 제외", min_value=0, max_value=1000, value=10, help="최근 N회차에 나온 당첨 조합과 일치하는 조합을 제외합니다.")

        st.markdown("---")

        # --- 연속 번호 필터 (신규) ---
        st.markdown("##### ⛓️ 연속 번호 상세 설정")
        consecutive_options = {
            "2개 연속": 2, "3개 연속": 3, "4개 연속": 4, "5개 연속": 5, "6개 연속": 6
        }
        excluded_consecutive_labels = st.multiselect(
            "🚫 제외할 연속 번호 길이",
            options=list(consecutive_options.keys()),
            help="선택된 길이의 연속 번호가 포함된 조합을 제외합니다. (예: '3개 연속' 선택 시, [1,2,3] 포함 조합 제외)"
        )
        exclude_consecutive_lengths = [consecutive_options[label] for label in excluded_consecutive_labels]

        st.markdown("---")

        # --- 구간 집중 필터 (신규) ---
        st.markdown("##### 📍 구간별 번호 개수 제한")
        st.markdown("각 번호대(앞자리 수)별로 조합에 포함될 수 있는 최대 공의 개수를 설정합니다. (기본값 6 = 제한 없음)")
        range_cols = st.columns(5)
        range_limits_inputs = {}
        range_definitions = {
            '0': ("1-9번대", range_cols[0]),
            '1': ("10번대", range_cols[1]),
            '2': ("20번대", range_cols[2]),
            '3': ("30번대", range_cols[3]),
            '4': ("40번대", range_cols[4]),
        }
        for key, (label, col) in range_definitions.items():
            range_limits_inputs[key] = col.number_input(label, min_value=0, max_value=6, value=6)

        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("2️⃣ 단계: AI 모델 제어 및 확률 예측"):
        st.markdown('<div class="ai-box">', unsafe_allow_html=True)
        ai_col1, ai_col2 = st.columns([1, 2])
        
        with ai_col1:
            st.markdown("#### 🧠 AI 모델 제어")
            latest_draw = engineer.get_latest_draw_number()
            
            # 세션 상태 초기화
            if 'train_end_draw' not in st.session_state:
                st.session_state.train_end_draw = latest_draw - 20
            if 'train_start_draw' not in st.session_state:
                st.session_state.train_start_draw = st.session_state.train_end_draw - 300
            
            train_end_draw = st.number_input(
                "학습 종료 회차", 
                value=st.session_state.train_end_draw, 
                min_value=100, 
                max_value=latest_draw,
                key='input_train_end'
            )
            train_start_draw = st.number_input(
                "학습 시작 회차", 
                value=st.session_state.train_start_draw, 
                min_value=1, 
                max_value=train_end_draw - 1,
                key='input_train_start'
            )
            
            # 세션 상태 업데이트
            st.session_state.train_end_draw = train_end_draw
            st.session_state.train_start_draw = train_start_draw

            st.markdown("---")
            st.info("설정된 회차로 학습된 모델이 있으면 불러오고, 없으면 새로 학습합니다.")
            
            if st.button("📈 AI 번호 확률 예측", type='primary', use_container_width=True):
                model_path = f'models/number_predictor_{train_start_draw}_{train_end_draw}.pkl'

                try:
                    number_predictor = LottoNumberPredictor(model_type='xgboost')
                    expected_version = engineer.get_feature_version()
                    model_exists = Path(model_path).exists()
                    need_training = not model_exists

                    if model_exists:
                        try:
                            with st.spinner(f"기존 학습 모델({train_start_draw}~{train_end_draw}회)을 불러옵니다..."):
                                number_predictor.load_model(model_path, expected_feature_version=expected_version)
                            st.success("✅ 기존 모델 로드 완료!")
                        except ValueError as load_err:
                            st.warning(f"⚠️ {load_err}")
                            need_training = True

                    if need_training:
                        training_label = "새로운 모델" if not model_exists else "모델을 다시 학습"
                        with st.spinner(f"{training_label}({train_start_draw}~{train_end_draw}회)을 학습합니다... (시간 소요)"):
                            number_predictor.train(engineer, start_draw=train_start_draw, end_draw=train_end_draw)
                            number_predictor.save_model(model_path)
                        st.success("✅ 모델 학습 및 저장 완료!")

                    st.session_state.number_predictor = number_predictor

                    try:
                        with st.spinner("AI가 번호별 출현 확률을 예측합니다..."):
                            st.session_state.predicted_probabilities = number_predictor.predict_probabilities(engineer)
                        st.success("✅ 확률 예측 완료!")
                    except ValueError as feature_err:
                        st.warning(f"⚠️ {feature_err} 최신 피처에 맞춰 모델을 다시 학습합니다.")
                        with st.spinner("AI 모델을 최신 피처로 재학습합니다... (시간 소요)"):
                            number_predictor.train(engineer, start_draw=train_start_draw, end_draw=train_end_draw)
                            number_predictor.save_model(model_path)
                        st.session_state.number_predictor = number_predictor
                        with st.spinner("재학습된 모델로 확률을 다시 계산합니다..."):
                            st.session_state.predicted_probabilities = number_predictor.predict_probabilities(engineer)
                        st.success("✅ 모델 재학습 및 확률 예측 완료!")

                except Exception as e:
                    st.error(f"❌ 작업 실패: {e}")

        with ai_col2:
            st.markdown("#### 📊 AI 번호 확률 분석 결과")
            if st.session_state.predicted_probabilities:
                # top_k를 세션 상태로 관리
                if 'top_k_value' not in st.session_state:
                    st.session_state.top_k_value = 20
                
                top_k = st.slider(
                    "확률 순위 표시 개수", 
                    10, 45, 
                    st.session_state.top_k_value,
                    key='slider_top_k'
                )
                st.session_state.top_k_value = top_k
                
                # 그래프 생성
                fig = ml_visualizer.plot_number_probabilities(
                    st.session_state.predicted_probabilities, 
                    top_k=top_k
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("버튼을 눌러 번호 확률 예측을 시작하세요.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("3️⃣ 단계: AI 기반 조합 생성", expanded=True):
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        gen_col1, gen_col2 = st.columns([1, 2])
        
        with gen_col1:
            st.markdown("#### ⚙️ 조합 생성 설정")
            pinned_numbers = []
            if st.session_state.predicted_probabilities:
                st.markdown("**🎯 AI 추천 번호 고정:**")
                top_numbers = sorted(st.session_state.predicted_probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
                for num, prob in top_numbers:
                    if st.checkbox(f"{num}번 ({prob*100:.2f}%)", key=f"pin_{num}"):
                        pinned_numbers.append(num)
            
            manual_include = st.text_input("수동으로 번호 고정 (쉼표로 구분)")
            if manual_include: 
                try:
                    pinned_numbers.extend([int(n.strip()) for n in manual_include.split(',') if n.strip()])
                except ValueError:
                    st.error("숫자만 입력해주세요.")
            pinned_numbers = sorted(list(set(pinned_numbers)))

            generation_method = st.selectbox("🛠️ 조합 생성 방식", options=["AI 조합 모델 기반", "AI 확률 예측 기반", "완전 랜덤 생성"])
            n_combos = st.slider("생성할 조합 개수", 1, 20, 5)

            if st.button("🚀 조합 생성 실행", type='primary', use_container_width=True):
                # 세션 상태에서 회차 정보 가져오기
                train_start_draw = st.session_state.get('train_start_draw', engineer.get_latest_draw_number() - 320)
                train_end_draw = st.session_state.get('train_end_draw', engineer.get_latest_draw_number() - 20)
                
                recommender.set_filters(
                    odd_even_balance=odd_even_balance,
                    exclude_recent_draws=exclude_recent_draws,
                    exclude_consecutive_lengths=exclude_consecutive_lengths,
                    range_limits=range_limits_inputs
                )

                final_combos = {}
                with st.spinner("필터에 맞는 조합을 생성 중입니다..."):
                    targets = pinned_numbers if pinned_numbers else [None]
                    for target_num in targets:
                        include_list = [target_num] if target_num else []
                        
                        combos_for_target = []
                        seen_combos = set()

                        if generation_method == "AI 확률 예측 기반":
                            if not st.session_state.predicted_probabilities:
                                st.error("2단계에서 AI 번호 확률을 먼저 예측해주세요.")
                                st.stop()
                            
                            top_n_pool = 25
                            candidate_pool = [num for num, _ in sorted(st.session_state.predicted_probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n_pool]]
                            candidate_pool = [n for n in candidate_pool if n not in include_list]

                            attempts = 0
                            while len(combos_for_target) < n_combos and attempts < 20000:
                                attempts += 1
                                remaining_count = 6 - len(include_list)
                                if len(candidate_pool) < remaining_count: break
                                
                                sample = random.sample(candidate_pool, remaining_count)
                                combo = sorted(include_list + sample)
                                combo_tuple = tuple(combo)
                                if combo_tuple in seen_combos: continue

                                if recommender.apply_filters(combo, include_list):
                                    combos_for_target.append((combo, 0.0))
                                    seen_combos.add(combo_tuple)

                        elif generation_method == "AI 조합 모델 기반":
                            model_path = f'models/combo_predictor_{train_start_draw}_{train_end_draw}.pkl'
                            try:
                                combo_predictor = LottoComboPredictor(model_type='xgboost')
                                expected_version = engineer.get_feature_version()
                                model_exists = Path(model_path).exists()
                                need_training = not model_exists

                                if model_exists:
                                    try:
                                        with st.spinner(f"기존 조합 모델({train_start_draw}~{train_end_draw}회)을 불러옵니다..."):
                                            combo_predictor.load_model(model_path, expected_feature_version=expected_version)
                                        st.success("✅ 기존 조합 모델 로드 완료!")
                                    except ValueError as load_err:
                                        st.warning(f"⚠️ {load_err}")
                                        need_training = True

                                if need_training:
                                    training_label = "새로운 조합 모델" if not model_exists else "조합 모델을 다시 학습"
                                    with st.spinner(f"{training_label}({train_start_draw}~{train_end_draw}회)을 학습합니다... (시간 소요)"):
                                        combo_predictor.train(engineer, start_draw=train_start_draw, end_draw=train_end_draw)
                                        combo_predictor.save_model(model_path)
                                    st.success("✅ 조합 모델 학습 및 저장 완료!")

                                st.session_state.combo_predictor = combo_predictor

                                # 만약 고정 수가 있다면, 조합 생성 방식을 변경합니다.
                                if include_list:
                                    st.warning("고정 번호가 있어, 조합 생성 및 평가에 시간이 더 소요될 수 있습니다.")
                                    pool = [n for n in range(1, 46) if n not in include_list]
                                    remaining_count = 6 - len(include_list)

                                    if len(pool) < remaining_count:
                                        st.error("고정 번호가 너무 많아 조합을 생성할 수 없습니다.")
                                        st.stop()

                                    combos_to_check_iterator = combinations(pool, remaining_count)
                                    temp_combos = list(combos_to_check_iterator)
                                    sample_size = min(len(temp_combos), 20000)
                                    
                                    sampled_indices = np.random.choice(len(temp_combos), size=sample_size, replace=False)
                                    
                                    valid_candidates = []
                                    with st.spinner(f"{sample_size}개 후보 조합을 필터링합니다..."):
                                        for idx in sampled_indices:
                                            base_combo = temp_combos[idx]
                                            final_combo = sorted(list(base_combo) + include_list)
                                            if recommender.apply_filters(final_combo, include_list):
                                                valid_candidates.append(final_combo)

                                    scored_combos = []
                                    with st.spinner(f"{len(valid_candidates)}개 유효 조합의 점수를 계산합니다..."):
                                        for combo in valid_candidates:
                                            score = combo_predictor.score_combination(engineer, combo, reference_draw=train_end_draw + 1)
                                            scored_combos.append((combo, score))
                                    
                                    scored_combos.sort(key=lambda x: x[1], reverse=True)
                                    combos_for_target = scored_combos[:n_combos]

                                else: # 고정 수가 없을 때의 원래 로직
                                    with st.spinner("학습된 모델로 최적 조합을 예측합니다..."):
                                        raw_combos = combo_predictor.predict_top_combos(engineer, n=n_combos * 20, candidate_pool='smart', pool_size=30)
                                    
                                    for combo, score in raw_combos:
                                        combo_tuple = tuple(sorted(combo))
                                        if combo_tuple in seen_combos: continue
                                        if recommender.apply_filters(combo, include_list):
                                            combos_for_target.append((list(combo_tuple), score))
                                            seen_combos.add(combo_tuple)
                                        if len(combos_for_target) >= n_combos: break

                            except Exception as e:
                                st.error(f"❌ 조합 생성 실패: {e}")
                                st.stop()

                        elif generation_method == "완전 랜덤 생성":
                            generated = recommender.generate_numbers(count=n_combos, include_numbers=include_list)
                            combos_for_target = [(c, 0.0) for c in generated]
                        
                        final_combos[target_num] = combos_for_target

                st.session_state.final_combos = final_combos
                st.session_state.prediction_active_filters = recommender.get_active_filters()

        with gen_col2:
            st.markdown("#### 🎯 최종 추천 조합")
            final_combos = st.session_state.get('final_combos', None)

            if final_combos:
                if not any(final_combos.values()):
                    st.warning("⚠️ 선택된 필터 조건을 만족하는 조합을 찾지 못했습니다. 필터를 완화해 보세요.")
                
                for target, combos in final_combos.items():
                    if target:
                        st.markdown(f"##### 📌 **{target}번**을 포함하는 추천 조합")
                    
                    if not combos:
                        st.warning("조건을 만족하는 조합이 없습니다.")
                        continue

                    for i, (combo, score) in enumerate(combos, 1):
                        score_text = f"- 신뢰도 {score*100:.2f}%" if score > 0 else ""
                        st.markdown(
                            f'<div class="number-display">#{i} [{', '.join(map(str, combo))}] {score_text}</div>',
                            unsafe_allow_html=True
                        )
                st.markdown("---")
                active_filters = st.session_state.get('prediction_active_filters', [])  
                if active_filters:
                    st.markdown("**🔧 적용된 필터:**")
                    for filter_name in active_filters:
                        st.markdown(f"- {filter_name}")
            else:
                st.info("왼쪽에서 설정을 완료하고 '조합 생성 실행' 버튼을 누르세요.")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
