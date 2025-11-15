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
from math import comb
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
if 'cart_size' not in st.session_state:
    st.session_state.cart_size = 25
if 'cart_items' not in st.session_state:
    st.session_state.cart_items = []

@st.cache_data(ttl=3600)
def run_filter_simulation(_recommender, num_samples, filters_tuple):
    """
    필터의 통과 비율을 추정하기 위해 시뮬레이션을 실행합니다.
    st.cache_data를 사용하여 동일한 필터 구성에 대한 반복 계산을 방지합니다.
    """
    # Unpack the tuple. Note that the order must match how it's created.
    odd_even_balance, exclude_recent_draws, exclude_consecutive_lengths, range_limits_items, pinned_numbers_tuple = filters_tuple
    range_limits = dict(range_limits_items)
    pinned_numbers = list(pinned_numbers_tuple)
    k = len(pinned_numbers)

    # Create a fresh recommender instance or set filters on the provided one
    # This ensures that the simulation uses the exact filters passed in
    _recommender.set_filters(
        odd_even_balance=list(odd_even_balance),
        exclude_recent_draws=exclude_recent_draws,
        exclude_consecutive_lengths=list(exclude_consecutive_lengths),
        range_limits=range_limits
    )

    pass_count = 0
    
    if k > 6:
        return 0.0

    remaining_pool = [n for n in range(1, 46) if n not in pinned_numbers]
    remaining_count = 6 - k

    # In a real scenario, generating truly random combinations is slow.
    # For a quick simulation, we can approximate by generating numbers and checking.
    # This is a simplified simulation loop.
    for _ in range(num_samples):
        if len(remaining_pool) < remaining_count:
            break # Should not happen with k <= 6
        sample = random.sample(remaining_pool, remaining_count)
        combo = sorted(pinned_numbers + sample)
        if _recommender.apply_filters(combo):
            pass_count += 1
    
    # Avoid division by zero
    if num_samples == 0:
        return 0.0
        
    return pass_count / num_samples

def load_data():
    """데이터 로드 및 초기화"""
    try:
        # 순서가 중요: Predictor가 먼저 생성되어야 Recommender에 주입할 수 있음
        fetcher = LottoFetcher()
        analyzer = LottoPatternAnalyzer()
        engineer = LottoFeatureEngineer()
        
        # Predictor들을 먼저 생성
        number_predictor = LottoNumberPredictor()
        combo_predictor = LottoComboPredictor()

        # Recommender를 생성할 때 number_predictor를 주입
        recommender = LottoRecommender(predictor=number_predictor)
        
        visualizer = LottoVisualizer()
        ml_visualizer = LottoMLVisualizer()
        
        # 모든 객체를 세션 상태에 저장
        st.session_state.fetcher = fetcher
        st.session_state.analyzer = analyzer
        st.session_state.engineer = engineer
        st.session_state.number_predictor = number_predictor
        st.session_state.combo_predictor = combo_predictor
        st.session_state.recommender = recommender
        st.session_state.visualizer = visualizer
        st.session_state.ml_visualizer = ml_visualizer
        st.session_state.data_loaded = True
        
        return True
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return False


def _run_training_and_generation_cycle(recommender, engineer, number_predictor, combo_predictor, start_draw, end_draw, n_trials, use_feedback, settings, num_to_generate, existing_combos):
    """Helper function to run a full train-predict-generate cycle."""
    
    # Train Number Predictor
    with st.spinner(f"[{start_draw}-{end_draw}회차] 번호 예측 모델 학습 중... (시도: {n_trials}회)"):
        number_predictor.train(engineer, start_draw=start_draw, end_draw=end_draw, enable_tuning=True, n_trials=n_trials)
    
    # Train Combo Predictor
    with st.spinner(f"[{start_draw}-{end_draw}회차] 조합 예측 모델 학습 중... (시도: {n_trials}회)"):
        combo_predictor.train(engineer, start_draw=start_draw, end_draw=end_draw, enable_tuning=True, n_trials=n_trials, use_feedback_training=use_feedback)

    # Predict Probabilities
    with st.spinner(f"[{start_draw}-{end_draw}회차] 번호별 출현 확률 예측 중..."):
        probabilities = number_predictor.predict_probabilities(engineer)

    # Generate Combinations
    with st.spinner(f"{num_to_generate}개 조합 생성 중..."):
        try:
            # Update settings for this specific cycle
            cycle_settings = settings.copy()
            cycle_settings['training_range'] = f"{start_draw}-{end_draw}"
            
            raw_combos = combo_predictor.predict_top_combos(
                feature_engineer=engineer,
                number_probabilities=probabilities,
                n=num_to_generate * 20, # Generate more to ensure enough valid ones after filtering
                candidate_pool='smart',
                settings=cycle_settings
            )
            
            newly_added = []
            master_seen_combos = {tuple(item['combo']) for item in existing_combos}
            
            for combo, score in raw_combos:
                if len(newly_added) >= num_to_generate:
                    break

                # [FIX] Apply the recommender's filters to the generated combination
                if not recommender.apply_filters(combo):
                    continue
                
                combo_tuple = tuple(sorted(combo))
                if combo_tuple in master_seen_combos:
                    continue

                is_valid = True
                for item in existing_combos:
                    if len(set(combo) & set(item['combo'])) > cycle_settings['max_overlap']:
                        is_valid = False
                        break
                
                if is_valid:
                    newly_added.append({
                        'combo': sorted(combo),
                        'score': score,
                        'settings': f"{start_draw}~{end_draw}회, AI조합"
                    })
                    master_seen_combos.add(combo_tuple)
            
            return newly_added

        except Exception as e:
            st.error(f"❌ 조합 생성 실패: {e}")
            st.exception(e)
            return []

def run_one_click_recommendation():
    """Runs the full one-click recommendation process."""
    st.session_state.cart_items = []
    st.session_state.cart_size = 25
    st.session_state.final_combos = None
    st.session_state['one_click'] = True # Flag to indicate one-click is running
    
    engineer = st.session_state.engineer
    number_predictor = st.session_state.number_predictor
    combo_predictor = st.session_state.combo_predictor
    recommender = st.session_state.recommender
    latest_draw = engineer.get_latest_draw_number()

    # Default settings
    default_filters = {
        "odd_even_balance": ['4:2', '3:3', '2:4'],
        "exclude_recent_draws": latest_draw,
        "exclude_consecutive_lengths": [3, 4, 5, 6],
        "range_limits": {'0': 3, '1': 3, '2': 3, '3': 3, '4': 2},
    }
    recommender.set_filters(**default_filters)
    
    base_settings = {
        "draw_no": latest_draw + 1,
        "generation_method": "AI 조합 모델 기반",
        "filters": default_filters,
        "max_overlap": 2,
    }

    # --- Group 1: Recent 300 Draws ---
    st.info("🚀 [1/2단계] 최근 300회차 데이터로 학습 및 10개 조합 생성...")
    start_draw_1 = max(1, latest_draw - 300)
    end_draw_1 = latest_draw
    
    group1_combos = _run_training_and_generation_cycle(
        recommender, engineer, number_predictor, combo_predictor,
        start_draw=start_draw_1, end_draw=end_draw_1,
        n_trials=500, use_feedback=True,
        settings=base_settings,
        num_to_generate=10,
        existing_combos=[]
    )
    st.session_state.cart_items.extend(group1_combos)
    st.success(f"✅ [1/2단계] {len(group1_combos)}개 조합 생성 완료!")
    
    # --- Group 2: All Draws ---
    st.info("🚀 [2/2단계] 전체 회차 데이터로 학습 및 15개 조합 생성...")
    start_draw_2 = 1
    end_draw_2 = latest_draw

    group2_combos = _run_training_and_generation_cycle(
        recommender, engineer, number_predictor, combo_predictor,
        start_draw=start_draw_2, end_draw=end_draw_2,
        n_trials=500, use_feedback=True,
        settings=base_settings,
        num_to_generate=15,
        existing_combos=st.session_state.cart_items # Pass existing items for overlap check
    )
    st.session_state.cart_items.extend(group2_combos)
    st.success(f"✅ [2/2단계] {len(group2_combos)}개 조합 생성 완료!")
    st.balloons()
    
    # Final rerun to update the whole UI at the end
    st.rerun()


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

    """AI 스마트 조합 탭의 모든 UI와 로직 (장바구니 기능 포함)"""

    st.markdown('<div class="sub-header">🤖 AI 스마트 조합</div>', unsafe_allow_html=True)



    recommender = st.session_state.recommender

    engineer = st.session_state.engineer

    ml_visualizer = st.session_state.ml_visualizer

    latest_draw = engineer.get_latest_draw_number()



    # Expander 1: Filter Settings (No changes)

    with st.expander("1️⃣ 단계: 룰 기반 필터 설정", expanded=True):

        st.markdown('<div class="info-box">', unsafe_allow_html=True)

        st.markdown("조합을 생성하기 위한 기본 규칙을 설정합니다. 모든 조합은 이 규칙을 따릅니다.")

        st.markdown("##### ⚖️ 기본 필터")

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            odd_even_options = ['6:0', '5:1', '4:2', '3:3', '2:4', '1:5', '0:6']

            odd_even_balance = st.multiselect("홀짝 밸런스", options=odd_even_options, default=['4:2', '3:3', '2:4'])

        with filter_col2:

            exclude_recent_draws = st.number_input("최근 당첨번호 제외", min_value=0, max_value=latest_draw, value=latest_draw, help="최근 N회차에 나온 당첨 조합과 일치하는 조합을 제외합니다.")



        st.markdown("---")

        st.markdown("##### ⛓️ 연속 번호 상세 설정")

        consecutive_options = {"2개 연속": 2, "3개 연속": 3, "4개 연속": 4, "5개 연속": 5, "6개 연속": 6}

        excluded_consecutive_labels = st.multiselect("🚫 제외할 연속 번호 길이", options=list(consecutive_options.keys()), default=['3개 연속', '4개 연속', '5개 연속', '6개 연속'], help="선택된 길이의 연속 번호가 포함된 조합을 제외합니다.")

        exclude_consecutive_lengths = [consecutive_options[label] for label in excluded_consecutive_labels]



        st.markdown("---")

        st.markdown("##### 📍 구간별 번호 개수 제한")

        st.markdown("각 번호대(앞자리 수)별로 조합에 포함될 수 있는 최대 공의 개수를 설정합니다. (기본값 6 = 제한 없음)")

        range_cols = st.columns(5)

        range_limits_inputs = {}

        range_defaults = {'0': 3, '1': 3, '2': 3, '3': 3, '4': 2}

        range_definitions = {'0': ("1-9번대", range_cols[0]), '1': ("10번대", range_cols[1]), '2': ("20번대", range_cols[2]), '3': ("30번대", range_cols[3]), '4': ("40번대", range_cols[4])}

        for key, (label, col) in range_definitions.items():

            range_limits_inputs[key] = col.number_input(label, min_value=0, max_value=6, value=range_defaults.get(key, 6))

        st.markdown('</div>', unsafe_allow_html=True)



    # Expander 2: AI Model Control (No changes)

    with st.expander("2️⃣ 단계: AI 모델 제어 및 확률 예측", expanded=True):

        st.markdown('<div class="ai-box">', unsafe_allow_html=True)

        ai_col1, ai_col2 = st.columns([1, 2])

        with ai_col1:

            st.markdown("#### 🧠 AI 모델 학습 설정")

            if 'train_end_draw' not in st.session_state: st.session_state.train_end_draw = latest_draw

            if 'train_start_draw' not in st.session_state: st.session_state.train_start_draw = st.session_state.train_end_draw - 300

            st.session_state.train_end_draw = st.number_input("학습 종료 회차", value=st.session_state.train_end_draw, min_value=100, max_value=latest_draw)

            st.session_state.train_start_draw = st.number_input("학습 시작 회차", value=st.session_state.train_start_draw, min_value=1, max_value=st.session_state.train_end_draw - 1)

            

            use_feedback = st.checkbox("🤖 예측 결과 피드백으로 학습 (오답노트)", value=True, help="과거에 예측했던 결과와 실제 당첨 결과를 비교하여 모델을 추가로 학습시킵니다. data/prediction_results.csv 파일이 필요합니다.")



            st.markdown("---")

            st.markdown("#### 🤖 AI 모델 상세 설정")

            st.info("모델은 LightGBM을 사용하며, 아래 설정을 통해 성능을 강화할 수 있습니다.")

            enable_tuning = st.checkbox("🤖 자동 하이퍼파라미터 최적화 (Optuna)", value=False, help="AI가 최적의 설정을 찾도록 합니다. 학습 시간이 몇 배 더 길어집니다.")

            n_trials = 500

            if enable_tuning:

                n_trials = st.number_input("최적화 시도 횟수 (n_trials)", min_value=10, max_value=500, value=500)

            st.markdown("---")

            if st.button("📈 AI 모델 학습 및 확률 예측", type='primary', use_container_width=True):

                try:

                    # 번호 예측 모델 학습 및 저장

                    with st.spinner("번호 예측 모델을 학습합니다..."):

                        number_predictor = st.session_state.number_predictor

                        number_predictor.train(engineer, start_draw=st.session_state.train_start_draw, end_draw=st.session_state.train_end_draw, enable_tuning=enable_tuning, n_trials=n_trials)

                        number_predictor.save_model()

                        st.session_state.number_predictor = number_predictor # 세션에 다시 저장

                    st.success("✅ 번호 예측 모델 학습 완료!")



                    # 조합 예측 모델 학습 및 저장

                    with st.spinner("조합 예측 모델을 학습합니다..."):

                        combo_predictor = st.session_state.combo_predictor

                        combo_predictor.train(

                            engineer, 

                            start_draw=st.session_state.train_start_draw, 

                            end_draw=st.session_state.train_end_draw, 

                            enable_tuning=enable_tuning, 

                            n_trials=n_trials,

                            use_feedback_training=use_feedback # 피드백 사용 여부 전달

                        )

                        combo_predictor.save_model()

                        st.session_state.combo_predictor = combo_predictor # 세션에 다시 저장

                    st.success("✅ 조합 예측 모델 학습 완료!")



                    # 확률 예측

                    with st.spinner("AI가 번호별 출현 확률을 예측합니다..."):

                        st.session_state.predicted_probabilities = number_predictor.predict_probabilities(engineer)

                    st.success("✅ 확률 예측 완료!")

                    st.rerun() # UI 갱신



                except Exception as e:

                    st.error(f"❌ 작업 실패: {e}")

                    st.exception(e)



        with ai_col2:

            st.markdown("#### 📊 AI 번호 확률 분석 결과")

            if st.session_state.predicted_probabilities:

                if 'top_k_value' not in st.session_state: st.session_state.top_k_value = 20

                top_k = st.slider("확률 순위 표시 개수", 10, 45, st.session_state.top_k_value)

                st.session_state.top_k_value = top_k

                fig = ml_visualizer.plot_number_probabilities(st.session_state.predicted_probabilities, top_k=top_k)

                st.pyplot(fig, use_container_width=True)

                plt.close(fig)

            else:

                st.info("버튼을 눌러 AI 모델 학습 및 번호 확률 예측을 시작하세요.")

        st.markdown('</div>', unsafe_allow_html=True)



    # Expander 3: Shopping Cart System

    with st.expander("3️⃣ 단계: 장바구니 및 최종 조합 생성", expanded=True):

        st.markdown('<div class="success-box">', unsafe_allow_html=True)



        # --- 1. Cart Management ---

        st.markdown("#### 🛒 장바구니 관리")

        cart_col1, cart_col2 = st.columns([3, 1])

        with cart_col1:

            st.session_state.cart_size = st.number_input("장바구니 크기 (총 게임 수)", min_value=1, max_value=100, value=st.session_state.cart_size)

        with cart_col2:

            if st.button("🗑️ 장바구니 비우기", use_container_width=True):

                st.session_state.cart_items = []

                st.session_state.final_combos = None

                st.rerun()

        

        cart_len = len(st.session_state.cart_items)

        cart_size = st.session_state.cart_size

        cart_progress = min(cart_len / cart_size, 1.0) if cart_size > 0 else 0.0

        st.progress(cart_progress)

        st.info(f"장바구니 현황: {cart_len} / {cart_size} 개")



        if st.session_state.cart_items:

            with st.expander("📋 장바구니 내용 보기"):

                for i, item in enumerate(st.session_state.cart_items):

                    st.text(f"  {i+1:02d}. {item['combo']} (설정: {item['settings']})")

        st.markdown("---")

        

        # --- 1.5 One-Click Recommendation ---

        st.markdown("#### ✨ 원클릭 추천")

        if st.button("⚡️ 나의 원클릭 추천 담기 (25개)", use_container_width=True, type="primary", help="최근 300회차(10개), 전체회차(15개)를 각각 최대치로 학습하여 장바구니에 담습니다. 시간이 매우 오래 소요될 수 있습니다."):

            run_one_click_recommendation()

        st.markdown("--- ")





        # --- 2. Batch Configuration & Add to Cart ---

        st.markdown("#### ⚙️ 조합 생성 후 장바구니에 담기")

        add_col1, add_col2 = st.columns([3, 1])

        with add_col1:

            generation_method = st.selectbox("🛠️ 조합 생성 방식", options=["AI 조합 모델 기반", "AI 확률 예측 기반", "완전 랜덤 생성"])

            max_overlap = st.slider("조합 간 최대 중복 번호", 0, 5, 2, help="장바구니에 있는 기존 조합들과의 최대 중복을 설정합니다.")

        with add_col2:

            n_to_add = st.number_input("장바구니에 담을 개수", min_value=1, max_value=cart_size - cart_len if cart_size > cart_len else 1)



        if st.button("🛒 장바구니에 담기", use_container_width=True):

            if cart_len >= cart_size:

                st.warning("장바구니가 이미 가득 찼습니다.")

            else:

                recommender.set_filters(odd_even_balance=odd_even_balance, exclude_recent_draws=exclude_recent_draws, exclude_consecutive_lengths=exclude_consecutive_lengths, range_limits=range_limits_inputs)

                newly_added = []

                

                # 현재 UI 설정값들을 딕셔너리로 집계

                predict_draw = engineer.get_latest_draw_number() + 1

                current_settings = {

                    "draw_no": predict_draw,

                    "generation_method": generation_method,

                    "training_range": f"{st.session_state.train_start_draw}-{st.session_state.train_end_draw}",

                    "filters": {

                        "odd_even_balance": odd_even_balance,

                        "exclude_recent_draws": exclude_recent_draws,

                        "exclude_consecutive_lengths": exclude_consecutive_lengths,

                        "range_limits": range_limits_inputs,

                    },

                    "max_overlap": max_overlap,

                }

                settings_str = f"{st.session_state.train_start_draw}~{st.session_state.train_end_draw}회, {generation_method[:5]}.."

                

                with st.spinner(f"{n_to_add}개 조합을 생성하여 장바구니에 담는 중..."):

                    if generation_method == "AI 조합 모델 기반":

                        if not st.session_state.combo_predictor or not st.session_state.combo_predictor.model:

                            st.error("AI 조합 모델이 학습되지 않았습니다. 2단계에서 'AI 모델 학습'을 먼저 실행해주세요.")

                            st.stop()

                    

                    if generation_method in ["AI 조합 모델 기반", "AI 확률 예측 기반"] and not st.session_state.predicted_probabilities:

                        st.error("AI 번호 확률이 예측되지 않았습니다. 2단계에서 'AI 모델 학습'을 먼저 실행해주세요.")

                        st.stop()



                    attempts = 0

                    max_attempts = 20

                    master_seen_combos = {tuple(item['combo']) for item in st.session_state.cart_items}



                    while len(newly_added) < n_to_add and attempts < max_attempts:

                        attempts += 1

                        needed = n_to_add - len(newly_added)

                        candidate_batch_size = max(needed * 5, 20)



                        candidate_combos = []

                        if generation_method == "AI 조합 모델 기반":

                            try:

                                raw_combos = st.session_state.combo_predictor.predict_top_combos(

                                    feature_engineer=engineer,

                                    number_probabilities=st.session_state.predicted_probabilities,

                                    n=candidate_batch_size * attempts,

                                    candidate_pool='smart',

                                    settings=current_settings # 현재 설정을 모델에 전달

                                )

                                candidate_combos = [(c, s) for c, s in raw_combos]

                            except Exception as e:

                                st.error(f"❌ 조합 모델 기반 생성 실패: {e}")

                                st.exception(e)

                                st.stop()

                        

                        elif generation_method == "AI 확률 예측 기반":

                            probs = st.session_state.predicted_probabilities

                            numbers = list(probs.keys())

                            p_values = np.array(list(probs.values()))

                            p_values /= p_values.sum()

                            gen_combos = [sorted(np.random.choice(numbers, 6, replace=False, p=p_values).tolist()) for _ in range(candidate_batch_size)]

                            candidate_combos = [(c, 0.0) for c in gen_combos]

                        

                        else: # 완전 랜덤 생성

                            gen_combos = [sorted(random.sample(range(1, 46), 6)) for _ in range(candidate_batch_size)]

                            candidate_combos = [(c, 0.0) for c in gen_combos]



                        cart_number_counts = {n: 0 for n in range(1, 46)}

                        current_combos_in_cart = st.session_state.cart_items + newly_added

                        for item in current_combos_in_cart:

                            for number in item['combo']:

                                cart_number_counts[number] += 1



                        scored_candidates = []

                        for combo, score in candidate_combos:

                            combo_tuple = tuple(sorted(combo))

                            if combo_tuple in master_seen_combos:

                                continue

                            diversity_score = sum(cart_number_counts[n] for n in combo)

                            scored_candidates.append({'combo': combo, 'ai_score': score, 'diversity_score': diversity_score})

                        

                        if scored_candidates:

                            max_div_score = max(c['diversity_score'] for c in scored_candidates) if scored_candidates else 0

                            min_div_score = min(c['diversity_score'] for c in scored_candidates) if scored_candidates else 0



                            for c in scored_candidates:

                                if max_div_score == min_div_score:

                                    norm_div_score = 1.0

                                else:

                                    norm_div_score = 1 - ((c['diversity_score'] - min_div_score) / (max_div_score - min_div_score))

                                

                                # ai_score는 이제 0-6 사이의 '예상 맞춘 개수'

                                norm_ai_score = c['ai_score'] / 6.0

                                c['final_score'] = (0.5 * norm_ai_score) + (0.5 * norm_div_score)



                            scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)



                        for candidate in scored_candidates:

                            if len(newly_added) >= n_to_add:

                                break

                            

                            combo = candidate['combo']

                            combo_tuple = tuple(sorted(combo))

                            

                            if combo_tuple in master_seen_combos:

                                continue



                            is_valid = True

                            for item in current_combos_in_cart:

                                if len(set(combo) & set(item['combo'])) > max_overlap:

                                    is_valid = False

                                    break

                            

                            if is_valid and recommender.apply_filters(combo):

                                newly_added.append({

                                    'combo': sorted(combo),

                                    'score': candidate['ai_score'],

                                    'settings': settings_str

                                })

                                master_seen_combos.add(combo_tuple)

                    

                    if len(newly_added) < n_to_add:

                        st.warning(f"필터 조건이 매우 엄격하여 요청하신 {n_to_add}개 중 {len(newly_added)}개만 생성할 수 있었습니다. 필터를 완화하거나 재시도해 보세요.")



                

                st.session_state.cart_items.extend(newly_added)

                st.success(f"{len(newly_added)}개의 조합을 장바구니에 담았습니다!")

                st.rerun()



        st.markdown("--- ")



        # --- 3. Finalization ---

        st.markdown("#### 🚀 최종 조합 생성 및 저장")

        finalize_col1, finalize_col2 = st.columns([1, 2])

        with finalize_col1:

            if st.button("🚀 장바구니 조합으로 최종 생성", use_container_width=True):

                if not st.session_state.cart_items:

                    st.warning("장바구니가 비어있습니다.")

                else:

                    final_combos = {None: [(item['combo'], item['score']) for item in st.session_state.cart_items]}

                    st.session_state.final_combos = final_combos

                    st.session_state.prediction_active_filters = recommender.get_active_filters()

                    st.success("장바구니의 모든 조합으로 최종 결과를 생성했습니다!")

        

        with finalize_col2:

            st.markdown("#### 🎯 최종 추천 조합")

            final_combos = st.session_state.get('final_combos', None)

            if final_combos:

                if not any(final_combos.values()):

                    st.warning("⚠️ 선택된 필터 조건을 만족하는 조합을 찾지 못했습니다. 필터를 완화해 보세요.")

                else:

                    for target, combos in final_combos.items():

                        if not combos: continue

                        for i, (combo, score) in enumerate(combos, 1):

                            score_text = f"- 예상 {score:.2f}개"

                            st.markdown(f'<div class="number-display">#{i} [{', '.join(map(str, combo))}] {score_text}</div>', unsafe_allow_html=True)

                    st.markdown("--- ")

                    active_filters = st.session_state.get('prediction_active_filters', [])

                    if active_filters: 

                        st.markdown("**🔧 적용된 필터:**")

                        for filter_name in active_filters: st.markdown(f"- {filter_name}")

                    st.markdown("--- ")

                    if st.button("💾 추천 조합 저장", use_container_width=True):

                        import json

                        

                        # 저장할 데이터 구조 생성

                        combos_to_save = [c for _, combos in final_combos.items() for c, s in combos]

                        combos_to_save = [[int(n) for n in c] for c in combos_to_save]



                        if not combos_to_save:

                            st.warning("저장할 조합이 없습니다.")

                        else:

                            predict_draw = engineer.get_latest_draw_number() + 1

                            num_games = len(combos_to_save)

                            

                            # 저장할 설정값들 집계

                            settings_data = {

                                "draw_no": predict_draw,

                                "generation_method": "원클릭 추천" if 'one_click' in st.session_state else generation_method,

                                "training_range": f"{st.session_state.train_start_draw}-{st.session_state.train_end_draw}",

                                "filters": {

                                    "odd_even_balance": odd_even_balance,

                                    "exclude_recent_draws": exclude_recent_draws,

                                    "exclude_consecutive_lengths": exclude_consecutive_lengths,

                                    "range_limits": range_limits_inputs,

                                },

                                "max_overlap": max_overlap,

                            }

                            if 'one_click' in st.session_state:

                                del st.session_state['one_click']

                            

                            # 최종 저장 객체

                            save_object = {

                                "settings": settings_data,

                                "combinations": combos_to_save

                            }



                            save_dir = Path('data/predictions')

                            save_dir.mkdir(parents=True, exist_ok=True)

                            filename = f"{predict_draw}_{num_games}.json"

                            save_path = save_dir / filename

                            

                            try:

                                with open(save_path, 'w', encoding='utf-8') as f:

                                    json.dump(save_object, f, indent=4, ensure_ascii=False)

                                st.success(f"✅ 조합과 설정을 성공적으로 저장했습니다: `{save_path}`")

                            except Exception as e:

                                st.error(f"❌ 파일 저장에 실패했습니다: {e}")

            else:

                st.info("장바구니에 조합을 담고 '최종 생성' 버튼을 누르세요.")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
