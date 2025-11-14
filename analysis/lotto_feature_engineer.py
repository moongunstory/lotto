"""
로또 6/45 피처 엔지니어링 모듈 (고속 벡터화 버전)
ML 학습을 위한 피처 생성 및 데이터셋 구축
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json

class LottoFeatureEngineer:
    """로또 피처 엔지니어링 클래스 (벡터화 최적화)"""

    # 피처 구성이 변경될 때마다 버전을 갱신한다.
    FEATURE_VERSION = "2025.11"  # 예측 피드백 피처 추가
    
    def __init__(self, data_path='data/lotto_history.csv'):
        self.data_path = Path(data_path)
        self.df = None
        self.features_df = None # 피처 캐시
        self.load_data()

    def get_feature_version(self):
        """현재 피처 엔지니어링 스키마 버전을 반환"""
        return self.FEATURE_VERSION
        
    def load_data(self):
        """데이터 로드"""
        if self.data_path.exists():
            self.df = pd.read_csv(self.data_path, index_col='draw_no')
            print(f"✅ 데이터 로드 완료: {len(self.df)}회차")
        else:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")

    def _create_feature_grid(self):
        """모든 번호와 모든 회차에 대한 그리드 생성"""
        print("📊 피처 그리드 생성 중...")
        # 1. 모든 회차, 모든 번호에 대한 기본 그리드 생성
        draws = np.arange(1, self.df.index.max() + 2)
        numbers = np.arange(1, 46)
        grid = pd.DataFrame(np.array(np.meshgrid(draws, numbers)).T.reshape(-1, 2), columns=['draw_no', 'number'])
        grid.set_index(['draw_no', 'number'], inplace=True)

        # 2. 실제 당첨 번호 데이터 "long" 포맷으로 변경
        winning_numbers_long = self.df.reset_index().melt(
            id_vars='draw_no',
            value_vars=[f'n{i}' for i in range(1, 7)],
            value_name='number'
        )
        winning_numbers_long['appeared'] = 1
        winning_numbers_long = winning_numbers_long.drop(columns='variable')
        winning_numbers_long = winning_numbers_long.astype(int).set_index(['draw_no', 'number'])

        # 3. 그리드에 당첨 여부(appeared) 병합
        grid = grid.join(winning_numbers_long, how='left')
        grid['appeared'] = grid['appeared'].fillna(0).astype(int)
        return grid

    def _calculate_pair_features(self, original_df):
        """번호 조합(궁합) 피처를 생성합니다."""
        print("   - (추가) 궁합 피처 계산 중...")
        from itertools import combinations
        
        # 모든 회차의 당첨번호 조합을 리스트로 만듦
        draws = original_df[[f'n{i}' for i in range(1, 7)]].values.tolist()

        # 모든 가능한 2개 번호 조합의 출현 횟수를 계산
        pair_counts = {}
        for draw in draws:
            for pair in combinations(sorted(draw), 2):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        # 데이터프레임으로 변환
        pair_df = pd.DataFrame(list(pair_counts.items()), columns=['pair', 'count'])
        pair_df[['num1', 'num2']] = pd.DataFrame(pair_df['pair'].tolist(), index=pair_df.index)

        # 각 번호별로 가장 궁합이 좋은 번호와 그 횟수를 찾음
        best_partners = {}
        all_numbers = range(1, 46)
        for num in all_numbers:
            # num이 포함된 모든 조합을 찾음
            related_pairs = pair_df[(pair_df['num1'] == num) | (pair_df['num2'] == num)]
            if not related_pairs.empty:
                # 가장 많이 나온 조합을 찾음
                best_pair_row = related_pairs.loc[related_pairs['count'].idxmax()]
                # 상대방 번호를 찾음
                partner = best_pair_row['num2'] if best_pair_row['num1'] == num else best_pair_row['num1']
                count = best_pair_row['count']
                best_partners[num] = {'best_partner': partner, 'best_partner_count': count}
            else:
                best_partners[num] = {'best_partner': 0, 'best_partner_count': 0} # 데이터 없는 경우

        # 최종 데이터프레임 생성
        partner_df = pd.DataFrame.from_dict(best_partners, orient='index')
        partner_df.index.name = 'number'
        return partner_df

    def calculate_all_features(self):
        """벡터화 연산을 사용하여 모든 피처를 한 번에 계산"""
        if self.features_df is not None:
            print("⚡️ 캐시된 피처를 사용합니다.")
            return self.features_df

        start_time = time.time()
        print("🚀 모든 피처를 새로 계산합니다 (벡터화 방식)... 시간이 소요될 수 있습니다.")

        # --- 기본 피처 계산 ---
        df = self._create_feature_grid()
        df.sort_index(inplace=True)
        grouped = df.groupby(level='number')

        print("   - (1/6) 출현 빈도 계산 중...")
        windows = [10, 30, 50, 100]
        for w in windows:
            df[f'recent_{w}_freq'] = grouped['appeared'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum()).fillna(0)
            df[f'recent_{w}_rate'] = df[f'recent_{w}_freq'] / w

        print("   - (2/6) 휴면 기간 계산 중...")
        appeared_draws = df.index.get_level_values('draw_no').to_series(index=df.index)
        df['last_appeared_draw'] = appeared_draws.where(df['appeared'] == 1)
        df['last_appeared_draw'] = grouped['last_appeared_draw'].ffill().groupby(level='number').shift(1)
        df['dormant_period'] = (df.index.get_level_values('draw_no') - df['last_appeared_draw']).fillna(999).astype(int)

        print("   - (3/6) 재출현 간격 통계 계산 중...")
        df['appeared_draw'] = np.where(df['appeared'] == 1, df.index.get_level_values('draw_no'), np.nan)
        df['reappear_gap'] = grouped['appeared_draw'].transform(lambda x: x.diff())
        gap_windows = [10, 30, 50, 1000]
        for w in gap_windows:
            df[f'avg_reappear_gap_{w}'] = grouped['reappear_gap'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean()).fillna(0)
            df[f'std_reappear_gap_{w}'] = grouped['reappear_gap'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).std()).fillna(0)
            df[f'max_reappear_gap_{w}'] = grouped['reappear_gap'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max()).fillna(0)

        print("   - (4/6) 전체 출현율 및 모멘텀 계산 중...")
        df['total_appearance_rate'] = grouped['appeared'].transform(lambda x: x.shift(1).expanding(1).mean()).fillna(0)
        df['momentum'] = (df['recent_10_freq'] * 0.5 + df['recent_30_freq'] * 0.3 + df['recent_50_freq'] * 0.2).fillna(0)

        print("   - (5/6) 기본 속성 피처 계산 중...")
        df_reset = df.reset_index()
        df['range_group'] = pd.cut(df_reset['number'].values, bins=[0, 10, 20, 30, 40, 45], labels=['1-10', '11-20', '21-30', '31-40', '41-45'])
        df['is_odd'] = (df_reset['number'].values % 2).astype(int)
        df['trend_ratio'] = (df['recent_10_freq'] / df['recent_30_freq']).fillna(0).replace(np.inf, 0)

        # --- 궁합 피처 계산 및 병합 ---
        pair_features_df = self._calculate_pair_features(self.df)
        
        # df의 인덱스에 맞게 병합
        df = df.reset_index().merge(pair_features_df, on='number', how='left').set_index(['draw_no', 'number'])
        
        # 사용하지 않는 중간 컬럼 제거
        df = df.drop(columns=['last_appeared_draw', 'appeared_draw', 'reappear_gap'])
        
        self.features_df = df
        end_time = time.time()
        print(f"✅ 모든 피처 계산 완료! (소요 시간: {end_time - start_time:.2f}초)")
        return df

    def build_number_training_data(self, start_draw=100, end_draw=None):
        """번호 예측용 학습 데이터셋 생성 (고속 슬라이싱)"""
        if self.features_df is None:
            self.calculate_all_features()

        if end_draw is None:
            end_draw = int(self.df.index.max())
        
        print(f"🔪 학습 데이터 슬라이싱: {start_draw}회 ~ {end_draw}회")
        
        # 1. 피처(X)와 타겟(y) 데이터 슬라이싱
        # X: start_draw ~ end_draw 회차의 피처를 사용
        # y: start_draw ~ end_draw 회차의 출현 여부를 타겟으로 사용
        train_indices = (self.features_df.index.get_level_values('draw_no') >= start_draw) & \
                        (self.features_df.index.get_level_values('draw_no') <= end_draw)
        
        features_slice = self.features_df.loc[train_indices]
        
        X = features_slice.drop(columns=['appeared'])
        y = features_slice['appeared']
        draw_list = features_slice.index.get_level_values('draw_no').tolist()

        # 범주형 변수 인코딩
        if 'range_group' in X.columns:
            X = pd.get_dummies(X, columns=['range_group'], prefix='range')
        
        print(f"✅ 학습 데이터 생성 완료: {len(X)}개 샘플")
        return X, y, draw_list

    def extract_number_features(self, target_draw_no):
        """특정 회차의 모든 번호에 대한 피처 추출 (고속)"""
        if self.features_df is None:
            self.calculate_all_features()
        
        # target_draw_no에 해당하는 피처를 가져옴
        try:
            features_for_draw = self.features_df.loc[target_draw_no]
        except KeyError:
            raise ValueError(f"{target_draw_no}회차에 대한 피처를 계산할 수 없습니다. 데이터 범위를 확인하세요.")
        
        return features_for_draw.reset_index().drop(columns=['appeared'])

    def extract_combo_features(self, numbers, reference_draw_no, settings_features=None):
        """6개 번호 조합의 피처 추출 (고속)"""
        numbers = sorted(numbers)
        features = {}

        # 1. 조합 자체의 통계
        features['sum_total'] = sum(numbers)
        features['number_range'] = max(numbers) - min(numbers)
        features['avg_number'] = np.mean(numbers)
        features['std_number'] = np.std(numbers)
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        features['odd_count'] = odd_count
        features['even_count'] = 6 - odd_count
        
        # 2. 번호별 개별 피처의 평균/합계 (벡터화된 방식으로 추출)
        try:
            num_features_df = self.extract_number_features(reference_draw_no)
            combo_num_features = num_features_df[num_features_df['number'].isin(numbers)]

            features['avg_dormant'] = combo_num_features['dormant_period'].mean()
            features['max_dormant'] = combo_num_features['dormant_period'].max()
            features['total_momentum'] = combo_num_features['momentum'].sum()
            features['avg_momentum'] = combo_num_features['momentum'].mean()
            features['total_recent_10_freq'] = combo_num_features['recent_10_freq'].sum()

        except ValueError:
             # 예측 시점의 피처를 계산할 수 없는 경우 (너무 과거 데이터 등)
            features['avg_dormant'] = 0
            features['max_dormant'] = 0
            features['total_momentum'] = 0
            features['avg_momentum'] = 0
            features['total_recent_10_freq'] = 0
        
        # 3. 예측 설정 피처 추가
        if settings_features:
            features.update(settings_features)

        return features

    def _extract_settings_features(self, settings_str):
        """
        예측 설정(JSON 문자열)을 피처 딕셔너리로 변환합니다.
        """
        if not settings_str or settings_str == '{}':
            return {}
            
        try:
            settings = json.loads(settings_str)
        except (json.JSONDecodeError, TypeError):
            return {}

        features = {}
        
        # Generation Method
        gen_method = settings.get('generation_method', 'unknown')
        features['gen_method_ai_combo'] = 1 if 'AI 조합' in gen_method else 0
        features['gen_method_ai_prob'] = 1 if 'AI 확률' in gen_method else 0
        features['gen_method_random'] = 1 if '랜덤' in gen_method else 0

        # Training Range
        try:
            train_range = settings.get('training_range', '0-0').split('-')
            features['train_range_start'] = int(train_range[0])
            features['train_range_end'] = int(train_range[1])
            features['train_range_len'] = features['train_range_end'] - features['train_range_start']
        except (ValueError, IndexError):
            features['train_range_start'] = 0
            features['train_range_end'] = 0
            features['train_range_len'] = 0

        # Max Overlap
        features['max_overlap'] = settings.get('max_overlap', 6)

        # Filters
        filters = settings.get('filters', {})
        features['filter_exclude_recent'] = filters.get('exclude_recent_draws', 0)
        
        # Odd/Even Balance
        oeb_options = ['6:0', '5:1', '4:2', '3:3', '2:4', '1:5', '0:6']
        oeb_filters = filters.get('odd_even_balance', [])
        for opt in oeb_options:
            features[f'filter_oeb_{opt.replace(":", "_")}'] = 1 if opt in oeb_filters else 0

        # Consecutive Lengths
        ecl_filters = filters.get('exclude_consecutive_lengths', [])
        for i in range(2, 7):
            features[f'filter_ecl_{i}'] = 1 if i in ecl_filters else 0
            
        # Range Limits
        rl_filters = filters.get('range_limits', {})
        for i in range(5):
            features[f'filter_rl_{i}'] = rl_filters.get(str(i), 6) # 6 is default (no limit)

        return features

    def build_combo_training_data_with_feedback(self, start_draw=100, end_draw=None, negative_samples_per_draw=3):
        """
        [신규] 예측 피드백을 포함한 조합 예측용 학습 데이터셋 생성
        - Target: 맞춘 개수 (0-6, Regression)
        - Samples:
            1. 과거 당첨 조합 (y=6)
            2. 과거 예측 조합 (y=실제 맞춘 개수, settings 피처 포함)
            3. 랜덤 생성 조합 (y=실제 맞춘 개수)
        """
        if end_draw is None:
            end_draw = self.get_latest_draw_number()

        all_features = []
        all_targets = []
        all_draws = []

        self.calculate_all_features()
        
        print(f"🛠️ [피드백] 조합 학습 데이터 생성 시작: {start_draw}회~{end_draw}회")

        # --- 1. 과거 당첨 조합 (Positive Samples) ---
        df_slice = self.df.loc[start_draw:end_draw]
        print(f"   - (1/3) 과거 당첨 조합 {len(df_slice)}개 처리 중...")
        for draw_no, row in df_slice.iterrows():
            winning_combo = [int(row[f'n{i}']) for i in range(1, 7)]
            try:
                # 당첨 조합은 예측 설정이 없으므로 settings_features=None
                combo_features = self.extract_combo_features(winning_combo, draw_no, None)
                all_features.append(combo_features)
                all_targets.append(6) # 1등은 6개 맞춤
                all_draws.append(draw_no)
            except ValueError:
                print(f"⚠️ {draw_no}회차 당첨 조합 피처 생성 실패")
                continue
        
        # --- 2. 과거 예측 조합 (Feedback Samples) ---
        feedback_path = self.data_path.parent / 'prediction_results.csv'
        if feedback_path.exists():
            feedback_df = pd.read_csv(feedback_path)
            feedback_df = feedback_df[(feedback_df['draw_no'] >= start_draw) & (feedback_df['draw_no'] <= end_draw)]
            print(f"   - (2/3) 예측 피드백 데이터 {len(feedback_df)}개 처리 중...")

            for _, row in feedback_df.iterrows():
                draw_no = row['draw_no']
                try:
                    combo = json.loads(row['combo'])
                    matches = row['matches']
                    settings_features = self._extract_settings_features(row['settings'])
                    
                    combo_features = self.extract_combo_features(combo, draw_no, settings_features)
                    all_features.append(combo_features)
                    all_targets.append(matches)
                    all_draws.append(draw_no)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"⚠️ {draw_no}회차 피드백 데이터 처리 중 오류: {e}")
                    continue
        else:
            print("   - (2/3) 예측 피드백 파일 없음. 건너뜁니다.")

        # --- 3. 랜덤 조합 (Negative Samples) ---
        print(f"   - (3/3) 랜덤 샘플 생성 중...")
        winning_combos_map = {
            draw_no: set(row[[f'n{i}' for i in range(1, 7)]].values)
            for draw_no, row in df_slice.iterrows()
        }
        
        for draw_no in df_slice.index:
            winning_set = winning_combos_map.get(draw_no)
            if not winning_set: continue

            generated_count = 0
            attempts = 0
            while generated_count < negative_samples_per_draw and attempts < 100:
                attempts += 1
                random_combo = sorted(np.random.choice(range(1, 46), size=6, replace=False))
                
                try:
                    # 랜덤 조합은 예측 설정이 없으므로 settings_features=None
                    combo_features = self.extract_combo_features(random_combo, draw_no, None)
                    matches = len(set(random_combo) & winning_set)
                    
                    all_features.append(combo_features)
                    all_targets.append(matches)
                    all_draws.append(draw_no)
                    generated_count += 1
                except ValueError:
                    continue

        if not all_features:
            print("⚠️ 생성된 학습 데이터가 없습니다. 회차 범위를 확인해주세요.")
            return pd.DataFrame(), pd.Series(), []

        X = pd.DataFrame(all_features).fillna(0)
        y = pd.Series(all_targets, dtype=float)
        
        print(f"✅ [피드백] 조합 학습 데이터 생성 완료: {len(X)}개 샘플")
        return X, y, all_draws

    def build_combo_training_data(self, start_draw=100, end_draw=None, negative_samples=5):
        """[구버전] 조합 예측용 학습 데이터셋 생성 (Target: 0 or 1)"""
        if end_draw is None:
            end_draw = self.get_latest_draw_number()

        all_features = []
        all_targets = []
        all_draws = []

        self.calculate_all_features()

        df_slice = self.df.loc[start_draw:end_draw]
        
        print(f"🛠️ [구버전] 조합 학습 데이터 생성 시작: {start_draw}회~{end_draw}회 ({len(df_slice)}회차)")

        for draw_no, row in df_slice.iterrows():
            winning_combo = [int(row[f'n{i}']) for i in range(1, 7)]
            try:
                combo_features = self.extract_combo_features(winning_combo, draw_no)
                all_features.append(combo_features)
                all_targets.append(1) # 1 for winning
                all_draws.append(draw_no)
            except ValueError:
                print(f"⚠️ {draw_no}회차 당첨 조합 피처 생성 실패")
                continue

            existing_combos = {tuple(sorted(winning_combo))}
            generated_count = 0
            attempts = 0
            while generated_count < negative_samples and attempts < 1000:
                attempts += 1
                random_combo = sorted(np.random.choice(range(1, 46), size=6, replace=False))
                if tuple(random_combo) not in existing_combos:
                    existing_combos.add(tuple(random_combo))
                    try:
                        combo_features = self.extract_combo_features(random_combo, draw_no)
                        all_features.append(combo_features)
                        all_targets.append(0) # 0 for random
                        all_draws.append(draw_no)
                        generated_count += 1
                    except ValueError:
                        continue
        
        if not all_features:
            print("⚠️ 생성된 학습 데이터가 없습니다.")
            return pd.DataFrame(), pd.Series(), []

        X = pd.DataFrame(all_features).fillna(0)
        y = pd.Series(all_targets)
        
        print(f"✅ [구버전] 조합 학습 데이터 생성 완료: {len(X)}개 샘플")
        return X, y, all_draws

    def get_latest_draw_number(self):
        """최신 회차 번호 반환"""
        return int(self.df.index.max())


if __name__ == "__main__":
    # 테스트
    engineer = LottoFeatureEngineer()
    
    print("\n" + "="*60)
    print("📊 Feature Engineer 속도 테스트")
    print("="*60)
    
    latest_draw = engineer.get_latest_draw_number()
    
    # 1. 전체 피처 계산 테스트
    engineer.calculate_all_features()
    
    # 2. 신규 학습 데이터셋 생성 테스트
    start = max(100, latest_draw - 100)
    end = latest_draw
    
    start_time = time.time()
    X, y, draws = engineer.build_combo_training_data_with_feedback(start_draw=start, end_draw=end)
    end_time = time.time()
    
    print(f"\n[ 신규 학습 데이터 생성 테스트 ]")
    if not X.empty:
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - Target (y) stats: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
        print(f"   - 소요 시간: {end_time - start_time:.2f}초")
        print(f"\n피처 목록 ({len(X.columns)}개):")
        print(X.columns.tolist())
    else:
        print("   - 생성된 데이터가 없습니다.")