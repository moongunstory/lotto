"""
로또 6/45 피처 엔지니어링 모듈 (고속 벡터화 버전)
ML 학습을 위한 피처 생성 및 데이터셋 구축
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

class LottoFeatureEngineer:
    """로또 피처 엔지니어링 클래스 (벡터화 최적화)"""

    # 피처 구성이 변경될 때마다 버전을 갱신한다.
    FEATURE_VERSION = "2024.02"
    
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

    def calculate_all_features(self):
        """벡터화 연산을 사용하여 모든 피처를 한 번에 계산"""
        if self.features_df is not None:
            print("⚡️ 캐시된 피처를 사용합니다.")
            return self.features_df

        start_time = time.time()
        print("🚀 모든 피처를 새로 계산합니다 (벡터화 방식)... 시간이 소요될 수 있습니다.")

        df = self._create_feature_grid()
        df.sort_index(inplace=True)

        # 그룹화 객체 생성
        grouped = df.groupby(level='number')

        # 1. 최근 N회 출현 빈도 (롤링 윈도우 사용)
        print("   - (1/5) 출현 빈도 계산 중...")
        windows = [10, 30, 50, 100]
        for w in windows:
            # shift(1)을 통해 현재 회차를 제외하고 이전 N회차까지의 합을 구함
            df[f'recent_{w}_freq'] = grouped['appeared'].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).sum()
            ).fillna(0)
            df[f'recent_{w}_rate'] = df[f'recent_{w}_freq'] / w

        # 2. 휴면 기간 (Dormant Period) - 수정된 로직
        print("   - (2/5) 휴면 기간 계산 중...")
        appeared_draws = df.index.get_level_values('draw_no').to_series(index=df.index)
        df['last_appeared_draw'] = appeared_draws.where(df['appeared'] == 1)
        df['last_appeared_draw'] = grouped['last_appeared_draw'].ffill()
        df['dormant_period'] = (df.index.get_level_values('draw_no') - df['last_appeared_draw']).fillna(999).astype(int)

        # 3. 재출현 간격 통계
        print("   - (3/5) 재출현 간격 통계 계산 중...")
        df['appeared_draw'] = np.where(df['appeared'] == 1, df.index.get_level_values('draw_no'), np.nan)
        df['reappear_gap'] = grouped['appeared_draw'].transform(lambda x: x.diff())
        
        gap_windows = [10, 30, 50, 1000] # 1000은 거의 전체 기간을 의미
        for w in gap_windows:
            df[f'avg_reappear_gap_{w}'] = grouped['reappear_gap'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean()).fillna(0)
            df[f'std_reappear_gap_{w}'] = grouped['reappear_gap'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).std()).fillna(0)
            df[f'max_reappear_gap_{w}'] = grouped['reappear_gap'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max()).fillna(0)

        # 4. 전체 출현율 (Expanding Window 사용)
        print("   - (4/5) 전체 출현율 및 모멘텀 계산 중...")
        df['total_appearance_rate'] = grouped['appeared'].transform(
            lambda x: x.shift(1).expanding(1).mean()
        ).fillna(0)

        # 5. 출현 모멘텀
        df['momentum'] = (df['recent_10_freq'] * 0.5 + df['recent_30_freq'] * 0.3 + df['recent_50_freq'] * 0.2).fillna(0)

        # 6. 구간별 정보 & 홀짝
        df_reset = df.reset_index()
        df['range_group'] = pd.cut(df_reset['number'].values, bins=[0, 10, 20, 30, 40, 45], labels=['1-10', '11-20', '21-30', '31-40', '41-45'])
        df['is_odd'] = (df_reset['number'].values % 2).astype(int)

        # 7. 최근 추세
        df['trend_ratio'] = (df['recent_10_freq'] / df['recent_30_freq']).fillna(0).replace(np.inf, 0)

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
        
        print(f"🔪 학습 데이터 슬라이싱: {start_draw}회 ~ {end_draw-1}회")
        
        # 1. 피처(X)와 타겟(y) 데이터 슬라이싱
        # X: start_draw ~ end_draw-1 회차의 피처를 사용
        # y: start_draw ~ end_draw-1 회차의 출현 여부를 타겟으로 사용
        train_indices = (self.features_df.index.get_level_values('draw_no') >= start_draw) & \
                        (self.features_df.index.get_level_values('draw_no') < end_draw)
        
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

    def extract_combo_features(self, numbers, reference_draw_no):
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

        return features

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
    
    # 2. 학습 데이터셋 생성 테스트 (큰 범위)
    start = max(100, latest_draw - 500)
    end = latest_draw
    
    start_time = time.time()
    X, y, draws = engineer.build_number_training_data(start_draw=start, end_draw=end)
    end_time = time.time()
    
    print(f"\n[ 학습 데이터 생성 테스트 ]")
    print(f"   - X shape: {X.shape}")
    print(f"   - y shape: {y.shape}")
    print(f"   - 소요 시간: {end_time - start_time:.2f}초")
    print(f"\n피처 목록 ({len(X.columns)}개):")
    print(X.columns.tolist())

    # 3. 특정 회차 피처 추출 테스트
    start_time = time.time()
    features = engineer.extract_number_features(latest_draw)
    end_time = time.time()
    print(f"\n[ 특정 회차 피처 추출 테스트 ]")
    print(f"   - 소요 시간: {end_time - start_time:.2f}초")
    print(features.head())
