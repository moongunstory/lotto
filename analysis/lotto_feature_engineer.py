"""
로또 6/45 피처 엔지니어링 모듈
ML 학습을 위한 피처 생성 및 데이터셋 구축
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


class LottoFeatureEngineer:
    """로또 피처 엔지니어링 클래스"""
    
    def __init__(self, data_path='data/lotto_history.csv'):
        self.data_path = Path(data_path)
        self.df = None
        self.load_data()
        
    def load_data(self):
        """데이터 로드"""
        if self.data_path.exists():
            self.df = pd.read_csv(self.data_path)
            print(f"✅ 데이터 로드 완료: {len(self.df)}회차")
        else:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
    
    # ========== 번호 히스토리 분석 ==========
    
    def _get_number_history(self, number, until_draw=None):
        """특정 번호의 출현 이력 반환"""
        if until_draw is None:
            df_subset = self.df
        else:
            df_subset = self.df[self.df['draw_no'] < until_draw]
        
        appearances = []
        for idx, row in df_subset.iterrows():
            if number in [row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']]:
                appearances.append(row['draw_no'])
        
        return appearances
    
    def _calculate_dormant_period(self, number, current_draw):
        """휴면 기간 계산 (마지막 출현 후 경과 회차)"""
        history = self._get_number_history(number, until_draw=current_draw)
        
        if not history:
            return 999  # 한 번도 안 나온 경우
        
        last_appearance = max(history)
        return current_draw - last_appearance
    
    def _calculate_reappear_gaps(self, number, until_draw=None):
        """재출현 간격들 계산"""
        history = self._get_number_history(number, until_draw)
        
        if len(history) < 2:
            return []
        
        gaps = []
        for i in range(1, len(history)):
            gaps.append(history[i] - history[i-1])
        
        return gaps
    
    # ========== 번호별 피처 추출 ==========
    
    def extract_number_features(self, target_draw_no):
        """
        특정 회차 시점에서 각 번호(1~45)의 피처 추출
        
        Args:
            target_draw_no: 타겟 회차 번호
            
        Returns:
            DataFrame: (45 rows × N features)
        """
        features_list = []
        
        for number in range(1, 46):
            features = self._extract_single_number_features(number, target_draw_no)
            features['number'] = number
            features_list.append(features)
        
        df_features = pd.DataFrame(features_list)
        return df_features
    
    def _extract_single_number_features(self, number, target_draw_no):
        """단일 번호의 피처 추출"""
        features = {}
        
        # 1. 최근 N회 출현 빈도
        for window in [10, 30, 50, 100]:
            start_draw = max(1, target_draw_no - window)
            recent_df = self.df[(self.df['draw_no'] >= start_draw) & 
                               (self.df['draw_no'] < target_draw_no)]
            
            count = 0
            for _, row in recent_df.iterrows():
                if number in [row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']]:
                    count += 1
            
            features[f'recent_{window}_freq'] = count
            features[f'recent_{window}_rate'] = count / window if window > 0 else 0
        
        # 2. 휴면 기간
        features['dormant_period'] = self._calculate_dormant_period(number, target_draw_no)
        
        # 3. 재출현 간격 통계
        gaps = self._calculate_reappear_gaps(number, until_draw=target_draw_no)
        if gaps:
            features['avg_reappear_gap'] = np.mean(gaps)
            features['std_reappear_gap'] = np.std(gaps) if len(gaps) > 1 else 0
            features['min_reappear_gap'] = np.min(gaps)
            features['max_reappear_gap'] = np.max(gaps)
        else:
            features['avg_reappear_gap'] = 0
            features['std_reappear_gap'] = 0
            features['min_reappear_gap'] = 0
            features['max_reappear_gap'] = 0
        
        # 4. 전체 출현율
        total_history = self._get_number_history(number, until_draw=target_draw_no)
        total_draws = len(self.df[self.df['draw_no'] < target_draw_no])
        features['total_appearance_rate'] = len(total_history) / total_draws if total_draws > 0 else 0
        
        # 5. 출현 모멘텀 (최근일수록 가중치)
        momentum = 0
        for window, weight in [(10, 0.5), (30, 0.3), (50, 0.2)]:
            start_draw = max(1, target_draw_no - window)
            recent_df = self.df[(self.df['draw_no'] >= start_draw) & 
                               (self.df['draw_no'] < target_draw_no)]
            
            count = 0
            for _, row in recent_df.iterrows():
                if number in [row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']]:
                    count += 1
            
            momentum += count * weight
        
        features['momentum'] = momentum
        
        # 6. 구간별 정보
        if 1 <= number <= 10:
            range_group = '1-10'
        elif 11 <= number <= 20:
            range_group = '11-20'
        elif 21 <= number <= 30:
            range_group = '21-30'
        elif 31 <= number <= 40:
            range_group = '31-40'
        else:
            range_group = '41-45'
        
        features['range_group'] = range_group
        
        # 7. 홀짝
        features['is_odd'] = 1 if number % 2 == 1 else 0
        
        # 8. 최근 추세 (최근 30회 vs 최근 10회 비율)
        if features['recent_30_freq'] > 0:
            features['trend_ratio'] = features['recent_10_freq'] / features['recent_30_freq']
        else:
            features['trend_ratio'] = 0
        
        return features
    
    # ========== 조합 피처 추출 ==========
    
    def extract_combo_features(self, numbers, reference_draw_no):
        """
        6개 번호 조합의 피처 추출
        
        Args:
            numbers: 6개 번호 리스트
            reference_draw_no: 기준 회차 번호
            
        Returns:
            dict: 조합 피처
        """
        numbers = sorted(numbers)
        features = {}
        
        # 1. 기본 통계
        features['sum_total'] = sum(numbers)
        features['number_range'] = max(numbers) - min(numbers)
        features['avg_number'] = np.mean(numbers)
        features['std_number'] = np.std(numbers)
        
        # 2. 홀짝 분포
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        features['odd_count'] = odd_count
        features['even_count'] = 6 - odd_count
        features['odd_even_balance'] = abs(odd_count - 3)  # 3:3에서 얼마나 벗어났는지
        
        # 3. 연속번호
        consecutive_pairs = 0
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] == 1:
                consecutive_pairs += 1
        features['consecutive_pairs'] = consecutive_pairs
        
        # 4. 구간 분포
        range_dist = {'1-10': 0, '11-20': 0, '21-30': 0, '31-40': 0, '41-45': 0}
        for num in numbers:
            if 1 <= num <= 10:
                range_dist['1-10'] += 1
            elif 11 <= num <= 20:
                range_dist['11-20'] += 1
            elif 21 <= num <= 30:
                range_dist['21-30'] += 1
            elif 31 <= num <= 40:
                range_dist['31-40'] += 1
            else:
                range_dist['41-45'] += 1
        
        for key, val in range_dist.items():
            features[f'range_{key}'] = val
        
        # 구간 집중도 (entropy)
        probs = [v/6 for v in range_dist.values() if v > 0]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        features['range_entropy'] = entropy
        
        # 5. 번호별 개별 피처의 평균
        dormant_periods = []
        momentums = []
        recent_10_freqs = []
        
        for num in numbers:
            num_features = self._extract_single_number_features(num, reference_draw_no)
            dormant_periods.append(num_features['dormant_period'])
            momentums.append(num_features['momentum'])
            recent_10_freqs.append(num_features['recent_10_freq'])
        
        features['avg_dormant'] = np.mean(dormant_periods)
        features['max_dormant'] = np.max(dormant_periods)
        features['total_momentum'] = sum(momentums)
        features['avg_momentum'] = np.mean(momentums)
        features['total_recent_10_freq'] = sum(recent_10_freqs)
        
        # 6. 최근 패턴 유사도 (최근 10회 당첨번호와의 유사도)
        recent_draws = self.df[self.df['draw_no'] < reference_draw_no].tail(10)
        similarity_scores = []
        
        for _, row in recent_draws.iterrows():
            recent_numbers = [row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']]
            overlap = len(set(numbers) & set(recent_numbers))
            similarity_scores.append(overlap)
        
        features['avg_similarity_to_recent'] = np.mean(similarity_scores) if similarity_scores else 0
        features['max_similarity_to_recent'] = np.max(similarity_scores) if similarity_scores else 0
        
        return features
    
    # ========== 학습 데이터셋 생성 ==========
    
    def build_number_training_data(self, start_draw=100, end_draw=None):
        """
        번호 예측용 학습 데이터셋 생성
        
        Args:
            start_draw: 시작 회차 (초기 데이터는 피처 계산에 필요)
            end_draw: 종료 회차 (None이면 최신 회차까지)
            
        Returns:
            X: 피처 DataFrame
            y: 타겟 (각 번호가 다음 회차에 출현했는지 0/1)
            draw_numbers: 회차 번호 리스트
        """
        if end_draw is None:
            end_draw = int(self.df['draw_no'].max())
        
        X_list = []
        y_list = []
        draw_list = []
        
        print(f"📊 학습 데이터 생성 중: {start_draw}회 ~ {end_draw-1}회")
        
        for draw_no in range(start_draw, end_draw):
            # 현재 회차의 피처 추출
            features_df = self.extract_number_features(draw_no)
            
            # 다음 회차의 실제 당첨번호
            next_draw = self.df[self.df['draw_no'] == draw_no]
            if next_draw.empty:
                continue
            
            next_draw = next_draw.iloc[0]
            winning_numbers = [
                int(next_draw['n1']), int(next_draw['n2']), int(next_draw['n3']),
                int(next_draw['n4']), int(next_draw['n5']), int(next_draw['n6'])
            ]
            
            # 각 번호마다 타겟 생성
            for idx, row in features_df.iterrows():
                number = int(row['number'])
                is_winning = 1 if number in winning_numbers else 0
                
                # 피처와 타겟 저장
                feature_dict = row.drop('number').to_dict()
                X_list.append(feature_dict)
                y_list.append(is_winning)
                draw_list.append(draw_no)
        
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        
        # 범주형 변수 인코딩
        if 'range_group' in X.columns:
            X = pd.get_dummies(X, columns=['range_group'], prefix='range')
        
        print(f"✅ 학습 데이터 생성 완료: {len(X)}개 샘플")
        print(f"   - 출현(1): {y.sum()}개 ({y.mean()*100:.2f}%)")
        print(f"   - 미출현(0): {(~y.astype(bool)).sum()}개")
        
        return X, y, draw_list
    
    def build_combo_training_data(self, start_draw=100, end_draw=None, negative_samples=5):
        """
        조합 예측용 학습 데이터셋 생성
        
        Args:
            start_draw: 시작 회차
            end_draw: 종료 회차
            negative_samples: 각 당첨 조합당 생성할 음성 샘플 수
            
        Returns:
            X: 피처 DataFrame
            y: 타겟 (실제 당첨=1, 랜덤 샘플=0)
            draw_numbers: 회차 번호 리스트
        """
        if end_draw is None:
            end_draw = int(self.df['draw_no'].max())
        
        X_list = []
        y_list = []
        draw_list = []
        
        print(f"📊 조합 학습 데이터 생성 중: {start_draw}회 ~ {end_draw}회")
        
        for draw_no in range(start_draw, end_draw + 1):
            draw_row = self.df[self.df['draw_no'] == draw_no]
            if draw_row.empty:
                continue
            
            draw_row = draw_row.iloc[0]
            winning_numbers = [
                int(draw_row['n1']), int(draw_row['n2']), int(draw_row['n3']),
                int(draw_row['n4']), int(draw_row['n5']), int(draw_row['n6'])
            ]
            
            # 1. 실제 당첨 조합 (positive sample)
            features = self.extract_combo_features(winning_numbers, draw_no)
            X_list.append(features)
            y_list.append(1)
            draw_list.append(draw_no)
            
            # 2. 랜덤 조합 (negative samples)
            for _ in range(negative_samples):
                random_numbers = sorted(np.random.choice(range(1, 46), size=6, replace=False))
                features = self.extract_combo_features(random_numbers, draw_no)
                X_list.append(features)
                y_list.append(0)
                draw_list.append(draw_no)
        
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        
        print(f"✅ 조합 학습 데이터 생성 완료: {len(X)}개 샘플")
        print(f"   - 당첨 조합(1): {y.sum()}개")
        print(f"   - 랜덤 조합(0): {(~y.astype(bool)).sum()}개")
        
        return X, y, draw_list
    
    def get_latest_draw_number(self):
        """최신 회차 번호 반환"""
        return int(self.df['draw_no'].max())


if __name__ == "__main__":
    # 테스트
    engineer = LottoFeatureEngineer()
    
    print("\n" + "="*60)
    print("📊 Feature Engineer 테스트")
    print("="*60)
    
    # 1. 번호별 피처 추출 테스트
    print("\n[1] 번호별 피처 추출 (1000회차 시점)")
    latest_draw = engineer.get_latest_draw_number()
    test_draw = min(1000, latest_draw)
    
    features = engineer.extract_number_features(test_draw)
    print(features.head(10))
    print(f"\n피처 개수: {len(features.columns)}개")
    print(f"번호 개수: {len(features)}개")
    
    # 2. 조합 피처 추출 테스트
    print("\n[2] 조합 피처 추출")
    test_numbers = [7, 12, 27, 31, 38, 42]
    combo_features = engineer.extract_combo_features(test_numbers, test_draw)
    print(f"조합: {test_numbers}")
    print(f"피처 개수: {len(combo_features)}개")
    for key, val in list(combo_features.items())[:10]:
        print(f"  {key}: {val}")
    
    # 3. 학습 데이터셋 생성 테스트 (작은 범위)
    print("\n[3] 학습 데이터셋 생성 테스트")
    start = max(100, latest_draw - 50)
    end = latest_draw
    
    X, y, draws = engineer.build_number_training_data(start_draw=start, end_draw=end)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\n피처 목록:")
    print(X.columns.tolist())
