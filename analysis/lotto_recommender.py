"""
로또 6/45 번호 추천 모듈
ML 예측 확률 및 사용자 정의 필터 기반 번호 추천 시스템
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

# LottoNumberPredictor 클래스를 임포트하기 위한 경로 설정
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lotto_number_predictor import LottoNumberPredictor


class LottoRecommender:
    """로또 번호 추천 클래스 (ML 예측 연동)"""
    
    def __init__(self, predictor: LottoNumberPredictor, data_path='data/lotto_history.csv'):
        self.predictor = predictor
        self.data_path = Path(data_path)
        self.df = None
        self.load_data()
        
        # 필터 설정
        self.filters = {
            'odd_even_balance': [],
            'exclude_recent_draws': 0,
            'exclude_consecutive_lengths': [],
            'range_limits': {},
        }
    
    def load_data(self):
        """데이터 로드"""
        if self.data_path.exists():
            self.df = pd.read_csv(self.data_path)
        else:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
    
    def set_filters(self, **kwargs):
        """필터 설정"""
        for key, value in kwargs.items():
            if key in self.filters:
                self.filters[key] = value
    
    def _get_recent_combinations(self, recent_draws=10):
        """최근 N회차 당첨 조합 추출"""
        if recent_draws <= 0:
            return set()
        recent_df = self.df.tail(recent_draws)
        recent_combos = set()
        for index, row in recent_df.iterrows():
            combo = tuple(sorted([int(row[f'n{i}']) for i in range(1, 7)]))
            recent_combos.add(combo)
        return recent_combos

    def _check_consecutive_rules(self, numbers):
        """연속 번호 규칙 검사"""
        exclude_lengths = self.filters.get('exclude_consecutive_lengths', [])
        if not exclude_lengths: return True
        numbers = sorted(numbers)
        if not numbers: return True
        current_streak = 1
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] == 1:
                current_streak += 1
            else:
                if current_streak in exclude_lengths: return False
                current_streak = 1
        return current_streak not in exclude_lengths

    def _check_range_limits(self, numbers):
        """구간별 번호 개수 제한 규칙 검사"""
        limits = self.filters.get('range_limits', {})
        if not limits: return True
        range_counts = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
        for n in numbers:
            if 1 <= n <= 9: range_counts['0'] += 1
            elif 10 <= n <= 19: range_counts['1'] += 1
            elif 20 <= n <= 29: range_counts['2'] += 1
            elif 30 <= n <= 39: range_counts['3'] += 1
            elif 40 <= n <= 45: range_counts['4'] += 1
        for range_key, max_count in limits.items():
            if range_counts[range_key] > max_count: return False
        return True

    def _check_odd_even_balance(self, numbers):
        """홀짝 밸런스 체크"""
        balance_ratios = self.filters.get('odd_even_balance', [])
        if not balance_ratios: return True
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        for ratio in balance_ratios:
            try:
                target_odd, _ = map(int, ratio.split(':'))
                if odd_count == target_odd: return True
            except (ValueError, AttributeError): continue
        return False

    def apply_filters(self, numbers, include_numbers=None):
        """모든 필터 적용"""
        if include_numbers and not all(n in numbers for n in include_numbers):
            return False
        if not self._check_odd_even_balance(numbers): return False
        if self.filters['exclude_recent_draws'] > 0:
            recent_combos = self._get_recent_combinations(self.filters['exclude_recent_draws'])
            if tuple(sorted(numbers)) in recent_combos: return False
        if not self._check_consecutive_rules(numbers): return False
        if not self._check_range_limits(numbers): return False
        return True

    def generate_numbers(self, feature_engineer, count=5, include_numbers=None, max_attempts=20000, max_overlap=6, target_draw_no=None):
        """
        ML 예측 확률 기반으로 번호를 추천 생성하고 필터링합니다.
        
        Args:
            feature_engineer: 피처 엔지니어링 객체
            count: 생성할 조합 개수
            include_numbers: 포함할 번호 목록
            max_attempts: 최대 시도 횟수
            max_overlap: 조합 간 최대 허용 중복 번호 개수
            target_draw_no: 예측 대상 회차
        """
        print("\n🎲 ML 예측 확률 기반 번호 추천 생성 시작...")
        
        # 1. ML 모델로 번호별 출현 확률 예측
        try:
            raw_probs = self.predictor.predict_probabilities(feature_engineer, draw_no=target_draw_no)
            print(f"   - {target_draw_no or '다음'}회차 번호별 예측 확률 확보 완료")
        except (RuntimeError, ValueError) as e:
            print(f"⚠️ 예측 모델 실행 중 오류: {e}")
            print("   - 오류로 인해 일반 랜덤 샘플링으로 전환합니다.")
            raw_probs = {i: 1/45 for i in range(1, 46)} # Fallback

        # 2. 확률 정규화 및 가중치 설정
        population = sorted(raw_probs.keys())
        probabilities = np.array([raw_probs[n] for n in population])
        probabilities /= probabilities.sum() # 정규화

        recommendations = []
        attempts = 0
        
        include_numbers = [int(n) for n in include_numbers or [] if 1 <= int(n) <= 45]
        if len(include_numbers) > 6:
            raise ValueError("포함 번호는 최대 6개까지 가능합니다.")

        print(f"   - 필터와 분산 로직을 적용하여 {count}개 조합 생성 중...")
        while len(recommendations) < count and attempts < max_attempts:
            attempts += 1
            
            # 3. 가중치 기반으로 번호 조합 생성
            remaining_count = 6 - len(include_numbers)
            
            # 포함 번호를 제외한 모집단과 확률 재설정
            current_population = [n for n in population if n not in include_numbers]
            current_probs = np.array([raw_probs[n] for n in current_population])
            current_probs /= current_probs.sum()

            if remaining_count > 0:
                remaining_numbers = np.random.choice(current_population, size=remaining_count, replace=False, p=current_probs)
                numbers = sorted(include_numbers + list(remaining_numbers))
            else:
                numbers = sorted(include_numbers)
            
            # 4. 필터 적용
            if not self.apply_filters(numbers, include_numbers):
                continue

            # 5. 조합 간 중복(분산) 체크
            is_overlapped = False
            if max_overlap < 6:
                for r in recommendations:
                    if len(set(numbers) & set(r)) > max_overlap:
                        is_overlapped = True
                        break
            
            if not is_overlapped:
                recommendations.append(numbers)
        
        if len(recommendations) < count:
            print(f"⚠️ 필터 조건이 너무 엄격하거나 시도 횟수가 부족합니다. {len(recommendations)}개만 생성되었습니다.")
        else:
            print("   - 생성 완료!")
        
        return recommendations

    def get_active_filters(self):
        """활성화된 필터 목록 반환"""
        active = []
        if self.filters.get('odd_even_balance'):
            active.append(f"홀짝 밸런스 ({', '.join(self.filters['odd_even_balance'])})")
        if self.filters.get('exclude_recent_draws', 0) > 0:
            active.append(f"최근 {self.filters['exclude_recent_draws']}회 당첨조합 제외")
        if self.filters.get('exclude_consecutive_lengths'):
            active.append(f"연속 번호 제외 ({', '.join(map(str, self.filters['exclude_consecutive_lengths']))}개 짜리)")
        range_filters = []
        range_map = {'0': '1-9', '1': '10', '2': '20', '3': '30', '4': '40'}
        for key, limit in self.filters.get('range_limits', {}).items():
            if limit < 6:
                range_filters.append(f"{range_map[key]}번대 최대 {limit}개")
        if range_filters:
            active.append(", ".join(range_filters))
        return active if active else ["적용된 필터 없음"]


if __name__ == "__main__":
    # 테스트를 위한 의존성 객체 생성
    from lotto_feature_engineer import LottoFeatureEngineer
    
    try:
        # 1. 피처 엔지니어와 예측 모델 준비
        print("테스트를 위해 피처 엔지니어와 예측 모델을 준비합니다.")
        engineer = LottoFeatureEngineer()
        predictor = LottoNumberPredictor()
        
        # 모델이 없으면 학습, 있으면 로드
        model_path = Path('models/number_predictor.pkl')
        if not model_path.exists():
            print("저장된 모델이 없어 새로 학습합니다.")
            predictor.train(engineer, end_draw=engineer.get_latest_draw_number() - 50)
            predictor.save_model(str(model_path))
        else:
            predictor.load_model(str(model_path), expected_feature_version=engineer.get_feature_version())

        # 2. 추천기 객체 생성 (예측 모델 주입)
        recommender = LottoRecommender(predictor=predictor)
        
        # 3. 필터 설정
        recommender.set_filters(
            odd_even_balance=['4:2', '3:3', '2:4'],
            exclude_consecutive_lengths=[3, 4],
            range_limits={'1': 3, '3': 3},
            exclude_recent_draws=10
        )
        
        # 4. 번호 추천 생성
        target_draw = engineer.get_latest_draw_number() + 1
        recommended_numbers = recommender.generate_numbers(
            feature_engineer=engineer,
            count=10,
            max_overlap=3, # 조합 간 최대 3개까지만 겹치도록 설정
            target_draw_no=target_draw
        )
        
        print("\n" + "="*60)
        print(f"🎯 최종 추천 번호 ({target_draw}회차)")
        print("="*60)
        for i, nums in enumerate(recommended_numbers, 1):
            print(f"  [조합 {i:2d}] {nums}")
        
        print(f"\n🔧 적용된 필터: {', '.join(recommender.get_active_filters())}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ 테스트 실행 중 오류가 발생했습니다: {e}")
        print("   - 데이터 파일이나 학습된 모델이 준비되었는지 확인하세요.")
        print("   - app.py를 통해 전체 프로세스를 실행하는 것을 권장합니다.")

