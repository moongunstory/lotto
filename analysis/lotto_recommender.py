"""
로또 6/45 번호 추천 모듈
사용자 정의 필터 기반 번호 추천 시스템
"""

import random
import pandas as pd
from pathlib import Path
from itertools import combinations


class LottoRecommender:
    """로또 번호 추천 클래스"""
    
    def __init__(self, data_path='data/lotto_history.csv'):
        self.data_path = Path(data_path)
        self.df = None
        self.load_data()
        
        # 필터 설정
        self.filters = {
            'remove_consecutive': False,
            'consecutive_level': 2,  # 2, 3, 6 중 선택
            'remove_all_even': False,
            'remove_all_odd': False,
            'remove_range_cluster': False,
            'remove_high_40s': False,
            'balance_odd_even': False,
            'exclude_recent_10': False
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
    
    def get_recent_numbers(self, recent_draws=10):
        """최근 N회차 출현 번호 추출"""
        recent_df = self.df.tail(recent_draws)
        recent_numbers = set()
        
        for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
            recent_numbers.update(recent_df[col].tolist())
        
        return recent_numbers
    
    def has_consecutive(self, numbers, level=2):
        """연속번호 체크"""
        numbers = sorted(numbers)
        consecutive_count = 0
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        if level == 2:
            return max_consecutive >= 2
        elif level == 3:
            return max_consecutive >= 3
        elif level == 6:
            return max_consecutive == 6
        
        return False
    
    def is_all_even(self, numbers):
        """모두 짝수인지 체크"""
        return all(n % 2 == 0 for n in numbers)
    
    def is_all_odd(self, numbers):
        """모두 홀수인지 체크"""
        return all(n % 2 == 1 for n in numbers)
    
    def is_range_clustered(self, numbers):
        """구간 집중 체크 (범위가 10 미만)"""
        return (max(numbers) - min(numbers)) < 10
    
    def has_high_40s(self, numbers):
        """40대 번호 몰림 체크 (5개 이상)"""
        count_40s = sum(1 for n in numbers if 40 <= n <= 45)
        return count_40s >= 5
    
    def check_odd_even_balance(self, numbers):
        """홀짝 밸런스 체크 (2:4 ~ 4:2)"""
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        return 2 <= odd_count <= 4
    
    def apply_filters(self, numbers, include_numbers=None):
        """필터 적용"""
        # 포함 번호가 있으면 반드시 포함되어야 함
        if include_numbers:
            if not all(n in numbers for n in include_numbers):
                return False
        
        # A. 연속번호 제거
        if self.filters['remove_consecutive']:
            if self.has_consecutive(numbers, self.filters['consecutive_level']):
                return False
        
        # B. 전부 짝수 제거
        if self.filters['remove_all_even']:
            if self.is_all_even(numbers):
                return False
        
        # C. 전부 홀수 제거
        if self.filters['remove_all_odd']:
            if self.is_all_odd(numbers):
                return False
        
        # D. 구간 집중 제거
        if self.filters['remove_range_cluster']:
            if self.is_range_clustered(numbers):
                return False
        
        # E. 40대 몰림 제거
        if self.filters['remove_high_40s']:
            if self.has_high_40s(numbers):
                return False
        
        # F. 홀짝 밸런스
        if self.filters['balance_odd_even']:
            if not self.check_odd_even_balance(numbers):
                return False
        
        # G. 최근 10회 번호 제외
        if self.filters['exclude_recent_10']:
            recent_numbers = self.get_recent_numbers(10)
            if any(n in recent_numbers for n in numbers):
                return False
        
        return True
    
    def generate_numbers(self, count=5, include_numbers=None, max_attempts=10000):
        """번호 추천 생성"""
        recommendations = []
        attempts = 0
        
        # 포함 번호 검증
        if include_numbers:
            include_numbers = [int(n) for n in include_numbers if 1 <= int(n) <= 45]
            if len(include_numbers) > 6:
                raise ValueError("포함 번호는 최대 6개까지 가능합니다.")
        else:
            include_numbers = []
        
        while len(recommendations) < count and attempts < max_attempts:
            attempts += 1
            
            # 포함 번호가 있으면 나머지만 랜덤 선택
            if include_numbers:
                remaining_count = 6 - len(include_numbers)
                available_numbers = [n for n in range(1, 46) if n not in include_numbers]
                remaining_numbers = random.sample(available_numbers, remaining_count)
                numbers = sorted(include_numbers + remaining_numbers)
            else:
                numbers = sorted(random.sample(range(1, 46), 6))
            
            # 필터 적용
            if self.apply_filters(numbers, include_numbers):
                recommendations.append(numbers)
        
        if len(recommendations) < count:
            print(f"⚠️ 필터 조건이 너무 엄격합니다. {len(recommendations)}개만 생성되었습니다.")
        
        return recommendations
    
    def calculate_filter_impact(self, sample_size=10000, include_numbers=None):
        """필터 영향도 계산 (시뮬레이션)"""
        passed = 0

        if include_numbers:
            include_numbers = [int(n) for n in include_numbers if 1 <= int(n) <= 45]
            include_numbers = sorted(set(include_numbers))
            if len(include_numbers) > 6:
                raise ValueError("포함 번호는 최대 6개까지 가능합니다.")
        else:
            include_numbers = []

        for _ in range(sample_size):
            if include_numbers:
                remaining_count = 6 - len(include_numbers)
                available_numbers = [n for n in range(1, 46) if n not in include_numbers]
                sampled = random.sample(available_numbers, remaining_count)
                numbers = sorted(include_numbers + sampled)
            else:
                numbers = random.sample(range(1, 46), 6)

            if self.apply_filters(numbers, include_numbers):
                passed += 1
        
        pass_rate = passed / sample_size * 100
        rejection_rate = 100 - pass_rate
        
        return {
            'pass_rate': round(pass_rate, 2),
            'rejection_rate': round(rejection_rate, 2),
            'passed': passed,
            'total': sample_size
        }
    
    def get_active_filters(self):
        """활성화된 필터 목록"""
        active = []
        
        filter_names = {
            'remove_consecutive': f'연속번호 제거 ({self.filters["consecutive_level"]}개 이상)',
            'remove_all_even': '전부 짝수 제거',
            'remove_all_odd': '전부 홀수 제거',
            'remove_range_cluster': '구간 집중 제거',
            'remove_high_40s': '40대 몰림 제거',
            'balance_odd_even': '홀짝 밸런스 (2:4~4:2)',
            'exclude_recent_10': '최근 10회 번호 제외'
        }
        
        for key, name in filter_names.items():
            if self.filters.get(key, False):
                active.append(name)
        
        return active


if __name__ == "__main__":
    recommender = LottoRecommender()
    
    # 필터 설정 예제
    recommender.set_filters(
        remove_consecutive=True,
        consecutive_level=2,
        balance_odd_even=True
    )
    
    # 번호 생성
    numbers = recommender.generate_numbers(count=5, include_numbers=[7, 27])
    
    print("🎲 추천 번호:")
    for i, nums in enumerate(numbers, 1):
        print(f"  [{i}] {nums}")
    
    # 필터 영향도
    impact = recommender.calculate_filter_impact()
    print(f"\n📊 필터 영향: {impact['rejection_rate']}% 조합 제외됨")
