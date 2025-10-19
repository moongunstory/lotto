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
            'odd_even_balance': [],          # e.g., ["4:2", "3:3"]
            'exclude_recent_draws': 0,      # 1 ~ 1000
            'exclude_consecutive_lengths': [], # e.g., [3, 4] (3연속, 4연속 조합 제외)
            'range_limits': {},             # e.g., {'0': 3, '1': 4, '2': 4, '3': 4, '4': 3} (0번대 3개, 10번대 4개...)
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
        if recent_draws <= 0:
            return set()
        recent_df = self.df.tail(recent_draws)
        recent_numbers = set()
        
        for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
            recent_numbers.update(recent_df[col].tolist())
        
        return recent_numbers

    def _check_consecutive_rules(self, numbers):
        """
        연속 번호 규칙 검사 (신규 로직)
        - exclude_consecutive_lengths: 제외할 연속 번호의 길이 목록
        """
        exclude_lengths = self.filters.get('exclude_consecutive_lengths', [])
        if not exclude_lengths:
            return True

        numbers = sorted(numbers)
        if not numbers:
            return True
        
        current_streak = 1
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] == 1:
                current_streak += 1
            else:
                if current_streak in exclude_lengths:
                    return False # 제외해야 할 길이를 발견
                current_streak = 1
        
        # 마지막 스트릭 확인
        if current_streak in exclude_lengths:
            return False
            
        return True

    def _check_range_limits(self, numbers):
        """
        구간별 번호 개수 제한 규칙 검사 (신규 로직)
        - range_limits: 구간별 최대 허용 개수 딕셔너리
        """
        limits = self.filters.get('range_limits', {})
        if not limits:
            return True

        range_counts = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
        for n in numbers:
            if 1 <= n <= 9: range_counts['0'] += 1
            elif 10 <= n <= 19: range_counts['1'] += 1
            elif 20 <= n <= 29: range_counts['2'] += 1
            elif 30 <= n <= 39: range_counts['3'] += 1
            elif 40 <= n <= 45: range_counts['4'] += 1

        for range_key, max_count in limits.items():
            if range_counts[range_key] > max_count:
                return False # 제한 개수 초과
        
        return True

    def _check_odd_even_balance(self, numbers):
        """홀짝 밸런스 체크 (다중 선택 가능)"""
        balance_ratios = self.filters.get('odd_even_balance', [])
        if not balance_ratios:
            return True  # 필터 미적용

        odd_count = sum(1 for n in numbers if n % 2 == 1)

        for ratio in balance_ratios:
            try:
                target_odd, _ = map(int, ratio.split(':'))
                if odd_count == target_odd:
                    return True # 한 개라도 맞으면 통과
            except (ValueError, AttributeError):
                continue # 포맷 오류 무시
       
        return False # 맞는 비율이 하나도 없음

    def apply_filters(self, numbers, include_numbers=None):
        """필터 적용 (신규 로직 통합)"""
        if include_numbers:
            if not all(n in numbers for n in include_numbers):
                return False

        # 1. 홀짝 밸런스
        if not self._check_odd_even_balance(numbers):
            return False

        # 2. 최근 당첨번호 제외
        if self.filters['exclude_recent_draws'] > 0:
            recent_numbers = self.get_recent_numbers(self.filters['exclude_recent_draws'])
            if any(n in recent_numbers for n in numbers):
                return False

        # 3. 연속 번호 규칙 (신규)
        if not self._check_consecutive_rules(numbers):
            return False

        # 4. 구간별 개수 제한 (신규)
        if not self._check_range_limits(numbers):
            return False

        return True

    def generate_numbers(self, count=5, include_numbers=None, max_attempts=10000):
        """필터를 만족하는 랜덤 번호 추천 생성"""
        recommendations = []
        attempts = 0
        
        if include_numbers:
            include_numbers = [int(n) for n in include_numbers if 1 <= int(n) <= 45]
            if len(include_numbers) > 6:
                raise ValueError("포함 번호는 최대 6개까지 가능합니다.")
        else:
            include_numbers = []
        
        while len(recommendations) < count and attempts < max_attempts:
            attempts += 1
            
            if include_numbers:
                remaining_count = 6 - len(include_numbers)
                available_numbers = [n for n in range(1, 46) if n not in include_numbers]
                remaining_numbers = random.sample(available_numbers, remaining_count)
                numbers = sorted(include_numbers + remaining_numbers)
            else:
                numbers = sorted(random.sample(range(1, 46), 6))
            
            if self.apply_filters(numbers, include_numbers):
                recommendations.append(numbers)
        
        if len(recommendations) < count:
            print(f"⚠️ 필터 조건이 너무 엄격합니다. {len(recommendations)}개만 생성되었습니다.")
        
        return recommendations

    def get_active_filters(self):
        """활성화된 필터 목록"""
        active = []

        # 홀짝
        odd_even_ratios = self.filters.get('odd_even_balance', [])
        if odd_even_ratios:
            active.append(f"홀짝 밸런스 ({ ', '.join(odd_even_ratios) })")

        # 최근 번호 제외
        exclude_draws = self.filters.get('exclude_recent_draws', 0)
        if exclude_draws > 0:
            active.append(f'최근 {exclude_draws}회 번호 제외')

        # 연속 번호
        exclude_consecutive = self.filters.get('exclude_consecutive_lengths', [])
        if exclude_consecutive:
            active.append(f"연속 번호 제외 ({ ', '.join(map(str, exclude_consecutive))}개 짜리)")

        # 구간 제한
        range_limits = self.filters.get('range_limits', {})
        range_filters = []
        range_map = {'0': '1-9', '1': '10', '2': '20', '3': '30', '4': '40'}
        for key, limit in range_limits.items():
            if limit < 6:
                range_name = range_map[key] + "번대"
                range_filters.append(f"{range_name} 최대 {limit}개")
        if range_filters:
            active.append(", ".join(range_filters))

        return active


if __name__ == "__main__":
    recommender = LottoRecommender()
    
    # 필터 설정 예제
    recommender.set_filters(
        odd_even_balance=['4:2', '3:3', '2:4'],
        exclude_consecutive_lengths=[3, 4], # 3연속, 4연속 제외
        range_limits={'1': 3, '3': 3} # 10번대, 30번대 최대 3개까지
    )
    
    # 번호 생성
    numbers = recommender.generate_numbers(count=5)
    
    print("🎲 추천 번호:")
    for i, nums in enumerate(numbers, 1):
        print(f"  [{i}] {nums}")
    
    print(f"\n🔧 적용된 필터: {recommender.get_active_filters()}")