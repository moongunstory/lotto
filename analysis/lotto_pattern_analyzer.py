"""
로또 6/45 패턴 분석 모듈
번호별 출현 빈도, 홀짝 비율, 구간 분포 등 통계 분석
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path


class LottoPatternAnalyzer:
    """로또 패턴 분석 클래스"""
    
    def __init__(self, data_path='data/lotto_history.csv'):
        self.data_path = Path(data_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        """데이터 로드"""
        if self.data_path.exists():
            self.df = pd.read_csv(self.data_path)
        else:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
    
    def get_number_frequency(self):
        """번호별 출현 빈도 계산"""
        all_numbers = []
        for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
            all_numbers.extend(self.df[col].tolist())
        
        frequency = Counter(all_numbers)
        freq_df = pd.DataFrame(list(frequency.items()), columns=['번호', '출현횟수'])
        freq_df = freq_df.sort_values('번호')
        freq_df['출현율(%)'] = (freq_df['출현횟수'] / len(self.df) * 100).round(2)
        
        return freq_df
    
    def get_recent_frequency(self, recent_draws=30):
        """최근 N회차 출현 빈도"""
        recent_df = self.df.tail(recent_draws)
        all_numbers = []
        for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
            all_numbers.extend(recent_df[col].tolist())
        
        frequency = Counter(all_numbers)
        return dict(frequency)
    
    def get_odd_even_ratio(self):
        """홀짝 비율 분석"""
        odd_even_counts = []
        
        for _, row in self.df.iterrows():
            numbers = [row[f'n{i}'] for i in range(1, 7)]
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            odd_even_counts.append(odd_count)
        
        ratio_df = pd.DataFrame(odd_even_counts, columns=['홀수개수'])
        ratio_df['짝수개수'] = 6 - ratio_df['홀수개수']
        
        distribution = ratio_df['홀수개수'].value_counts().sort_index()
        
        return {
            'distribution': distribution,
            'most_common': distribution.idxmax(),
            'percentage': (distribution / len(ratio_df) * 100).round(2).to_dict()
        }
    
    def get_range_distribution(self):
        """번호 구간 분포 분석"""
        ranges = {
            '1-10': 0,
            '11-20': 0,
            '21-30': 0,
            '31-40': 0,
            '41-45': 0
        }
        
        for _, row in self.df.iterrows():
            numbers = [row[f'n{i}'] for i in range(1, 7)]
            for num in numbers:
                if 1 <= num <= 10:
                    ranges['1-10'] += 1
                elif 11 <= num <= 20:
                    ranges['11-20'] += 1
                elif 21 <= num <= 30:
                    ranges['21-30'] += 1
                elif 31 <= num <= 40:
                    ranges['31-40'] += 1
                elif 41 <= num <= 45:
                    ranges['41-45'] += 1
        
        total = sum(ranges.values())
        percentages = {k: round(v/total*100, 2) for k, v in ranges.items()}
        
        return {'counts': ranges, 'percentages': percentages}
    
    def get_consecutive_distribution(self):
        """연속번호 개수 분포 분석"""
        consecutive_counts = []
        
        for _, row in self.df.iterrows():
            numbers = sorted([row[f'n{i}'] for i in range(1, 7)])
            consecutive = 0
            
            for i in range(len(numbers) - 1):
                if numbers[i+1] - numbers[i] == 1:
                    consecutive += 1
            
            consecutive_counts.append(consecutive)
        
        distribution = pd.Series(consecutive_counts).value_counts().sort_index()
        
        return {
            'distribution': distribution,
            'percentage': (distribution / len(consecutive_counts) * 100).round(2).to_dict()
        }
    
    def analyze_number_set(self, numbers):
        """특정 번호 조합 분석"""
        numbers = sorted(numbers)
        
        analysis = {
            'numbers': numbers,
            'odd_count': sum(1 for n in numbers if n % 2 == 1),
            'even_count': sum(1 for n in numbers if n % 2 == 0),
            'range_spread': max(numbers) - min(numbers),
            'consecutive_count': 0,
            'range_distribution': {}
        }
        
        # 연속번호 개수
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] == 1:
                analysis['consecutive_count'] += 1
        
        # 구간 분포
        for num in numbers:
            if 1 <= num <= 10:
                analysis['range_distribution']['1-10'] = analysis['range_distribution'].get('1-10', 0) + 1
            elif 11 <= num <= 20:
                analysis['range_distribution']['11-20'] = analysis['range_distribution'].get('11-20', 0) + 1
            elif 21 <= num <= 30:
                analysis['range_distribution']['21-30'] = analysis['range_distribution'].get('21-30', 0) + 1
            elif 31 <= num <= 40:
                analysis['range_distribution']['31-40'] = analysis['range_distribution'].get('31-40', 0) + 1
            elif 41 <= num <= 45:
                analysis['range_distribution']['41-45'] = analysis['range_distribution'].get('41-45', 0) + 1
        
        return analysis
    
    def get_statistics_summary(self):
        """전체 통계 요약"""
        return {
            'total_draws': len(self.df),
            'latest_draw': int(self.df['draw_no'].max()),
            'number_frequency': self.get_number_frequency(),
            'odd_even_ratio': self.get_odd_even_ratio(),
            'range_distribution': self.get_range_distribution(),
            'consecutive_distribution': self.get_consecutive_distribution()
        }


if __name__ == "__main__":
    analyzer = LottoPatternAnalyzer()
    stats = analyzer.get_statistics_summary()
    
    print(f"📊 총 {stats['total_draws']}회차 분석")
    print(f"🎯 최신 회차: {stats['latest_draw']}회")
    print("\n📈 홀짝 비율:")
    print(stats['odd_even_ratio']['percentage'])
