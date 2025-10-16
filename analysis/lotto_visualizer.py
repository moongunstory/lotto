"""
로또 6/45 시각화 모듈
matplotlib 기반 그래프 및 차트 생성
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from pathlib import Path
import platform

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    elif system == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    else:  # Linux
        plt.rc('font', family='DejaVu Sans')
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


class LottoVisualizer:
    """로또 시각화 클래스"""
    
    def __init__(self):
        set_korean_font()
        self.colors = {
            'primary': '#4A90E2',
            'secondary': '#7B68EE',
            'success': '#50C878',
            'warning': '#FFD700',
            'danger': '#FF6B6B'
        }
    
    def plot_number_frequency(self, frequency_df, highlight_numbers=None, save_path=None):
        """번호별 출현 빈도 막대그래프"""
        fig, ax = plt.subplots(figsize=(16, 6))
        
        numbers = frequency_df['번호'].values
        counts = frequency_df['출현횟수'].values
        
        # 색상 설정
        colors = [self.colors['warning'] if highlight_numbers and n in highlight_numbers 
                  else self.colors['primary'] for n in numbers]
        
        bars = ax.bar(numbers, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 평균선 추가
        mean_count = counts.mean()
        ax.axhline(y=mean_count, color='red', linestyle='--', linewidth=2, 
                   label=f'평균: {mean_count:.1f}회')
        
        ax.set_xlabel('번호', fontsize=12, fontweight='bold')
        ax.set_ylabel('출현 횟수', fontsize=12, fontweight='bold')
        ax.set_title('📊 로또 번호별 전체 출현 빈도', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(numbers)
        ax.set_xticklabels(numbers, fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        # 하이라이트 표시
        if highlight_numbers:
            for i, n in enumerate(numbers):
                if n in highlight_numbers:
                    ax.text(n, counts[i] + 2, '★', ha='center', fontsize=16, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_recent_trend(self, analyzer, recent_draws=30, highlight_numbers=None, save_path=None):
        """최근 출현 추세 그래프"""
        recent_freq = analyzer.get_recent_frequency(recent_draws)
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        numbers = sorted(recent_freq.keys())
        counts = [recent_freq.get(n, 0) for n in numbers]
        
        colors = [self.colors['warning'] if highlight_numbers and n in highlight_numbers 
                  else self.colors['secondary'] for n in numbers]
        
        ax.bar(numbers, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('번호', fontsize=12, fontweight='bold')
        ax.set_ylabel('출현 횟수', fontsize=12, fontweight='bold')
        ax.set_title(f'📈 최근 {recent_draws}회차 출현 추세', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(range(1, 46))
        ax.set_xticklabels(range(1, 46), fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_odd_even_distribution(self, odd_even_data, save_path=None):
        """홀짝 비율 분포 파이차트"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        distribution = odd_even_data['distribution']
        labels = [f'홀{i}:짝{6-i}' for i in distribution.index]
        sizes = distribution.values
        
        colors_list = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90', '#87CEEB', '#9370DB', '#BA55D3']
        explode = [0.05 if i == odd_even_data['most_common'] else 0 for i in distribution.index]
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            colors=colors_list[:len(sizes)],
            explode=explode,
            startangle=90,
            textprops={'fontsize': 11}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('🎯 홀짝 비율 분포', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_range_distribution(self, range_data, save_path=None):
        """구간 분포 막대그래프"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ranges = list(range_data['percentages'].keys())
        percentages = list(range_data['percentages'].values())
        
        colors_list = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90', '#87CEEB']
        
        bars = ax.bar(ranges, percentages, color=colors_list, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('번호 구간', fontsize=12, fontweight='bold')
        ax.set_ylabel('출현 비율 (%)', fontsize=12, fontweight='bold')
        ax.set_title('📊 번호 구간별 출현 분포', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_filter_impact(self, impact_data, save_path=None):
        """필터 영향도 파이차트"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = ['통과', '제외됨']
        sizes = [impact_data['pass_rate'], impact_data['rejection_rate']]
        colors = [self.colors['success'], self.colors['danger']]
        explode = (0, 0.1)
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode,
            startangle=90,
            textprops={'fontsize': 13}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)
        
        ax.set_title(f'🎯 필터 적용 영향도\n(총 {impact_data["total"]:,}개 조합 테스트)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_investment_simulation(self, monthly_cost, expected_return, improved_return, save_path=None):
        """투자 시뮬레이션 그래프"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 월별 투자 및 수익
        months = ['1월', '2월', '3월', '4월', '5월', '6월']
        costs = [monthly_cost] * len(months)
        returns_basic = [expected_return] * len(months)
        returns_improved = [improved_return] * len(months)
        
        x = np.arange(len(months))
        width = 0.25
        
        ax1.bar(x - width, costs, width, label='투자액', color='#FF6B6B', alpha=0.8)
        ax1.bar(x, returns_basic, width, label='기본 기대수익', color='#87CEEB', alpha=0.8)
        ax1.bar(x + width, returns_improved, width, label='필터 적용 수익', color='#50C878', alpha=0.8)
        
        ax1.set_xlabel('월', fontsize=11, fontweight='bold')
        ax1.set_ylabel('금액 (원)', fontsize=11, fontweight='bold')
        ax1.set_title('💰 월별 투자 시뮬레이션', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(months)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 누적 손익
        cumulative_cost = np.cumsum(costs)
        cumulative_basic = np.cumsum(returns_basic)
        cumulative_improved = np.cumsum(returns_improved)
        
        ax2.plot(months, cumulative_cost, 'o-', label='누적 투자', 
                color='#FF6B6B', linewidth=2, markersize=8)
        ax2.plot(months, cumulative_basic, 's-', label='기본 누적수익', 
                color='#87CEEB', linewidth=2, markersize=8)
        ax2.plot(months, cumulative_improved, '^-', label='개선 누적수익', 
                color='#50C878', linewidth=2, markersize=8)
        
        ax2.set_xlabel('월', fontsize=11, fontweight='bold')
        ax2.set_ylabel('누적 금액 (원)', fontsize=11, fontweight='bold')
        ax2.set_title('📈 누적 손익 추이', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    print("시각화 모듈 로드 완료")
