"""
로또 6/45 ML 시각화 모듈
ML 예측 결과 및 모델 성능 시각화
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import platform


# 한글 폰트 설정
def set_korean_font():
    """한글 폰트 및 이모지 지원 설정"""
    system = platform.system()

    if system == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    elif system == 'Windows':
        # 이모지를 지원하는 폰트를 폴백으로 추가
        plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI Emoji']
    else:  # Linux
        # Linux 환경에서는 noto-color-emoji 폰트 설치가 필요할 수 있습니다.
        # sudo apt-get install fonts-noto-color-emoji
        plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans', 'Noto Color Emoji']

    plt.rcParams['axes.unicode_minus'] = False
    # 일부 환경(특히 Windows HiDPI)에서는 Matplotlib이 매우 큰 DPI를 사용해
    # Streamlit에서 이미지를 변환할 때 DecompressionBombError가 발생할 수 있다.
    # 기본 Figure/Save DPI를 안전한 값으로 제한한다.
    safe_dpi = 120
    plt.rcParams['figure.dpi'] = safe_dpi
    plt.rcParams['savefig.dpi'] = safe_dpi


class LottoMLVisualizer:
    """로또 ML 시각화 클래스"""
    
    def __init__(self):
        set_korean_font()
        self.default_dpi = 120
        self.default_figsize = (14, 7)
        self.colors = {
            'primary': '#4A90E2',
            'secondary': '#7B68EE',
            'success': '#50C878',
            'warning': '#FFD700',
            'danger': '#FF6B6B',
            'gradient': ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        }

    def _create_figure(self, figsize):
        """안전한 DPI가 적용된 Figure 생성"""
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(self.default_dpi)
        return fig, ax
    
    def plot_number_probabilities(self, probabilities, top_k=20, highlight_numbers=None, save_path=None):
        """
        번호별 확률 막대그래프
        
        Args:
            probabilities: {번호: 확률} 딕셔너리
            top_k: 상위 K개만 표시
            highlight_numbers: 강조할 번호 리스트
            save_path: 저장 경로
        """
        fig, ax = self._create_figure(self.default_figsize)
        
        # 확률 순으로 정렬
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        numbers = [x[0] for x in sorted_probs]
        probs = [x[1] * 100 for x in sorted_probs]  # 퍼센트로 변환
        
        # 색상 설정
        colors = []
        for num in numbers:
            if highlight_numbers and num in highlight_numbers:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['primary'])
        
        # 막대그래프
        bars = ax.bar(range(len(numbers)), probs, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # 평균선
        mean_prob = np.mean(probs)
        ax.axhline(y=mean_prob, color='red', linestyle='--', linewidth=2,
                  label=f'평균: {mean_prob:.2f}%')
        
        # 레이블
        ax.set_xlabel('번호', fontsize=14, fontweight='bold')
        ax.set_ylabel('출현 확률 (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'🤖 AI 예측: 다음 회차 번호별 출현 확률 (상위 {top_k}개)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(range(len(numbers)))
        ax.set_xticklabels(numbers, fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=12)
        
        # 값 표시
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 순위 표시
        for i, num in enumerate(numbers[:3]):
            medals = ['🥇', '🥈', '🥉']
            ax.text(i, probs[i] + 1, medals[i], ha='center', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        
        return fig
    
    def plot_probability_heatmap(self, probabilities, save_path=None):
        """
        45개 번호를 5x9 히트맵으로 표시
        
        Args:
            probabilities: {번호: 확률} 딕셔너리
            save_path: 저장 경로
        """
        fig, ax = self._create_figure((14, 8))
        
        # 5x9 그리드 생성
        grid = np.zeros((5, 9))
        for i in range(1, 46):
            row = (i - 1) // 9
            col = (i - 1) % 9
            grid[row, col] = probabilities.get(i, 0) * 100
        
        # 히트맵
        sns.heatmap(grid, annot=False, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': '확률 (%)'}, ax=ax,
                   linewidths=1, linecolor='white')
        
        # 번호 표시
        for i in range(1, 46):
            row = (i - 1) // 9
            col = (i - 1) % 9
            prob = probabilities.get(i, 0) * 100
            color = 'white' if prob > 15 else 'black'
            ax.text(col + 0.5, row + 0.5, f'{i}\n{prob:.1f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color=color)
        
        ax.set_title('🔥 번호별 확률 히트맵', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        
        return fig
    
    def plot_combo_scores(self, combos_with_scores, top_k=10, save_path=None):
        """
        조합별 신뢰도 막대그래프
        
        Args:
            combos_with_scores: [([번호들], 점수), ...] 리스트
            top_k: 상위 K개만 표시
            save_path: 저장 경로
        """
        fig, ax = self._create_figure((14, 10))
        
        # 상위 K개
        top_combos = combos_with_scores[:top_k]
        
        labels = [f"[{', '.join(map(str, combo))}]" for combo, _ in top_combos]
        scores = [score * 100 for _, score in top_combos]  # 퍼센트로
        
        # 색상 그라데이션
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(scores)))
        
        # 수평 막대그래프
        bars = ax.barh(range(len(labels)), scores, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('신뢰도 점수 (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'🎯 AI 추천 조합 TOP {top_k}', fontsize=18, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 값 표시
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                   f'{score:.2f}%',
                   ha='left', va='center', fontsize=11, fontweight='bold')
        
        # 순위 표시
        medals = ['🥇', '🥈', '🥉']
        for i in range(min(3, len(labels))):
            ax.text(-3, i, medals[i], ha='center', va='center', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_importance, top_k=20, save_path=None):
        """
        피처 중요도 수평 막대그래프
        
        Args:
            feature_importance: [(피처명, 중요도), ...] 리스트
            top_k: 상위 K개만 표시
            save_path: 저장 경로
        """
        fig, ax = self._create_figure((12, 10))
        
        # 상위 K개
        top_features = feature_importance[:top_k]
        
        features = [f for f, _ in top_features]
        importances = [imp * 100 for _, imp in top_features]  # 퍼센트로
        
        # 색상 그라데이션
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(importances)))
        
        # 수평 막대그래프
        bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=11)
        ax.set_xlabel('중요도 (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'🔍 피처 중요도 분석 (상위 {top_k}개)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 값 표시
        for bar, imp in zip(bars, importances):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                   f'{imp:.2f}%',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        
        return fig
    
    def plot_backtest_results(self, backtest_data, model_type='number', save_path=None):
        """
        백테스트 결과 시각화
        
        Args:
            backtest_data: 백테스트 결과 딕셔너리
            model_type: 'number' or 'combo'
            save_path: 저장 경로
        """
        if model_type == 'number':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.set_dpi(self.default_dpi)
            
            # 상위 K개 선택시 평균 적중 개수
            top_k_perf = backtest_data.get('top_k_performance', {})
            
            if top_k_perf:
                ks = list(top_k_perf.keys())
                hits = list(top_k_perf.values())
                
                bars = ax1.bar(range(len(ks)), hits, color=self.colors['primary'], 
                              alpha=0.8, edgecolor='black', linewidth=1)
                ax1.set_xticks(range(len(ks)))
                ax1.set_xticklabels([f'상위\n{k}개' for k in ks], fontsize=11)
                ax1.set_ylabel('평균 적중 개수', fontsize=12, fontweight='bold')
                ax1.set_title('📊 상위 K개 선택시 평균 적중', fontsize=14, fontweight='bold')
                ax1.grid(axis='y', alpha=0.3)
                
                for bar, hit in zip(bars, hits):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{hit:.2f}',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 평가 지표
            metrics = ['정확도', '정밀도', '재현율', 'F1', 'AUC']
            values = [
                backtest_data.get('accuracy', 0),
                backtest_data.get('precision', 0),
                backtest_data.get('recall', 0),
                backtest_data.get('f1', 0),
                backtest_data.get('auc', 0)
            ]
            
            colors_list = [self.colors['success'], self.colors['primary'], 
                          self.colors['warning'], self.colors['secondary'], 
                          self.colors['danger']]
            
            bars = ax2.bar(range(len(metrics)), values, color=colors_list, 
                          alpha=0.8, edgecolor='black', linewidth=1)
            ax2.set_xticks(range(len(metrics)))
            ax2.set_xticklabels(metrics, fontsize=11)
            ax2.set_ylabel('점수', fontsize=12, fontweight='bold')
            ax2.set_title('📈 모델 평가 지표', fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1.1)
            ax2.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        elif model_type == 'combo':
            fig, ax = self._create_figure((12, 6))
            
            # 일치 개수 분포
            match_counts = backtest_data.get('match_counts', {})
            
            if match_counts:
                matches = list(match_counts.keys())
                counts = list(match_counts.values())
                
                colors_list = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90', 
                              '#87CEEB', '#9370DB', '#50C878']
                
                bars = ax.bar(matches, counts, color=colors_list[:len(matches)], 
                             alpha=0.8, edgecolor='black', linewidth=1.5)
                ax.set_xlabel('일치 개수', fontsize=12, fontweight='bold')
                ax.set_ylabel('회차 수', fontsize=12, fontweight='bold')
                ax.set_title('🎯 예측 조합 일치 개수 분포', fontsize=16, fontweight='bold', pad=20)
                ax.set_xticks(matches)
                ax.set_xticklabels([f'{m}개' for m in matches], fontsize=11)
                ax.grid(axis='y', alpha=0.3)
                
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}회',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_vs_actual(self, predicted_numbers, actual_numbers, save_path=None):
        """
        예측 번호 vs 실제 당첨번호 비교 차트
        
        Args:
            predicted_numbers: 예측 번호 리스트
            actual_numbers: 실제 당첨번호 리스트
            save_path: 저장 경로
        """
        fig, ax = self._create_figure((14, 6))
        
        # Venn 다이어그램 스타일 표현
        all_numbers = sorted(set(predicted_numbers) | set(actual_numbers))
        
        y_pred = []
        y_actual = []
        colors = []
        
        for num in all_numbers:
            in_pred = num in predicted_numbers
            in_actual = num in actual_numbers
            
            if in_pred and in_actual:
                y_pred.append(1)
                y_actual.append(1)
                colors.append(self.colors['success'])  # 둘 다 있음
            elif in_pred:
                y_pred.append(1)
                y_actual.append(0)
                colors.append(self.colors['primary'])  # 예측만
            else:
                y_pred.append(0)
                y_actual.append(1)
                colors.append(self.colors['danger'])  # 실제만
        
        x = np.arange(len(all_numbers))
        width = 0.35
        
        ax.bar(x - width/2, y_pred, width, label='예측 번호', 
               color=self.colors['primary'], alpha=0.7)
        ax.bar(x + width/2, y_actual, width, label='실제 당첨번호', 
               color=self.colors['danger'], alpha=0.7)
        
        ax.set_xlabel('번호', fontsize=12, fontweight='bold')
        ax.set_ylabel('포함 여부', fontsize=12, fontweight='bold')
        ax.set_title('🎯 예측 번호 vs 실제 당첨번호', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(all_numbers, fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['미포함', '포함'], fontsize=11)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # 일치 번호 강조
        matched = set(predicted_numbers) & set(actual_numbers)
        if matched:
            match_text = ', '.join(map(str, sorted(matched)))
            ax.text(0.5, 0.95, f'✅ 일치: {match_text} ({len(matched)}개)',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=13, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        
        return fig
    
    def plot_probability_trend(self, number, history_data, save_path=None):
        """
        특정 번호의 확률 추세 그래프
        
        Args:
            number: 번호
            history_data: [(회차, 확률), ...] 리스트
            save_path: 저장 경로
        """
        fig, ax = self._create_figure((14, 6))
        
        draws = [d for d, _ in history_data]
        probs = [p * 100 for _, p in history_data]
        
        ax.plot(draws, probs, marker='o', linewidth=2, markersize=6,
               color=self.colors['primary'], label=f'{number}번')
        
        # 평균선
        mean_prob = np.mean(probs)
        ax.axhline(y=mean_prob, color='red', linestyle='--', linewidth=2,
                  label=f'평균: {mean_prob:.2f}%')
        
        ax.set_xlabel('회차', fontsize=12, fontweight='bold')
        ax.set_ylabel('출현 확률 (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'📈 {number}번 출현 확률 추세', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    print("ML 시각화 모듈 로드 완료")
    
    # 테스트 데이터
    test_probabilities = {i: np.random.random() * 0.3 + 0.05 for i in range(1, 46)}
    test_combos = [
        ([7, 12, 27, 31, 38, 42], 0.876),
        ([3, 19, 27, 33, 41, 44], 0.834),
        ([8, 15, 23, 29, 37, 43], 0.801),
        ([5, 11, 18, 25, 32, 39], 0.789),
        ([2, 14, 21, 28, 35, 40], 0.776)
    ]
    
    visualizer = LottoMLVisualizer()
    
    # 테스트 시각화
    print("\n📊 테스트 시각화 생성 중...")
    
    fig1 = visualizer.plot_number_probabilities(test_probabilities, top_k=20)
    fig1.savefig('/tmp/test_number_probs.png', dpi=visualizer.default_dpi, bbox_inches='tight')
    plt.close()
    print("✅ 번호 확률 그래프 생성")

    fig2 = visualizer.plot_combo_scores(test_combos, top_k=5)
    fig2.savefig('/tmp/test_combo_scores.png', dpi=visualizer.default_dpi, bbox_inches='tight')
    plt.close()
    print("✅ 조합 점수 그래프 생성")

    fig3 = visualizer.plot_probability_heatmap(test_probabilities)
    fig3.savefig('/tmp/test_heatmap.png', dpi=visualizer.default_dpi, bbox_inches='tight')
    plt.close()
    print("✅ 확률 히트맵 생성")
    
    print("\n✅ 모든 테스트 완료!")
