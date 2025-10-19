"""
로또 6/45 조합 예측 모듈
6개 번호 조합을 직접 예측하고 스코어링
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import combinations
import xgboost as xgb


class LottoComboPredictor:
    """로또 조합 예측 클래스"""
    
    def __init__(self, model_type='xgboost'):
        """
        Args:
            model_type: 'gradient_boosting', 'random_forest', 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_version = None
        
    def _create_model(self):
        """모델 생성"""
        if self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"지원하지 않는 모델: {self.model_type}")
    
    def train(self, feature_engineer, start_draw=100, end_draw=None, 
              negative_samples=5, validation_split=0.2):
        """
        조합 스코어링 모델 학습
        
        Args:
            feature_engineer: LottoFeatureEngineer 인스턴스
            start_draw: 시작 회차
            end_draw: 종료 회차
            negative_samples: 당첨 조합당 생성할 음성 샘플 수
            validation_split: 검증 데이터 비율
        """
        print("\n" + "="*60)
        print(f"🎯 조합 예측 모델 학습 시작 ({self.model_type})")
        print("="*60)
        
        # 1. 학습 데이터 생성
        X, y, draws = feature_engineer.build_combo_training_data(
            start_draw=start_draw,
            end_draw=end_draw,
            negative_samples=negative_samples
        )
        
        self.feature_names = X.columns.tolist()
        
        # 2. Train/Validation 분할
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"\n📊 데이터 분할:")
        print(f"   - 학습: {len(X_train)}개 (당첨: {y_train.sum()}개)")
        print(f"   - 검증: {len(X_val)}개 (당첨: {y_val.sum()}개)")
        
        # 3. 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 4. 모델 학습
        print(f"\n🔧 모델 학습 중...")
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # 5. 평가
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_val = self.model.predict(X_val_scaled)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        val_mse = mean_squared_error(y_val, y_pred_val)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        
        print(f"\n📈 학습 결과:")
        print(f"   - 학습 MSE: {train_mse:.4f}")
        print(f"   - 검증 MSE: {val_mse:.4f}")
        print(f"   - 학습 MAE: {train_mae:.4f}")
        print(f"   - 검증 MAE: {val_mae:.4f}")
        print(f"   - 학습 R²: {train_r2:.4f}")
        print(f"   - 검증 R²: {val_r2:.4f}")
        
        # 6. 당첨 조합 vs 랜덤 조합 점수 비교
        winning_mask = y_val == 1
        random_mask = y_val == 0
        
        if winning_mask.sum() > 0 and random_mask.sum() > 0:
            avg_winning_score = y_pred_val[winning_mask].mean()
            avg_random_score = y_pred_val[random_mask].mean()
            
            print(f"\n🎯 조합 점수 비교:")
            print(f"   - 실제 당첨 조합 평균 점수: {avg_winning_score:.4f}")
            print(f"   - 랜덤 조합 평균 점수: {avg_random_score:.4f}")
            print(f"   - 점수 차이: {avg_winning_score - avg_random_score:.4f}")
        
        print("\n✅ 학습 완료!")

        self.feature_version = feature_engineer.get_feature_version()

        return {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2
        }
    
    def score_combination(self, feature_engineer, numbers, reference_draw=None):
        """
        특정 6개 조합의 점수 계산
        
        Args:
            feature_engineer: LottoFeatureEngineer 인스턴스
            numbers: 6개 번호 리스트
            reference_draw: 기준 회차 (None이면 최신)
            
        Returns:
            float: 0.0 ~ 1.0 점수 (높을수록 좋음)
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 실행하세요.")

        if reference_draw is None:
            reference_draw = feature_engineer.get_latest_draw_number() + 1

        if self.feature_version and self.feature_version != feature_engineer.get_feature_version():
            raise ValueError(
                "학습된 조합 모델의 피처 버전이 현재 데이터 스키마와 다릅니다. 모델을 다시 학습해주세요."
            )

        # 피처 추출
        features = feature_engineer.extract_combo_features(numbers, reference_draw)
        
        # DataFrame으로 변환
        X = pd.DataFrame([features])
        X = X[self.feature_names]
        
        # 스케일링
        X_scaled = self.scaler.transform(X)
        
        # 점수 예측
        score = self.model.predict(X_scaled)[0]
        
        # 0~1 범위로 클리핑
        score = np.clip(score, 0.0, 1.0)
        
        return float(score)
    
    def predict_top_combos(self, feature_engineer, n=10, candidate_pool='smart', 
                          pool_size=25, reference_draw=None):
        """
        상위 N개 조합 예측
        
        Args:
            feature_engineer: LottoFeatureEngineer 인스턴스
            n: 반환할 조합 개수
            candidate_pool: 
                - 'smart': NumberPredictor가 필요 (상위 번호로만 조합)
                - 'balanced': 확률 분포에 따라 샘플링
                - 'random': 완전 랜덤 조합
            pool_size: 후보 풀 크기 (smart 모드에서 사용)
            reference_draw: 기준 회차
            
        Returns:
            list of tuples: [([번호들], 점수), ...]
        """
        if reference_draw is None:
            reference_draw = feature_engineer.get_latest_draw_number() + 1

        if self.feature_version and self.feature_version != feature_engineer.get_feature_version():
            raise ValueError(
                "학습된 조합 모델의 피처 버전이 현재 데이터 스키마와 다릅니다. 모델을 다시 학습해주세요."
            )

        print(f"\n🎯 상위 {n}개 조합 예측 (모드: {candidate_pool})")
        
        scored_combos = []
        
        if candidate_pool == 'random':
            # 랜덤 조합 생성
            print(f"🔄 랜덤 조합 {n * 100}개 생성 및 평가 중...")
            for _ in range(n * 100):
                numbers = sorted(np.random.choice(range(1, 46), size=6, replace=False))
                score = self.score_combination(feature_engineer, numbers, reference_draw)
                scored_combos.append((numbers, score))
        
        elif candidate_pool == 'balanced':
            # 균등 분포 샘플링
            print(f"🔄 균등 샘플링 조합 {n * 100}개 생성 및 평가 중...")
            for _ in range(n * 100):
                numbers = sorted(np.random.choice(range(1, 46), size=6, replace=False))
                score = self.score_combination(feature_engineer, numbers, reference_draw)
                scored_combos.append((numbers, score))
        
        elif candidate_pool == 'smart':
            # 상위 pool_size개 번호로만 조합 생성
            print(f"🔄 상위 {pool_size}개 번호로 조합 생성 및 평가 중...")
            
            # 단순히 최근 빈도로 상위 번호 선택
            recent_freq = {}
            recent_df = feature_engineer.df.tail(50)
            for _, row in recent_df.iterrows():
                for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
                    num = int(row[col])
                    recent_freq[num] = recent_freq.get(num, 0) + 1
            
            top_numbers = sorted(recent_freq.items(), key=lambda x: x[1], reverse=True)
            top_numbers = [num for num, _ in top_numbers[:pool_size]]
            
            if len(top_numbers) < pool_size:
                # 부족하면 나머지 번호 추가
                remaining = [i for i in range(1, 46) if i not in top_numbers]
                top_numbers.extend(remaining[:pool_size - len(top_numbers)])
            
            print(f"   선택된 상위 번호: {sorted(top_numbers)}")
            
            # 조합 생성 (최대 5000개)
            all_combos = list(combinations(top_numbers, 6))
            sample_size = min(len(all_combos), 5000)

            if len(all_combos) > 0:
                sampled_indices = np.random.choice(len(all_combos), size=sample_size, replace=False)
                
                for idx in sampled_indices:
                    numbers = list(all_combos[idx])
                    score = self.score_combination(feature_engineer, numbers, reference_draw)
                    scored_combos.append((numbers, score))
        
        # 점수 기준 정렬
        scored_combos.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ 평가 완료! 상위 {n}개 반환")
        
        return scored_combos[:n]
    
    def generate_with_number_probs(self, feature_engineer, number_predictor, 
                                   n=100, reference_draw=None):
        """
        번호별 확률 기반으로 가중 샘플링하여 조합 생성
        
        Args:
            feature_engineer: LottoFeatureEngineer 인스턴스
            number_predictor: LottoNumberPredictor 인스턴스 (학습 완료된)
            n: 생성할 조합 개수
            reference_draw: 기준 회차
            
        Returns:
            list of tuples: [([번호들], 점수), ...]
        """
        if reference_draw is None:
            reference_draw = feature_engineer.get_latest_draw_number() + 1
        
        print(f"\n🤖 ML 확률 기반 조합 생성 ({n}개)")
        
        # 1. 번호별 확률 가져오기
        probabilities = number_predictor.predict_probabilities(feature_engineer, reference_draw)
        
        numbers = list(probabilities.keys())
        probs = list(probabilities.values())
        probs = np.array(probs)
        probs = probs / probs.sum()  # 정규화
        
        print(f"   상위 10개 확률:")
        top_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
        for num, prob in top_probs:
            print(f"   {num:2d}번: {prob*100:.2f}%")
        
        # 2. 확률에 비례하여 조합 샘플링
        scored_combos = []
        
        for _ in range(n):
            # 가중 샘플링 (복원 추출 후 중복 제거)
            sampled = np.random.choice(numbers, size=20, replace=True, p=probs)
            unique_numbers = list(set(sampled))
            
            # 6개가 안되면 추가 샘플링
            while len(unique_numbers) < 6:
                additional = np.random.choice(numbers, size=1, p=probs)[0]
                if additional not in unique_numbers:
                    unique_numbers.append(additional)
            
            combo = sorted(unique_numbers[:6])
            score = self.score_combination(feature_engineer, combo, reference_draw)
            scored_combos.append((combo, score))
        
        # 점수 기준 정렬
        scored_combos.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ 생성 완료!")
        
        return scored_combos
    
    def backtest(self, feature_engineer, test_draws=20):
        """
        백테스트: 예측한 조합들이 실제 당첨번호와 얼마나 겹치는지
        
        Args:
            feature_engineer: LottoFeatureEngineer 인스턴스
            test_draws: 테스트할 회차 수
            
        Returns:
            dict: 평가 지표
        """
        print("\n" + "="*60)
        print(f"🔬 조합 예측 백테스트 시작 (최근 {test_draws}회차)")
        print("="*60)
        
        latest_draw = feature_engineer.get_latest_draw_number()
        start_draw = latest_draw - test_draws + 1
        
        df = feature_engineer.df
        
        match_counts = {i: 0 for i in range(7)}  # 0~6개 일치
        top_scores = []
        
        for draw_no in range(start_draw, latest_draw + 1):
            # 실제 당첨번호
            actual_row = df[df['draw_no'] == draw_no]
            if actual_row.empty:
                continue
            
            actual_row = actual_row.iloc[0]
            actual_numbers = set([
                int(actual_row['n1']), int(actual_row['n2']), int(actual_row['n3']),
                int(actual_row['n4']), int(actual_row['n5']), int(actual_row['n6'])
            ])
            
            # 상위 10개 조합 예측
            predicted_combos = self.predict_top_combos(
                feature_engineer, 
                n=10, 
                candidate_pool='smart',
                pool_size=25,
                reference_draw=draw_no
            )
            
            # 최고 일치 개수 확인
            max_match = 0
            for combo, score in predicted_combos:
                match = len(set(combo) & actual_numbers)
                max_match = max(max_match, match)
                
                if score > 0.7:  # 고득점 조합 저장
                    top_scores.append((draw_no, combo, score, match))
            
            match_counts[max_match] += 1
            
            print(f"   {draw_no}회: 최대 {max_match}개 일치")
        
        # 통계
        print(f"\n📊 백테스트 결과 ({test_draws}회차):")
        for i in range(7):
            count = match_counts[i]
            percentage = count / test_draws * 100 if test_draws > 0 else 0
            print(f"   {i}개 일치: {count}회 ({percentage:.1f}%)")
        
        avg_match = sum(k * v for k, v in match_counts.items()) / test_draws
        print(f"\n   평균 일치 개수: {avg_match:.2f}개")
        
        return {
            'match_counts': match_counts,
            'avg_match': avg_match,
            'top_scores': top_scores
        }
    
    def save_model(self, path='models/combo_predictor.pkl'):
        """모델 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'feature_version': self.feature_version
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ 모델 저장 완료: {path}")
    
    def load_model(self, path='models/combo_predictor.pkl', expected_feature_version=None):
        """모델 로드"""
        if not Path(path).exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.feature_names = save_data['feature_names']
        self.model_type = save_data.get('model_type', 'unknown')
        self.feature_version = save_data.get('feature_version')

        if expected_feature_version and self.feature_version != expected_feature_version:
            raise ValueError(
                f"저장된 조합 모델 피처 버전({self.feature_version})과 현재 버전({expected_feature_version})이 다릅니다."
            )
        
        print(f"✅ 모델 로드 완료: {path}")


if __name__ == "__main__":
    from analysis.lotto_feature_engineer import LottoFeatureEngineer
    from .lotto_number_predictor import LottoNumberPredictor
    
    print("\n" + "="*60)
    print("🎯 Combo Predictor 테스트")
    print("="*60)
    
    # --- 1. Setup ---
    engineer = LottoFeatureEngineer('lotto/data/lotto_history.csv')
    latest_draw = engineer.get_latest_draw_number()
    
    combo_predictor = LottoComboPredictor(model_type='xgboost')
    
    train_end = latest_draw - 20
    train_start = max(100, train_end - 200)
    
    results = combo_predictor.train(
        engineer,
        start_draw=train_start,
        end_draw=train_end,
        negative_samples=5,
        validation_split=0.2
    )
    
    # --- 2. Prediction Method 1: Self-contained (Smart) ---
    print("\n" + "="*60)
    print(f"🎯 다음 회차 ({latest_draw + 1}회) 조합 예측 [1. 자체 방식 (Smart)]")
    print("="*60)
    
    top_combos_smart = combo_predictor.predict_top_combos(
        engineer,
        n=10,
        candidate_pool='smart',
        pool_size=25
    )
    
    print(f"\n🏆 상위 10개 조합 (자체 방식):")
    for i, (combo, score) in enumerate(top_combos_smart, 1):
        bar = "█" * int(score * 50)
        print(f"{i:2d}. {str(combo):<22} {bar}  {score:.4f}")

    # --- 3. Prediction Method 2: Integrated with Number Predictor ---
    print("\n" + "="*60)
    print(f"🎯 다음 회차 ({latest_draw + 1}회) 조합 예측 [2. 번호 예측 모델 연동]")
    print("="*60)

    try:
        number_predictor = LottoNumberPredictor()
        number_predictor.load_model('models/number_predictor.pkl', expected_feature_version=engineer.get_feature_version())

        top_combos_ml = combo_predictor.generate_with_number_probs(
            feature_engineer=engineer,
            number_predictor=number_predictor,
            n=2000,
            reference_draw=latest_draw + 1
        )

        print(f"\n🏆 상위 10개 조합 (모델 연동 방식):")
        for i, (combo, score) in enumerate(top_combos_ml[:10], 1):
            bar = "█" * int(score * 50)
            print(f"{i:2d}. {str(combo):<22} {bar}  {score:.4f}")

    except FileNotFoundError:
        print("\n⚠️ 번호 예측 모델('models/number_predictor.pkl')을 찾을 수 없어 이 단계는 건너뜁니다.")
    except Exception as e:
        print(f"\n⚠️ 모델 연동 중 오류 발생: {e}")

    # --- 4. Save the Combo Predictor Model ---
    combo_predictor.save_model('models/combo_predictor.pkl')

    print("\n✅ 테스트 완료!")
