"""
로또 6/45 조합 예측 모듈
6개 번호 조합을 직접 예측하고 스코어링
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
import lightgbm as lgb
import optuna

# Optuna 로깅 레벨 설정
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LottoComboPredictor:
    """로또 조합 예측 클래스"""
    
    def __init__(self):
        self.model_type = 'lightgbm'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_version = None
        
    def _create_model(self, params=None):
        """LightGBM Regressor 모델 생성"""
        params = params or {}
        base_params = {
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'regression_l1', # MAE
        }
        final_params = {**base_params, **params}
        return lgb.LGBMRegressor(**final_params)

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optuna를 사용한 하이퍼파라미터 튜닝"""
        print(f"\n⚙️ Optuna 하이퍼파라미터 튜닝 시작 ({self.model_type}, {n_trials}회 시도)")

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            }
            model = self._create_model(param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"✅ 튜닝 완료! 최적 MSE: {study.best_value:.4f}")
        print("   - 최적 파라미터:", study.best_params)
        return study.best_params

    def train(self, feature_engineer, start_draw=100, end_draw=None, 
              validation_split=0.2, enable_tuning=False, n_trials=50,
              use_feedback_training=True): # 신규 파라미터
        """조합 스코어링 모델 학습"""
        print("\n" + "="*60)
        print(f"🎯 조합 예측 모델 학습 시작 ({self.model_type})")
        print("="*60)
        
        if use_feedback_training:
            print("   - 학습 방식: 예측 피드백 활용 (Target: 0-6 맞춘 개수)")
            X, y, _ = feature_engineer.build_combo_training_data_with_feedback(
                start_draw=start_draw, end_draw=end_draw
            )
        else:
            print("   - 학습 방식: 구버전 (Target: 0 또는 1)")
            X, y, _ = feature_engineer.build_combo_training_data(
                start_draw=start_draw, end_draw=end_draw
            )

        self.feature_names = X.columns.tolist()
        
        # 데이터가 없는 경우 중단
        if X.empty:
            print("⚠️ 학습 데이터가 없어 모델 학습을 중단합니다.")
            return None

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"\n📊 데이터 분할:")
        print(f"   - 학습: {len(X_train)}개 (Target 평균: {y_train.mean():.2f})")
        print(f"   - 검증: {len(X_val)}개 (Target 평균: {y_val.mean():.2f})")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        best_params = {}
        if enable_tuning:
            best_params = self.tune_hyperparameters(X_train_scaled, y_train, X_val_scaled, y_val, n_trials)

        print(f"\n🔧 모델 학습 중...")
        self.model = self._create_model(best_params)
        self.model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        
        y_pred_val = self.model.predict(X_val_scaled)
        val_mse = mean_squared_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        
        print(f"\n📈 학습 결과 (검증 데이터 기준):")
        print(f"   - 검증 MSE: {val_mse:.4f} (낮을수록 좋음)")
        print(f"   - 검증 R²: {val_r2:.4f} (높을수록 좋음)")
        
        print("\n✅ 학습 완료!")
        self.feature_version = feature_engineer.get_feature_version()
        return {'val_mse': val_mse, 'val_r2': val_r2}

    def score_combination(self, feature_engineer, numbers, reference_draw=None, settings=None):
        if self.model is None: raise RuntimeError("모델이 학습되지 않았습니다.")
        
        if reference_draw is None: reference_draw = feature_engineer.get_latest_draw_number() + 1
        
        if self.feature_version and self.feature_version != feature_engineer.get_feature_version():
            raise ValueError(f"학습된 조합 모델의 피처 버전({self.feature_version})이 현재 데이터 스키마({feature_engineer.get_feature_version()})와 다릅니다. 모델을 재학습시켜주세요.")

        # 예측 시점의 설정을 피처로 변환
        settings_features = feature_engineer._extract_settings_features(json.dumps(settings) if settings else '{}')
        
        features = feature_engineer.extract_combo_features(numbers, reference_draw, settings_features)
        
        # 학습 시 사용된 피처 순서에 맞게 DataFrame 생성
        X = pd.DataFrame([features])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        score = self.model.predict(X_scaled)[0]
        
        # 예측된 점수(맞춘 개수)를 0과 6 사이로 클리핑
        return float(np.clip(score, 0.0, 6.0))

    def predict_top_combos(self, feature_engineer, number_probabilities, n=10, candidate_pool='smart', pool_size=25, reference_draw=None, settings=None):
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. predict_top_combos를 호출하기 전에 train()을 먼저 실행해야 합니다.")

        if reference_draw is None: reference_draw = feature_engineer.get_latest_draw_number() + 1
        
        candidate_combos = self._generate_candidate_combos(feature_engineer, number_probabilities, candidate_pool, pool_size, n)
        
        # 각 후보 조합에 대해 점수 계산 (현재 설정을 전달)
        scored_combos = [(combo, self.score_combination(feature_engineer, combo, reference_draw, settings)) for combo in candidate_combos]
        
        scored_combos.sort(key=lambda x: x[1], reverse=True)
        return scored_combos[:n]

    def _generate_candidate_combos(self, feature_engineer, number_probabilities, mode, pool_size, num_combos):
        if mode == 'smart':
            if not number_probabilities:
                # Fallback to the old method if probabilities are not provided
                print("⚠️ 번호 확률 정보가 없어 기존 '핫 넘버' 방식으로 후보를 생성합니다.")
                recent_freq = feature_engineer.df.tail(50)[['n1','n2','n3','n4','n5','n6']].values.flatten()
                unique, counts = np.unique(recent_freq, return_counts=True)
                top_numbers = [num for num, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:pool_size]]
                if len(top_numbers) < 6: return []
                all_combos = list(combinations(top_numbers, 6))
                sample_size = min(len(all_combos), 5000)
                indices = np.random.choice(len(all_combos), size=sample_size, replace=False) if len(all_combos) > 0 else []
                return [list(all_combos[i]) for i in indices]

            # New ideal logic using probabilities
            numbers = list(number_probabilities.keys())
            p_values = np.array(list(number_probabilities.values()))
            
            # Ensure probabilities sum to 1
            p_sum = p_values.sum()
            if p_sum == 0: # Avoid division by zero if all probs are 0
                p_values = np.full(len(numbers), 1/len(numbers))
            else:
                p_values /= p_sum

            # Generate a large number of candidates using weighted sampling
            num_candidates_to_generate = num_combos * 20 
            
            # Use a set to store unique combinations to avoid duplicates during generation
            candidate_set = set()
            # Add a timeout to prevent infinite loops if it's hard to generate unique combos
            max_gen_attempts = num_candidates_to_generate * 5
            attempts = 0
            while len(candidate_set) < num_candidates_to_generate and attempts < max_gen_attempts:
                combo = tuple(sorted(np.random.choice(numbers, size=6, replace=False, p=p_values).tolist()))
                candidate_set.add(combo)
                attempts += 1
            
            return [list(c) for c in candidate_set]

        else: # random
            return [sorted(np.random.choice(range(1, 46), size=6, replace=False).tolist()) for _ in range(num_combos * 20)]

    def save_model(self, path='models/combo_predictor.pkl'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'model': self.model, 'scaler': self.scaler, 'feature_names': self.feature_names,
            'model_type': self.model_type, 'feature_version': self.feature_version
        }
        with open(path, 'wb') as f: pickle.dump(save_data, f)
        print(f"✅ 모델 저장 완료: {path}")

    def load_model(self, path='models/combo_predictor.pkl', expected_feature_version=None):
        if not Path(path).exists(): raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        with open(path, 'rb') as f: save_data = pickle.load(f)
        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.feature_names = save_data['feature_names']
        self.model_type = save_data.get('model_type', 'lightgbm')
        self.feature_version = save_data.get('feature_version')
        if expected_feature_version and self.feature_version != expected_feature_version:
            # 버전이 다른 경우 경고만 하고 로드는 허용. 단, 예측 시 에러 발생 가능.
            print(f"⚠️ 경고: 저장된 조합 모델 피처 버전({self.feature_version})과 현재 버전({expected_feature_version})이 다릅니다. 모델 재학습을 권장합니다.")
        print(f"✅ 모델 로드 완료: {path} (모델 타입: {self.model_type})")

# (Other methods omitted for brevity)
