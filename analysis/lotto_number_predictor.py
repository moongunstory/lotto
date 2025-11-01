"""
로또 6/45 번호별 확률 예측 모듈
ML 모델을 사용하여 각 번호(1~45)의 다음 회차 출현 확률 예측
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import optuna

# Optuna 로깅 레벨 설정
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LottoNumberPredictor:
    """로또 번호별 출현 확률 예측 클래스"""
    
    def __init__(self):
        self.model_type = 'lightgbm'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}
        self.feature_version = None
        
    def _create_model(self, params=None):
        """LightGBM 모델 생성"""
        params = params or {}
        base_params = {
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'binary',
        }
        final_params = {**base_params, **params}
        return lgb.LGBMClassifier(**final_params)

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
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
            }
            model = self._create_model(param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
            y_prob = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_prob)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"✅ 튜닝 완료! 최적 AUC: {study.best_value:.4f}")
        print("   - 최적 파라미터:", study.best_params)
        return study.best_params

    def train(self, feature_engineer, start_draw=100, end_draw=None, validation_split=0.2, 
              enable_tuning=False, n_trials=50):
        """모델 학습"""
        print("\n" + "="*60)
        print(f"🤖 번호 예측 모델 학습 시작 ({self.model_type})")
        print("="*60)
        
        X, y, _ = feature_engineer.build_number_training_data(start_draw=start_draw, end_draw=end_draw)
        self.feature_names = X.columns.tolist()
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"\n📊 데이터 분할:")
        print(f"   - 학습: {len(X_train)}개 (출현: {y_train.sum()}개)")
        print(f"   - 검증: {len(X_val)}개 (출현: {y_val.sum()}개)")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        best_params = {}
        if enable_tuning:
            best_params = self.tune_hyperparameters(X_train_scaled, y_train, X_val_scaled, y_val, n_trials)

        print(f"\n🔧 모델 학습 중...")
        self.model = self._create_model(best_params)
        
        self.model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        
        y_pred_val = self.model.predict(X_val_scaled)
        y_prob_val = self.model.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_prob_val) if y_val.sum() > 0 else 0.0
        
        print(f"\n📈 학습 결과 (검증 AUC): {val_auc:.4f}")
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        print("\n✅ 학습 완료!")
        self.feature_version = feature_engineer.get_feature_version()
        return {'val_auc': val_auc}

    def predict_probabilities(self, feature_engineer, draw_no=None):
        if self.model is None: raise RuntimeError("모델이 학습되지 않았습니다.")
        # ... (rest of the method is the same)
        current_feature_version = feature_engineer.get_feature_version()
        if self.feature_version and self.feature_version != current_feature_version:
            raise ValueError("학습된 모델의 피처 버전이 현재 데이터 스키마와 일치하지 않습니다. 모델을 다시 학습해주세요.")
        if draw_no is None: draw_no = feature_engineer.get_latest_draw_number() + 1
        features_df = feature_engineer.extract_number_features(draw_no)
        numbers = features_df['number'].values
        X = features_df.drop('number', axis=1)
        if 'range_group' in X.columns: X = pd.get_dummies(X, columns=['range_group'], prefix='range')
        missing_features = [col for col in self.feature_names if col not in X.columns]
        for col in missing_features: X[col] = 0
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return dict(zip(map(int, numbers), map(float, probabilities)))

    def save_model(self, path='models/number_predictor.pkl'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'model': self.model, 'scaler': self.scaler, 'feature_names': self.feature_names,
            'feature_importance': self.feature_importance, 'model_type': self.model_type,
            'feature_version': self.feature_version
        }
        with open(path, 'wb') as f: pickle.dump(save_data, f)
        print(f"✅ 모델 저장 완료: {path}")

    def load_model(self, path='models/number_predictor.pkl', expected_feature_version=None):
        if not Path(path).exists(): raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        with open(path, 'rb') as f: save_data = pickle.load(f)
        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.feature_names = save_data['feature_names']
        self.feature_importance = save_data.get('feature_importance', {})
        self.model_type = save_data.get('model_type', 'lightgbm')
        self.feature_version = save_data.get('feature_version')
        if expected_feature_version and self.feature_version != expected_feature_version:
            raise ValueError(f"저장된 모델 피처 버전({self.feature_version})과 현재 버전({expected_feature_version})이 다릅니다.")
        print(f"✅ 모델 로드 완료: {path} (모델 타입: {self.model_type})")

# Other methods like get_top_numbers, get_feature_importance can remain mostly the same
# ... (omitted for brevity)

