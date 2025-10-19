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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb


class LottoNumberPredictor:
    """로또 번호별 출현 확률 예측 클래스"""
    
    def __init__(self, model_type='xgboost'):
        """
        Args:
            model_type: 'random_forest' or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}
        self.feature_version = None
        
    def _create_model(self):
        """모델 생성"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                scale_pos_weight=5  # 클래스 불균형 보정
            )
        else:
            raise ValueError(f"지원하지 않는 모델: {self.model_type}")
    
    def train(self, feature_engineer, start_draw=100, end_draw=None, validation_split=0.2):
        """
        모델 학습
        
        Args:
            feature_engineer: LottoFeatureEngineer 인스턴스
            start_draw: 학습 시작 회차
            end_draw: 학습 종료 회차
            validation_split: 검증 데이터 비율
        """
        print("\n" + "="*60)
        print(f"🤖 번호 예측 모델 학습 시작 ({self.model_type})")
        print("="*60)
        
        # 1. 학습 데이터 생성
        X, y, draws = feature_engineer.build_number_training_data(
            start_draw=start_draw,
            end_draw=end_draw
        )
        
        self.feature_names = X.columns.tolist()
        
        # 2. Train/Validation 분할
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"\n📊 데이터 분할:")
        print(f"   - 학습: {len(X_train)}개 (출현: {y_train.sum()}개)")
        print(f"   - 검증: {len(X_val)}개 (출현: {y_val.sum()}개)")
        
        # 3. 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 4. 모델 학습
        print(f"\n🔧 모델 학습 중...")
        self.model = self._create_model()
        print("   - XGBoost 모델 학습. 10 라운드마다 진행 로그가 표시됩니다.")
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=10
        )
        
        # 5. 평가
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_val = self.model.predict(X_val_scaled)
        y_prob_val = self.model.predict_proba(X_val_scaled)[:, 1]
        
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        val_precision = precision_score(y_val, y_pred_val, zero_division=0)
        val_recall = recall_score(y_val, y_pred_val, zero_division=0)
        val_f1 = f1_score(y_val, y_pred_val, zero_division=0)
        
        try:
            val_auc = roc_auc_score(y_val, y_prob_val)
        except:
            val_auc = 0.0
        
        print(f"\n📈 학습 결과:")
        print(f"   - 학습 정확도: {train_acc:.4f}")
        print(f"   - 검증 정확도: {val_acc:.4f}")
        print(f"   - 검증 정밀도: {val_precision:.4f}")
        print(f"   - 검증 재현율: {val_recall:.4f}")
        print(f"   - 검증 F1: {val_f1:.4f}")
        print(f"   - 검증 AUC: {val_auc:.4f}")
        
        # 6. 피처 중요도 계산
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importances))
            
            print(f"\n🔍 상위 10개 중요 피처:")
            sorted_features = sorted(self.feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
            for feat, importance in sorted_features:
                print(f"   {feat}: {importance:.4f}")
        
        print("\n✅ 학습 완료!")

        self.feature_version = feature_engineer.get_feature_version()

        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_auc': val_auc
        }
    
    def predict_probabilities(self, feature_engineer, draw_no=None):
        """
        다음 회차 각 번호 출현 확률 예측
        
        Args:
            feature_engineer: LottoFeatureEngineer 인스턴스
            draw_no: 기준 회차 (None이면 최신 회차)
            
        Returns:
            dict: {번호: 확률}
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 실행하세요.")

        current_feature_version = feature_engineer.get_feature_version()
        if self.feature_version and self.feature_version != current_feature_version:
            raise ValueError(
                "학습된 모델의 피처 버전이 현재 데이터 스키마와 일치하지 않습니다. 모델을 다시 학습해주세요."
            )

        if draw_no is None:
            draw_no = feature_engineer.get_latest_draw_number() + 1

        # 피처 추출
        features_df = feature_engineer.extract_number_features(draw_no)
        
        # 번호 저장
        numbers = features_df['number'].values
        
        # 피처만 추출
        X = features_df.drop('number', axis=1)
        
        # 범주형 변수 인코딩 (학습시와 동일하게)
        if 'range_group' in X.columns:
            X = pd.get_dummies(X, columns=['range_group'], prefix='range')

        current_columns = set(X.columns)
        missing_features = [col for col in self.feature_names if col not in current_columns]

        # range_* 피처는 범주가 등장하지 않아도 0으로 추가해도 안전하다.
        safe_fill_features = [col for col in missing_features if col.startswith('range_')]
        for col in safe_fill_features:
            X[col] = 0

        remaining_missing = [col for col in missing_features if col not in safe_fill_features]
        if remaining_missing:
            preview = ', '.join(remaining_missing[:5])
            if len(remaining_missing) > 5:
                preview += ', ...'
            raise ValueError(
                f"모델이 사용하는 피처({preview})를 현재 데이터에서 찾을 수 없습니다. 모델을 다시 학습해주세요."
            )

        X = X[self.feature_names]

        # 스케일링
        X_scaled = self.scaler.transform(X)
        
        # 확률 예측
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # 번호: 확률 딕셔너리 생성
        result = {}
        for number, prob in zip(numbers, probabilities):
            result[int(number)] = float(prob)
        
        return result
    
    def get_top_numbers(self, feature_engineer, k=20, draw_no=None):
        """
        확률 높은 상위 K개 번호 반환
        
        Args:
            feature_engineer: LottoFeatureEngineer 인스턴스
            k: 반환할 번호 개수
            draw_no: 기준 회차
            
        Returns:
            list of tuples: [(번호, 확률), ...]
        """
        probabilities = self.predict_probabilities(feature_engineer, draw_no)
        
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_probs[:k]
    
    def backtest(self, feature_engineer, test_draws=20):
        """
        최근 N회차로 백테스트
        
        Args:
            feature_engineer: LottoFeatureEngineer 인스턴스
            test_draws: 테스트할 회차 수
            
        Returns:
            dict: 평가 지표
        """
        print("\n" + "="*60)
        print(f"🔬 백테스트 시작 (최근 {test_draws}회차)")
        print("="*60)
        
        latest_draw = feature_engineer.get_latest_draw_number()
        start_draw = latest_draw - test_draws + 1
        
        all_predictions = []
        all_actuals = []
        hit_counts = []  # 각 회차별 맞춘 개수
        top_k_hits = {6: [], 10: [], 15: [], 20: []}  # 상위 K개 선택시 적중 개수
        
        df = feature_engineer.df
        
        for draw_no in range(start_draw, latest_draw + 1):
            # 예측
            probabilities = self.predict_probabilities(feature_engineer, draw_no)
            
            # 실제 당첨번호
            if draw_no not in df.index:
                continue
            actual_row = df.loc[draw_no]
            actual_numbers = [
                int(actual_row['n1']), int(actual_row['n2']), int(actual_row['n3']),
                int(actual_row['n4']), int(actual_row['n5']), int(actual_row['n6'])
            ]
            
            # 각 번호별 예측 vs 실제
            for number in range(1, 46):
                pred_prob = probabilities[number]
                actual = 1 if number in actual_numbers else 0
                
                all_predictions.append(pred_prob)
                all_actuals.append(actual)
            
            # 상위 K개 선택시 적중률
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            for k in top_k_hits.keys():
                top_k_numbers = [num for num, _ in sorted_probs[:k]]
                hits = len(set(top_k_numbers) & set(actual_numbers))
                top_k_hits[k].append(hits)
        
        # 전체 평가 지표
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        # 임계값 0.5로 이진 분류
        binary_predictions = (all_predictions > 0.5).astype(int)
        
        accuracy = accuracy_score(all_actuals, binary_predictions)
        precision = precision_score(all_actuals, binary_predictions, zero_division=0)
        recall = recall_score(all_actuals, binary_predictions, zero_division=0)
        f1 = f1_score(all_actuals, binary_predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(all_actuals, all_predictions)
        except:
            auc = 0.0
        
        print(f"\n📊 전체 예측 성능:")
        print(f"   - 정확도: {accuracy:.4f}")
        print(f"   - 정밀도: {precision:.4f}")
        print(f"   - 재현율: {recall:.4f}")
        print(f"   - F1 Score: {f1:.4f}")
        print(f"   - AUC: {auc:.4f}")
        
        print(f"\n🎯 상위 K개 선택시 평균 적중 개수:")
        for k, hits in top_k_hits.items():
            avg_hits = np.mean(hits)
            max_hits = np.max(hits)
            print(f"   - 상위 {k:2d}개: 평균 {avg_hits:.2f}개 (최대 {max_hits}개)")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'top_k_performance': {k: np.mean(v) for k, v in top_k_hits.items()}
        }
    
    def save_model(self, path='models/number_predictor.pkl'):
        """모델 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'feature_version': self.feature_version
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"✅ 모델 저장 완료: {path}")

    def load_model(self, path='models/number_predictor.pkl', expected_feature_version=None):
        """모델 로드"""
        if not Path(path).exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.feature_names = save_data['feature_names']
        self.feature_importance = save_data.get('feature_importance', {})
        self.model_type = save_data.get('model_type', 'unknown')
        self.feature_version = save_data.get('feature_version')

        if expected_feature_version and self.feature_version != expected_feature_version:
            raise ValueError(
                f"저장된 모델 피처 버전({self.feature_version})과 현재 버전({expected_feature_version})이 다릅니다."
            )

        print(f"✅ 모델 로드 완료: {path}")
    
    def get_feature_importance(self, top_k=20):
        """
        피처 중요도 반환
        
        Args:
            top_k: 상위 K개 반환
            
        Returns:
            list of tuples: [(피처명, 중요도), ...]
        """
        if not self.feature_importance:
            return []
        
        sorted_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)
        
        return sorted_features[:top_k]


if __name__ == "__main__":
    from analysis.lotto_feature_engineer import LottoFeatureEngineer

    print("\n" + "="*60)
    print("🤖 Number Predictor 테스트")
    print("="*60)
    
    # Feature Engineer 생성
    engineer = LottoFeatureEngineer('lotto/data/lotto_history.csv')
    latest_draw = engineer.get_latest_draw_number()
    
    # Predictor 생성 및 학습
    predictor = LottoNumberPredictor(model_type='xgboost')
    
    # 학습 (최근 200회차 사용, 마지막 20회차는 테스트용 제외)
    train_end = latest_draw - 20
    train_start = max(100, train_end - 200)
    
    results = predictor.train(
        engineer, 
        start_draw=train_start, 
        end_draw=train_end,
        validation_split=0.2
    )
    
    # 다음 회차 예측
    print("\n" + "="*60)
    print(f"🎯 다음 회차 ({latest_draw + 1}회) 예측")
    print("="*60)
    
    probabilities = predictor.predict_probabilities(engineer)
    top_numbers = predictor.get_top_numbers(engineer, k=20)
    
    print(f"\n🏆 상위 20개 번호:")
    for i, (number, prob) in enumerate(top_numbers, 1):
        bar = "█" * int(prob * 50)
        print(f"{i:2d}. {number:2d}번  {bar}  {prob*100:.2f}%")
    
    # 백테스트
    backtest_results = predictor.backtest(engineer, test_draws=20)
    
    # 모델 저장
    predictor.save_model('models/number_predictor.pkl')
    
    print("\n✅ 테스트 완료!")
