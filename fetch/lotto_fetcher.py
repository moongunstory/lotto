"""
로또 6/45 데이터 수집 모듈
동행복권 API를 통해 1회차부터 최신 회차까지 데이터 자동 수집
"""

import requests
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm


class LottoFetcher:
    """로또 데이터 수집 클래스"""
    
    def __init__(self, data_path='data/lotto_history.csv'):
        self.base_url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
    
    def fetch_draw(self, draw_no):
        """특정 회차 데이터 가져오기"""
        try:
            response = requests.get(f"{self.base_url}{draw_no}", timeout=10)
            data = response.json()
            
            if data['returnValue'] == 'success':
                return {
                    'draw_no': draw_no,
                    'date': data['drwNoDate'],
                    'n1': data['drwtNo1'],
                    'n2': data['drwtNo2'],
                    'n3': data['drwtNo3'],
                    'n4': data['drwtNo4'],
                    'n5': data['drwtNo5'],
                    'n6': data['drwtNo6'],
                    'bonus': data['bnusNo'],
                    'total_sales': data.get('totSellamnt', 0),
                    'first_prize': data.get('firstWinamnt', 0),
                    'first_winners': data.get('firstPrzwnerCo', 0)
                }
            return None
        except Exception as e:
            print(f"회차 {draw_no} 수집 실패: {e}")
            return None
    
    def get_latest_draw_no(self):
        """최신 회차 번호 확인"""
        for draw_no in range(1200, 0, -1):  # 현재 약 1150회차까지 진행됨
            data = self.fetch_draw(draw_no)
            if data:
                return draw_no
        return 1
    
    def fetch_all_data(self, start_draw=1, end_draw=None):
        """전체 데이터 수집"""
        if end_draw is None:
            end_draw = self.get_latest_draw_no()
        
        print(f"🎲 로또 데이터 수집 시작: {start_draw}회 ~ {end_draw}회")
        
        all_data = []
        for draw_no in tqdm(range(start_draw, end_draw + 1), desc="데이터 수집 중"):
            data = self.fetch_draw(draw_no)
            if data:
                all_data.append(data)
            time.sleep(0.1)  # API 부하 방지
        
        df = pd.DataFrame(all_data)
        df = df.dropna()  # 결측치 제거
        df.to_csv(self.data_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 수집 완료: {len(df)}개 회차 데이터 저장됨")
        return df
    
    def update_data(self):
        """기존 데이터 업데이트"""
        if self.data_path.exists():
            df = pd.read_csv(self.data_path)
            last_draw = int(df['draw_no'].max())
            latest_draw = self.get_latest_draw_no()
            
            if latest_draw > last_draw:
                print(f"🔄 신규 회차 업데이트: {last_draw + 1}회 ~ {latest_draw}회")
                new_df = self.fetch_all_data(last_draw + 1, latest_draw)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(self.data_path, index=False, encoding='utf-8-sig')
                return df
            else:
                print("✅ 이미 최신 데이터입니다.")
                return df
        else:
            return self.fetch_all_data()
    
    def load_data(self):
        """저장된 데이터 불러오기"""
        if self.data_path.exists():
            return pd.read_csv(self.data_path)
        else:
            print("⚠️ 데이터 파일이 없습니다. 수집을 시작합니다...")
            return self.fetch_all_data()


if __name__ == "__main__":
    fetcher = LottoFetcher()
    df = fetcher.update_data()
    print(f"\n📊 총 {len(df)}개 회차 데이터")
    print(df.tail())