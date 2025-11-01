import sys
import json
from pathlib import Path

# LottoFetcher를 import하기 위해 경로 추가
sys.path.append(str(Path(__file__).resolve().parent))

from fetch.lotto_fetcher import LottoFetcher


def check_lotto_rank(user_numbers, winning_numbers, bonus_number):
    """
    사용자의 로또 번호 등수를 확인합니다.
    """
    matches = len(user_numbers.intersection(winning_numbers))

    if matches == 6:
        return ("1등", matches)
    elif matches == 5:
        if bonus_number in user_numbers:
            return ("2등", matches)
        else:
            return ("3등", matches)
    elif matches == 4:
        return ("4등", matches)
    elif matches == 3:
        return ("5등", matches)
    else:
        return ("낙첨", matches)


def get_prize_money(rank):
    """
    등수에 따른 당첨금을 반환합니다. (1-3등은 변동)
    """
    if rank == "1등":
        return "변동 (판매량에 따라 결정)"
    elif rank == "2등":
        return "변동 (판매량에 따라 결정)"
    elif rank == "3등":
        return "변동 (판매량에 따라 결정)"
    elif rank == "4등":
        return "50,000원"
    elif rank == "5등":
        return "5,000원"
    else:
        return "0원"


def main():
    """
    메인 로직
    """
    # 1. 로또 회차 입력받기
    try:
        draw_no_input = input("결과를 확인할 회차를 입력하세요 (예: 1195): ")
        draw_no = int(draw_no_input)
    except ValueError:
        print("❌ 잘못된 입력입니다. 숫자를 입력해주세요.")
        return

    # 2. 당첨번호 자동 조회
    print(f"🔄 {draw_no}회차 당첨번호를 조회합니다...")
    fetcher = LottoFetcher()
    winning_data = fetcher.fetch_draw(draw_no)

    if not winning_data or winning_data.get('returnValue') != 'success':
        print(f"❌ {draw_no}회차 당첨번호를 조회할 수 없습니다. 회차 번호를 확인해주세요.")
        return

    winning_numbers = {winning_data[f'drwtNo{i}'] for i in range(1, 7)}
    bonus_number = winning_data['bnusNo']
    
    print("---")
    print(f"🎯 {draw_no}회 당첨번호: {sorted(list(winning_numbers))}")
    print(f"✨ 보너스 번호: {bonus_number}")
    print("---")

    # 3. 예측 파일 목록 보여주고 선택받기
    predictions_dir = Path('data/predictions')
    if not predictions_dir.exists() or not any(predictions_dir.glob('*.json')):
        print(f"❌ 확인할 예측 파일이 '{predictions_dir}' 폴더에 없습니다.")
        print("AI 스마트 조합 탭에서 '추천 조합 저장'을 먼저 실행해주세요.")
        return

    prediction_files = sorted([f for f in predictions_dir.glob('*.json')], reverse=True)
    
    print("확인할 예측 파일을 선택하세요:")
    for i, f in enumerate(prediction_files):
        print(f"  [{i+1}] {f.name}")
    
    try:
        file_choice_input = input(f"번호를 선택하세요 (1-{len(prediction_files)}): ")
        file_idx = int(file_choice_input) - 1
        if not 0 <= file_idx < len(prediction_files):
            raise ValueError
        selected_file = prediction_files[file_idx]
    except (ValueError, IndexError):
        print("❌ 잘못된 선택입니다.")
        return

    # 4. 선택된 예측 파일 로드
    with open(selected_file, 'r', encoding='utf-8') as f:
        user_combos = json.load(f)

    print(f"\n--- 📄 '{selected_file.name}' 파일 결과 확인 ---")
    
    # 5. 결과 분석 및 요약
    rank_counts = {"1등": 0, "2등": 0, "3등": 0, "4등": 0, "5등": 0, "낙첨": 0}
    total_prize = 0
    
    for i, user_numbers_list in enumerate(user_combos):
        user_numbers = set(user_numbers_list)
        rank, matches = check_lotto_rank(user_numbers, winning_numbers, bonus_number)
        
        result_line = f"조합 {i+1:02d} {str(user_numbers_list):<28} -> 맞은 개수: {matches}개, 결과: {rank}"
        if rank != "낙첨":
            print(f"🎉 {result_line}")
        else:
            print(result_line)


        rank_counts[rank] += 1
        if rank == "4등":
            total_prize += 50000
        elif rank == "5등":
            total_prize += 5000

    print("\n--- 📊 최종 결과 ---")
    total_wins = sum(count for rank, count in rank_counts.items() if rank != '낙첨')
    if total_wins == 0:
        print("아쉽지만, 이번 회차에는 당첨된 조합이 없습니다.")
    else:
        for rank, count in rank_counts.items():
            if count > 0:
                print(f"{rank}: {count}개")

    print(f"\n💰 총 당첨금 (4, 5등만 합산): {total_prize:,}원")


if __name__ == "__main__":
    main()