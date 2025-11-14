#!/bin/bash
# 로또 VIP 프로그램 실행 (Linux 버전)

echo "========================================"
echo "🎯 로또 VIP 분석 프로그램 시작"
echo "========================================"
echo
echo "가상 환경을 활성화합니다..."
source ./.venv/bin/activate
echo "가상 환경 활성화 완료."
echo
echo "브라우저가 자동으로 열립니다..."
echo "종료하려면 Ctrl+C를 누르세요."
echo "========================================"
echo

streamlit run app.py
