@echo off
chcp 65001 > nul
title 로또 VIP 프로그램 실행

echo ========================================
echo 🎯 로또 VIP 분석 프로그램 시작
echo ========================================
echo.
echo 브라우저가 자동으로 열립니다...
echo 종료하려면 Ctrl+C를 누르세요.
echo ========================================
echo.

streamlit run app.py

pause
