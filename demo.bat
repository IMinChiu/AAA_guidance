@echo off

REM "Please change the path to your own path"
cd "C:\Users\chenp\Downloads\aorta_demo_v3"

REM "Please change the path to your own path"
call C:\ProgramData\miniconda3\Scripts\activate.bat

call conda activate echo
call python demo.py --device GPU --jobs 2

pause