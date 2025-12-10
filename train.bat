@echo off

cd /d C:\

python -m venv csce

call csce\scripts\activate.bat

echo ----------------------------------------------------------
echo Environment activated
echo ----------------------------------------------------------

python -m pip install --upgrade pip

echo Installing requirements....
C:\csce768\Scripts\python.exe -m pip install --no-cache-dir -r "C:\Users\jrhoo\OneDrive - University of South Carolina\School shtuff\Fall 2025 OD\CSCE 768\Final project\requirements.txt"

cd /d C:\Users\jrhoo\OneDrive - University of South Carolina\School shtuff\Fall 2025 OD\CSCE 768\Final project\root

set CUDA_VISIBLE_DEVICES=
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
C:\csce768\Scripts\python.exe -u train_cv.py

cmd /k