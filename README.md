#Project: SGAN Anomaly Detection (K-Fold CV + YAML config)

This repository contains a modular SGAN implementation for anomaly detection on ultrasonic C-scan images.

#Quick Start
1. Arrange your dataset (images) and labels CSV.

- `data_dir` should contain image files referenced by `labels_csv`.
- `labels_csv` must contain two columns: `filename,label` where `label` is 0 (normal) or 1 (defect).

Example `labels.csv`:
filename,label 
img0001.jpg,0 
img0002.jpg,1

2. Edit `config.yaml` with your paths and hyperparameters.

3. Run training:

Run train.bat

or

Run setup_venv.bat and in cmd paste:
```bash
set CUDA_VISIBLE_DEVICES=
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
C:\csce768\Scripts\python.exe -u train_cv.py
```
4. Results saved under out_dir/fold{n}

5. To evaluate images, place them into test_data folder and set paths for the
generator epoch and image for evaluation in eval.py

Then run:
```bash
C:\csce768\Scripts\python.exe eval.py
```

