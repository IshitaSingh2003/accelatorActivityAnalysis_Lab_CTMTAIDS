# Human Activity Recognition from Accelerometer Data

This project classifies **six human activities** (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing) using smartphone **accelerometer data** (`x_accel`, `y_accel`, `z_accel`) with timestamps and user IDs.

## Dataset

- ~1.09M samples, 36 users.  
- Columns: `user`, `activity`, `timestamp`, `x_accel`, `y_accel`, `z_accel`.  
- Sampling rate: 20 Hz (1 sample every 50 ms).  

## Preprocessing

- Loaded CSV and assigned headers.  
- Removed trailing semicolons and converted accelerations to `float`.  
- Label‑encoded `activity` to integers 0–5.  
- Standardized accelerometer axes; imputed occasional NaNs with column means.  

## Temporal feature extraction

- Sorted data by `user` and `timestamp`.  
- Used sliding windows (size 40, step 20) per user–activity sequence.  
- For each window and axis, computed mean, std, min, max, energy, and cross‑axis correlations (xy, xz, yz).  
- Built a window‑level feature table where each row = one time window, label = activity.  

## Models

- **Classical (tabular)**:  
  - Multinomial Logistic Regression (with imputation + standardization).  
  - Random Forest Classifier.  
  - Linear Discriminant Analysis (LDA).  

- **Sequence model**:  
  - LSTM network taking windows of shape `(time_steps=40, features=3)` and outputting one of 6 activities.  

## Evaluation

- Train–test split with stratification by activity.  
- Metrics: accuracy, precision, recall, F1 (per class) and confusion matrices.  
- Compared per‑sample vs. window‑based vs. LSTM models to show benefits of using temporal patterns.
