import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

# Đường dẫn tệp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XGB_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_best_model.pkl')
ANN_PATH = os.path.join(BASE_DIR, 'models', 'deep_ann_model.h5')

def predict_sample(t1, rh1, t_out, hour):
    # Tải mô hình
    xgb = joblib.load(XGB_PATH)
    ann = tf.keras.models.load_model(ANN_PATH)
    
    # Giả lập vector đặc trưng (30 đặc trưng như lúc train)
    # Trong thực tế bạn cần scale dữ liệu dựa trên scaler đã lưu
    input_data = np.random.rand(1, 30) 
    
    # Dự báo
    p1 = np.expm1(xgb.predict(input_data)[0])
    p2 = np.expm1(ann.predict(input_data, verbose=0)[0][0])
    return p1, p2

if __name__ == "__main__":
    print("--- Smart Building Energy AI Demo ---")
    p1, p2 = predict_sample(22.5, 45.0, 15.0, 18)
    print(f"XGBoost: {p1:.2f} Wh | Deep ANN: {p2:.2f} Wh")
