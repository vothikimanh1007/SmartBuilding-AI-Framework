from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
import os

app = FastAPI(title="Smart Building AI API")

# CẤU HÌNH CORS: Cho phép trang web từ Vercel có thể gọi đến API này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bạn có thể thay "*" bằng link vercel của bạn để bảo mật hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn đến mô hình
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XGB_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_best_model.pkl')
ANN_PATH = os.path.join(BASE_DIR, 'models', 'deep_ann_model.h5')

# Tải mô hình khi khởi động server
xgb_model = joblib.load(XGB_PATH)
ann_model = tf.keras.models.load_model(ANN_PATH)

# Định nghĩa cấu trúc dữ liệu đầu vào
class SensorInput(BaseModel):
    T1: float
    RH_1: float
    T_out: float
    hour: int

@app.get("/")
def home():
    return {"message": "Smart Building AI API is running!"}

@app.post("/predict")
async def predict(data: SensorInput):
    try:
        # Giả lập vector 30 đặc trưng tương ứng với lúc huấn luyện
        # (Trong thực tế bạn cần gán đúng vị trí index cho các biến T1, RH_1,...)
        features = np.zeros((1, 30))
        features[0, 0] = data.T1
        features[0, 1] = data.RH_1
        features[0, 18] = data.T_out
        features[0, 24] = np.sin(2 * np.pi * data.hour / 24) # hr_sin
        features[0, 25] = np.cos(2 * np.pi * data.hour / 24) # hr_cos
        
        # Chạy dự báo
        xgb_log = xgb_model.predict(features)[0]
        ann_log = ann_model.predict(features, verbose=0)[0][0]
        
        # Đảo ngược hàm log1p (vì lúc train ta dùng log)
        res_xgb = float(np.expm1(xgb_log))
        res_ann = float(np.expm1(ann_log))
        
        return {
            "status": "success",
            "xgboost_prediction": round(res_xgb, 2),
            "ann_prediction": round(res_ann, 2),
            "unit": "Wh"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Cách lưu và cấu trúc thư mục

Bạn nên lưu tệp này theo cấu trúc sau để các đường dẫn `models/` hoạt động chính xác:

```text
SmartBuilding-AI-Framework/
├── app.py                 <-- Lưu ở đây
├── models/
│   ├── xgboost_best_model.pkl
│   └── deep_ann_model.h5
├── datasets/
├── requirements.txt       <-- Đảm bảo có 'fastapi' và 'uvicorn'
└── README.md
```

### 3. Hướng dẫn đẩy lên GitHub và Triển khai

1.  **Cập nhật GitHub:**
    Sử dụng lệnh sau để đẩy tệp lên kho lưu trữ:
    ```bash
    git add app.py
    git commit -m "Add FastAPI backend for model inference"
    git push origin main
