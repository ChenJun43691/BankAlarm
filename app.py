# app.py
import os
import csv
import joblib
import pandas as pd
from flask import Flask, request, render_template
from datetime import datetime

# --- Flask 基本設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- 載入模型（包含 preprocessor 與 features）---
MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
preprocessor = model_data["preprocessor"]
# features = model_data["features"]  # 這個不用拿來排序，ColumnTransformer 會用欄位名稱抓

# --- 小工具：字串→整數（失敗給預設值）---
def to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

# --- 首頁：表單 ---
@app.route("/")
def form():
    return render_template("form.html")

# --- 健康檢查（雲端監控可用）---
@app.route("/healthz")
def healthz():
    return "ok", 200

# --- 表單送出與預測 ---
@app.route("/submit", methods=["POST"])
def submit():
    try:
        form_data = request.form

        # 1) 將表單轉成「訓練時的欄位名」DataFrame
        #    注意：年齡區間用 one-hot（和你訓練時一致）
        age_band = form_data.get("年齡區間", "")

        df = pd.DataFrame([{
            "性別(1:男；2:女）": to_int(form_data.get("性別(1:男；2:女）", 1), 1),
            "金融卡收取方式": form_data.get("金融卡收取方式", "").strip(),  # 例如：郵寄 / 親取
            "20歲以下": 1 if age_band == "20歲以下" else 0,
            "20-29歲": 1 if age_band == "20-29歲" else 0,
            "30-39歲": 1 if age_band == "30-39歲" else 0,
            "40-49歲": 1 if age_band == "40-49歲" else 0,
            "50歲以上": 1 if age_band == "50歲以上" else 0,
            "60歲以上": 1 if age_band == "60歲以上" else 0,
            "戶籍通訊地址不同": to_int(form_data.get("戶籍通訊地址不同", 0), 0),
            "戶籍金融卡收取地址不同": to_int(form_data.get("戶籍金融卡收取地址不同", 0), 0),
            "ISP 國籍(1:本國;2:境外)": to_int(form_data.get("ISP 國籍(1:本國;2:境外)", 1), 1),
            "年齡": to_int(form_data.get("年齡", 0), 0),
        }])

        # 2) 前處理 + 預測
        Xp = preprocessor.transform(df)
        y_pred = int(model.predict(Xp)[0])
        # 若想看機率分數（邏輯回歸可用）
        proba = None
        try:
            proba = float(model.predict_proba(Xp)[:, 1][0])
        except Exception:
            pass

        # 3) 轉中文訊息
        if y_pred == 1:
            prediction = "可能為高風險帳戶！請行員加強 KYC！"
        else:
            prediction = "正常帳戶！"

        # 4) 寫入日誌（CSV）
        #    - 把使用者基本欄位也存起來，方便日後分析（姓名/地址等）
        #    - CSV 欄位固定順序，避免日後 append 欄位對不齊
        log_path = os.path.join(LOG_FOLDER, "prediction_log.csv")
        log_fields = [
            "時間",
            "姓名",
            "戶籍地址",
            "通訊地址",
            "金融卡收取地址",
            "性別(1:男；2:女）",
            "金融卡收取方式",
            "20歲以下", "20-29歲", "30-39歲", "40-49歲", "50歲以上", "60歲以上",
            "戶籍通訊地址不同",
            "戶籍金融卡收取地址不同",
            "ISP 國籍(1:本國;2:境外)",
            "年齡",
            "預測結果",
            "預測機率"
        ]
        log_row = {
            "時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "姓名": form_data.get("姓名", "").strip(),
            "戶籍地址": form_data.get("戶籍地址", "").strip(),
            "通訊地址": form_data.get("通訊地址", "").strip(),
            "金融卡收取地址": form_data.get("金融卡收取地址", "").strip(),
            "性別(1:男；2:女）": df.at[0, "性別(1:男；2:女）"],
            "金融卡收取方式": df.at[0, "金融卡收取方式"],
            "20歲以下": df.at[0, "20歲以下"],
            "20-29歲": df.at[0, "20-29歲"],
            "30-39歲": df.at[0, "30-39歲"],
            "40-49歲": df.at[0, "40-49歲"],
            "50歲以上": df.at[0, "50歲以上"],
            "60歲以上": df.at[0, "60歲以上"],
            "戶籍通訊地址不同": df.at[0, "戶籍通訊地址不同"],
            "戶籍金融卡收取地址不同": df.at[0, "戶籍金融卡收取地址不同"],
            "ISP 國籍(1:本國;2:境外)": df.at[0, "ISP 國籍(1:本國;2:境外)"],
            "年齡": df.at[0, "年齡"],
            "預測結果": prediction,
            "預測機率": round(proba, 4) if proba is not None else ""
        }
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(log_row)

        # 5) 回到表單畫面並顯示結果
        #    也把剛送出的值原樣帶回（若你想在 form.html 顯示）
        return render_template("form.html",
                               prediction=prediction,
                               proba=f"{proba:.3f}" if proba is not None else None,
                               **form_data)

    except Exception as e:
        # 任何錯誤在頁面上顯示，方便除錯
        return f"處理錯誤：{e}"

# --- 啟動（本機與雲端皆可）---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)