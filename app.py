
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mplfinance as mpf

st.set_page_config(page_title="台股預測", page_icon="💰")

st.title("台股上漲或下跌預測")

ticker = st.text_input("輸入股票代碼（如 2330.TW）", value="2330.TW")

# 設定期間
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365)

if st.button("開始預測"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            st.error(f"錯誤：{ticker} 抓不到資料，請確認代碼或日期")
        else:
            # 顯示資料
            st.success(f"成功取得 {ticker} 的資料，共 {len(data)} 筆")

            # 顯示 K 線圖
            fig, _ = mpf.plot(data, type='candle', style='charles', title=f"{ticker} K 線圖", volume=True, returnfig=True)
            st.pyplot(fig)

            # 加入目標欄位：漲(1)或跌(0)
            data["Tomorrow_Close"] = data["Close"].shift(-1)
            data["Target"] = (data["Tomorrow_Close"] > data["Close"]).astype(int)
            data.dropna(inplace=True)

            # 使用簡單特徵
            features = data[["Open", "High", "Low", "Close", "Volume"]]
            target = data["Target"]

            # 切分資料集
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

            # 建立模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 預測與評估
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.info(f"預測準確率：{acc*100:.2f}%")

            # 顯示最後一天預測
            last_input = features.iloc[[-1]]
            prediction = model.predict(last_input)[0]
            result = "上漲" if prediction == 1 else "下跌"
            st.subheader(f"明天預測：{result}")
    except Exception as e:
        st.error(f"發生錯誤：{str(e)}")
