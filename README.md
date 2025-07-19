# 📊 Sales Forecast Prediction using XGBoost 🧠
This project uses historical sales data to forecast future sales using lag-based time series features and the XGBoost regression model. It involves data preprocessing, visualization, feature engineering, and model evaluation.

# 🗂️ Project Files
1. sales_forecast_prediction.ipynb: Main Jupyter Notebook containing code for data processing, visualization, feature engineering, and prediction.

2. train.csv: Dataset containing historical order and sales records.

# 🛠️ Libraries Used
1. Python

2. Pandas & NumPy – Data manipulation 📊

3. Matplotlib & Seaborn – Visualization 🖼️

4. Scikit-learn – Preprocessing and metrics

5. XGBoost – Gradient Boosting Regression Model ⚡

# 📦 Dataset Overview
Contains order-level data with fields like:

Order Date 🗓️

Sales 💰

Goal: Predict future Sales using lag-based features.

🔍 Data Preprocessing
```python
df = pd.read_csv('train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
```
Grouped by Order Date to aggregate daily sales.

# 📈 Visualization
```python
plt.plot(sales_by_date['Order Date'], sales_by_date['Sales'])
```
Trend line showing how sales change over time.

Useful for spotting seasonality or spikes.

# 🧠 Feature Engineering – Lag Creation
```python
def create_features(data, lag=5):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['Sales'].shift(i)
```
Lag features allow the model to learn from past values to predict future ones.

Missing values from lag shifts are dropped.

# 🔀 Train/Test Split
```python
X = sales_with_lags.drop(['Order Date', 'Sales'], axis=1)
y = sales_with_lags['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```
Data is not shuffled to preserve time order.

# ⚙️ Model Training (XGBoost)
```python
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
```
Trains a powerful gradient boosting model.

Suitable for time series regression with structured input.

# 📏 Evaluation
```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
```
Metric used: Mean Squared Error (MSE)

Lower values indicate better performance.

