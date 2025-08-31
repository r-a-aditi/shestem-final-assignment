import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Section A: Data Import & Preprocessing
df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vRmOY52vyHKgJD_SeV063EOV6pyJsvQfUjmUMiyyzALs9AiH0q_YOhi2MBuxWCgK-N6HKUFZ4k-Hbqn/pub?output=csv")
print(df.head(10))

print("\n\n---Missing value identification---\n")
print(df.isnull().sum()) # no missing data found

df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['Day_of_week'] = df['DateTime'].dt.dayofweek
df['is_weekend'] = df['Day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Section B: Exploratory Data Analysis

# Traffic volume over time
df.set_index('DateTime', inplace=True)
daily_avg = df.resample('D')['Vehicles'].mean()

plt.figure(figsize=(12, 4))
plt.plot(daily_avg.index, daily_avg.values)
plt.xlabel('Date')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("\n\n")

# Average traffic volume by Hour
plt.figure(figsize=(12, 4))
grp_by_hr = df.groupby('Hour')['Vehicles'].mean()
grp_by_hr.plot(kind='bar')
plt.xlabel('\nHour')
plt.ylabel('Average Traffic Volume')
plt.title('Average Traffic Volume by Hour')
plt.tight_layout()
plt.show()
print("\n\n")

# Average traffic volume by Day of Week
plt.figure(figsize=(12, 4))
grp_by_day = df.groupby('Day_of_week')['Vehicles'].mean()
grp_by_day.plot(kind='bar', color='lightcoral')
plt.xlabel('Day of Week')
plt.ylabel('Average Traffic Volume')
plt.title('\n\n\nAverage Traffic Volume by Day of Week')
plt.tight_layout()
plt.show()

print("\n\n")

# Section C: Model Building & Evaluation

X = df.drop('Vehicles', axis=1)
y = df['Vehicles']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("\n\n--Linear Regression--\n")
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_lr = reg.predict(X_test)

rmse_lr = root_mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Root Mean Square Error (RMSE) = {:.3f}".format(rmse_lr))
print("R-squared = {:.3f}".format(r2_lr))


print("\n\n--XGBoost--\n")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

rmse_xgb = root_mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("Root Mean Square Error (RMSE) = {:.3f}".format(rmse_xgb))
print("R-squared = {:.3f}\n".format(r2_xgb))

# Comparing errors of Linear regression and XGBoost

plt.figure(figsize=(4,3))
plt.bar('Linear Regression', rmse_lr, color='skyblue')
plt.bar('XGBoost', rmse_xgb, color='lightcoral')
plt.ylabel('Root Mean Square Error (RMSE)')
plt.tight_layout()
plt.show() 
# XGBoost gives better result with less error


# Analysis for section D

print("----Top three hours with the highest predicted traffic\n")
print(pd.DataFrame(grp_by_hr).sort_values(by='Vehicles', ascending=False).head(3))

grp_by_junc = df.groupby('Junction')['Vehicles'].mean()
print("\n----Top three junctions with the highest predicted traffic\n")
print(pd.DataFrame(grp_by_junc).sort_values(by='Vehicles', ascending=False))

#  Identify one instance where the model prediction was significantly off
print("\n----Spotting the worst prediction of Linear regression\n")

errors = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_lr,
    'Error': y_pred_lr - y_test,
    'Absolute_Error': abs(y_pred_lr - y_test),
    'Junction' : X_test['Junction']
})
worst_case = errors.sort_values(by='Absolute_Error', ascending=False).head(1)
print(worst_case)

print("\n----Spotting the worst prediction of XGBoost\n")

errors = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_xgb,
    'Error': y_pred_xgb - y_test,
    'Absolute_Error': abs(y_pred_xgb - y_test),
    'Junction' : X_test['Junction']
})
worst_case = errors.sort_values(by='Absolute_Error', ascending=False).head(1)
print(worst_case)


# Section G: Visualization & Insights

# Actual vs predicted traffic volumes for the test set

# Linear regression

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.scatter(y_test, y_pred_lr, alpha=0.7, color = 'skyblue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Actual Traffic Volume')
plt.ylabel('Predicted Traffic Volume')
plt.title('Actual vs. Predicted Traffic Volume in Linear regression')

# XGBoost

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_xgb, alpha=0.7, color = 'lightcoral')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Actual Traffic Volume')
plt.title('Actual vs. Predicted Traffic Volume in XGBoost')

plt.tight_layout()
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Looking at the last 24 hours of vehicle counts to predict the next hour.
sequence_length = 24
data_values = df['Vehicles'].values
X_seq, y_seq = create_sequences(data_values, sequence_length)
X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

# Build GRU Model
model = Sequential()
model.add(GRU(64, input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_seq, y_seq, epochs=2, batch_size=32, validation_split=0.2)

# Predict on the training data
y_pred_gru = model.predict(X_seq)

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))

plt.scatter(y_seq, y_pred_gru, alpha=0.7, color = 'green')
plt.plot([min(y_seq), max(y_seq)], [min(y_seq), max(y_seq)], 'k--', lw=2)
plt.title('GRU Predictions vs Actual')
plt.xlabel('Time Step')
plt.ylabel('Vehicles')
plt.tight_layout()
plt.show()
