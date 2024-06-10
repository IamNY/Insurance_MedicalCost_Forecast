import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from scipy import stats

# 데이터 로드
data = pd.read_csv('data/insurance.csv')

# 의료비 변수에 대한 이상치 제거
z_scores = np.abs(stats.zscore(data['MedicalCost']))
data_cleaned = data[z_scores < 3].copy()

# 원래 인덱스 저장
data_cleaned.loc[:, 'OriginalIndex'] = data_cleaned.index

# 일부 데이터의 의료비용을 결측값으로 설정하기 전에 저장
missing_rate = 0.2
n_missing = int(len(data_cleaned) * missing_rate)

# RandomState 객체 생성
rng = np.random.RandomState(42)
missing_indices = rng.choice(data_cleaned.index, n_missing, replace=False)

original_values_sample = data_cleaned.loc[missing_indices, ['OriginalIndex', 'MedicalCost']].sort_values('OriginalIndex')

# 의료비용을 결측값으로 설정
data_cleaned.loc[missing_indices, 'MedicalCost'] = np.nan

# 결측값이 있는 데이터와 없는 데이터 분리
data_train = data_cleaned.dropna(subset=['MedicalCost'])
data_test = data_cleaned[data_cleaned['MedicalCost'].isna()]

# 특성과 타겟 분리
X_train = data_train.drop(['MedicalCost', 'OriginalIndex'], axis=1)
y_train = data_train['MedicalCost']
X_test = data_test.drop(['MedicalCost', 'OriginalIndex'], axis=1)

# 범주형 변수 인코딩
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# 스케일링
scaler = StandardScaler()
X_train[['Age', 'BMI', 'Children']] = scaler.fit_transform(X_train[['Age', 'BMI', 'Children']])
X_test[['Age', 'BMI', 'Children']] = scaler.transform(X_test[['Age', 'BMI', 'Children']])

# 하이퍼파라미터 그리드 설정
param_grids = {
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'Ridge': {'alpha': [1.0, 0.1, 0.01]},
    'Lasso': {'alpha': [1.0, 0.1, 0.01]},
    'ElasticNet': {'alpha': [1.0, 0.1, 0.01], 'l1_ratio': [0.2, 0.5, 0.8]},
    'SVR': {'C': [1, 10], 'epsilon': [0.1, 0.2]},
    'KNN Regressor': {'n_neighbors': [3, 5, 7]},
    'Decision Tree': {'max_depth': [None, 10, 20]},
    'AdaBoost': {'n_estimators': [50, 100]},
    'Extra Trees': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
}

# 모델 리스트
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'KNN Regressor': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Extra Trees': ExtraTreesRegressor()
}

# 결과 저장용 데이터프레임
results = pd.DataFrame(columns=['Model', 'R2', 'MSE', 'MAE', 'Average_Difference'])

# 결과 저장용 데이터프레임
actual_vs_predicted_list = []

for model_name, model in models.items():
    if model_name in param_grids:
        param_grid = param_grids[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model

    y_pred = best_model.predict(X_test)

    # 예측값 정렬
    y_pred_df = pd.DataFrame({'OriginalIndex': data_test['OriginalIndex'], 'Prediction': y_pred}).sort_values('OriginalIndex')
    y_pred_sorted = y_pred_df['Prediction'].values

    mse = mean_squared_error(original_values_sample['MedicalCost'], y_pred_sorted)
    r2 = r2_score(original_values_sample['MedicalCost'], y_pred_sorted)
    mae = mean_absolute_error(original_values_sample['MedicalCost'], y_pred_sorted)

    results = pd.concat([results, pd.DataFrame({
        'Model': [model_name],
        'R2': [r2],
        'MAE': [f"{mae:.2f}"],
        'MSE': [f"{mse:.2f}"],
    })], ignore_index=True)

    # 각 모델별 실제 값과 예측 값 출력
    actual_vs_predicted = pd.DataFrame({
        'Model': model_name,
        'OriginalIndex': original_values_sample['OriginalIndex'],
        'Actual_MedicalCost': [f"{val:.2f}" for val in original_values_sample['MedicalCost']],
        'Predicted_Index': y_pred_df['OriginalIndex'],
        'Predicted_MedicalCost': [f"{pred:.2f}" for pred in y_pred_sorted],
        'Difference': [f"{abs(val - pred):.2f}" for val, pred in zip(original_values_sample['MedicalCost'], y_pred_sorted)]
    })
    actual_vs_predicted_list.append(actual_vs_predicted)

# 각 모델별 Difference의 평균을 계산
average_differences = []

for df in actual_vs_predicted_list:
    avg_diff = np.mean(df['Difference'].astype(float))
    average_differences.append(avg_diff)

# 결과 데이터프레임에 Difference 평균 추가
results['Average_Difference'] = [f"{avg_diff:.2f}" for avg_diff in average_differences]

# 결과 출력
print(results)

# 각 모델별 실제 값과 예측 값 출력
for df in actual_vs_predicted_list:
    avg_diff = np.mean(df['Difference'].astype(float))
    print(f"\nModel: {df['Model'].iloc[0]} (Average Difference: {avg_diff:.2f})")
    print(df[['OriginalIndex', 'Actual_MedicalCost', 'Predicted_Index', 'Predicted_MedicalCost', 'Difference']].head(10).to_string(index=False))
