
# 특성을 기반으로 한 의료비 예측

이 저장소는 여러가지 특성 데이터를 기반으로 의료비를 예측하는 머신 러닝 프로젝트를 포함하고 있습니다. 여러 회귀 모델을 사용하여 의료비를 예측하고 그 성능을 평가합니다.

## 목차

- [소개](#소개)
- [데이터 전처리](#데이터-전처리)
- [모델 학습 및 평가](#모델-학습-및-평가)
- [결과](#결과)
- [필수 패키지](#필수-패키지)
- [사용법](#사용법)
- [기여](#기여)
- [라이센스](#라이센스)

## 소개

이 프로젝트의 목표는 특성 데이터에 기반하여 개인이 발생시킬 의료비를 예측하는 것입니다. 사용된 데이터셋은 `insurance.csv` 파일로, 나이, BMI, 자녀 수 등과 같은 특성을 포함하고 있습니다.

## 데이터 전처리

1. **데이터 로드**:
    - `pandas`를 사용하여 CSV 파일에서 데이터셋을 로드합니다.

    ```python
    data = pd.read_csv('data/insurance.csv')
    ```

2. **이상치 제거**:
    - Z-스코어를 사용하여 `MedicalCost` 변수의 이상치를 제거합니다.

    ```python
    z_scores = np.abs(stats.zscore(data['MedicalCost']))
    data_cleaned = data[z_scores < 3].copy()
    ```

3. **결측값 처리**:
    - 일부 `MedicalCost` 데이터를 결측값으로 설정하여 결측값을 시뮬레이션합니다.

    ```python
    missing_rate = 0.2
    n_missing = int(len(data_cleaned) * missing_rate)
    rng = np.random.RandomState(42)
    missing_indices = rng.choice(data_cleaned.index, n_missing, replace=False)
    original_values_sample = data_cleaned.loc[missing_indices, ['OriginalIndex', 'MedicalCost']].sort_values('OriginalIndex')
    data_cleaned.loc[missing_indices, 'MedicalCost'] = np.nan
    ```

4. **데이터 분리**:
    - `MedicalCost` 열에 결측값을 생성하여, 테스트 데이터 세트(MedicalCost 삭제) 와 학습 데이터 세트(MedicalCost 있음)를 분리합니다.

    ```python
    data_train = data_cleaned.dropna(subset=['MedicalCost'])
    data_test = data_cleaned[data_cleaned['MedicalCost'].isna()]
    ```

5. **특성 인코딩 및 스케일링**:
    - 범주형 변수는 원-핫 인코딩을 사용하고, 수치형 변수는 스케일링을 적용합니다.

    ```python
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    scaler = StandardScaler()
    X_train[['Age', 'BMI', 'Children']] = scaler.fit_transform(X_train[['Age', 'BMI', 'Children']])
    X_test[['Age', 'BMI', 'Children']] = scaler.transform(X_test[['Age', 'BMI', 'Children']])
    ```

## 모델 학습 및 평가

GridSearchCV를 사용하여 최적의 하이퍼파라미터를 찾고 다양한 회귀 모델을 학습 및 평가합니다. 사용된 모델은 다음과 같습니다:

- 선형 회귀
- 랜덤 포레스트
- XGBoost
- 그래디언트 부스팅
- 리지 회귀
- 라쏘 회귀
- 엘라스틱넷
- 서포트 벡터 회귀 (SVR)
- K-최근접 이웃 (KNN)
- 의사결정 나무
- 에이다부스트
- 엑스트라 트리

```python
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    ...
}

param_grids = {
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    ...
}
```

모델은 MSE(평균 제곱 오차), MAE(평균 절대 오차), R2(결정 계수)와 같은 지표를 사용하여 평가됩니다.

## 결과

각 모델의 성능은 데이터프레임에 요약되고 실제 값과 예측 값을 비교한 결과가 출력됩니다.

```python
results = pd.DataFrame(columns=['Model', 'R2', 'MSE', 'MAE', 'Average_Difference'])
print(results)
```

## 필수 패키지

이 프로젝트를 실행하려면 다음 Python 라이브러리가 필요합니다:

- pandas
- numpy
- scikit-learn
- xgboost
- scipy

pip을 사용하여 패키지를 설치할 수 있습니다:

```sh
pip install pandas numpy scikit-learn xgboost scipy
```

## 사용법

1. 저장소를 클론합니다:

    ```sh
    git clone https://github.com/yourusername/Insurance_MedicalCost_Forecast.git
    ```

2. 프로젝트 디렉토리로 이동합니다:

    ```sh
    cd Insurance_MedicalCost_Forecast
    ```

3. 스크립트를 실행합니다:

    ```sh
    python Insurance_MedicalCost_Forecast.py
    ```

## 기여

기여는 언제나 환영합니다! 개선 사항이나 버그 수정을 위해 이슈를 열거나 풀 리퀘스트를 제출해 주세요.

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.
