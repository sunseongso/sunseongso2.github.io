```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
```


```python
test = test_data.copy()
```


```python
#훈련셋 맨위의 몇행 확인
train_data.head()
```


```python
# PassengerId열을 인덱스 열로 명시적으로 설정한다
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")
```


```python
#데이터가 어느정도 누락되었는지 확인
train_data.info()
```


```python
train_data[train_data["Sex"]=="female"]["Age"].median()
```


```python
#수치속성 살펴보기
train_data.describe()
```


```python
#타겟이 0 또는 1인 정보 확인해보기
train_data["Survived"].value_counts()
```

범주적 속성 살펴보기


```python
train_data["Pclass"].value_counts()
```


```python
train_data["Sex"].value_counts()
```


```python
train_data["Embarked"].value_counts()
```


```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
```

범주적 속성을 위한 파이프라인 구축


```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
```


```python
cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),    
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
```


```python
#마지막으로, 수치 및 범주형 파이프라인 결합
from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])
```


```python
X_train = preprocess_pipeline.fit_transform(train_data)
X_train
```


```python
y_train = train_data["Survived"]

```


```python
#분류기를 훈련할 준비가 끝났으니 RandomForestClassifier부터 시작해보기
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
```


```python
#모델 훈련한 것으로 테스트셋 예측에 활용하기
X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)

```

예측 결과를 csv파일로 만들어야함


```python
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
```


```python
test.info()
```


```python
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('./submission.csv', index=False)
```
