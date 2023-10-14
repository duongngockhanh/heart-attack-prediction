import pandas as pd

def get_dataframe():
    # Đọc tập dữ liệu
    df = pd.read_csv('cleveland.csv', header=None)
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df['target'] = df['target'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['thal'] = df['thal'].fillna(df['thal'].mean())
    df['ca'] = df['ca'].fillna(df['ca'].mean())
    return df