# automate_Gina.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_pipeline(input_path='dataset_raw/car_data.csv', output_dir='preprocessing/car_preprocessing'):
    # Buat folder output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)

    # 1. Label Encoding kolom kategorikal
    le = LabelEncoder()
    if df['Gender'].dtype == 'object':
        df['Gender'] = le.fit_transform(df['Gender'])
        joblib.dump(le, os.path.join(output_dir, 'label_encoder_gender.pkl'))

    # 2. Hapus kolom User ID
    if 'User ID' in df.columns:
        df.drop(columns=['User ID'], inplace=True)

    # 3. Normalisasi fitur
    scaler = MinMaxScaler()
    X = df.drop(columns='Purchased')
    y = df['Purchased']
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # 4. Oversampling dengan SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Simpan data hasil preprocessing
    pd.DataFrame(X_train, columns=X.columns).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    pd.DataFrame(y_train, columns=['Purchased']).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    pd.DataFrame(y_test, columns=['Purchased']).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    print("[INFO] Preprocessing selesai dan data disimpan di:", output_dir)

if __name__ == '__main__':
    preprocess_pipeline()
