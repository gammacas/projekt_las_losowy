import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

KOLUMNY = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

URL_TRAIN = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
URL_TEST = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv"

def load_and_preprocess_data(url_train: str, url_test: str) -> tuple:
    print("[1/4] Pobieranie i przygotowanie danych")
    df_train = pd.read_csv(url_train, header=None, names=KOLUMNY)
    df_test = pd.read_csv(url_test, header=None, names=KOLUMNY)
    
    print(f"Liczba unikalnych ataków: TRENINGOWY = {df_train[df_train['label'] != 'normal']['label'].nunique()}, TESTOWY = {df_test[df_test['label'] != 'normal']['label'].nunique()}")

    print("\n--- Top 10 najczęstszych ataków ---")
    train_attacks = df_train[df_train['label'] != 'normal']['label']
    test_attacks = df_test[df_test['label'] != 'normal']['label']
    
    print("Zbiór TRENINGOWY:")
    print(train_attacks.value_counts().head(10).to_string())
    print("\nZbiór TESTOWY:")
    print(test_attacks.value_counts().head(10).to_string())
    print("-----------------------------------\n")
    
    df_train = df_train.drop(columns=['difficulty'])
    df_test = df_test.drop(columns=['difficulty'])

    y_train = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
    y_test = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

    X_train_raw = df_train.drop(columns=['label'])
    X_test_raw = df_test.drop(columns=['label'])


    kat_cechy = ['protocol_type', 'service', 'flag']
    X_train_encoded = pd.get_dummies(X_train_raw, columns=kat_cechy, dtype=int)
    X_test_encoded = pd.get_dummies(X_test_raw, columns=kat_cechy, dtype=int)

    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    X_train_encoded.columns = X_train_encoded.columns.astype(str)
    X_test_encoded.columns = X_test_encoded.columns.astype(str)

    num_cechy = [col for col in X_train_raw.columns if col not in kat_cechy]

    scaler = StandardScaler()
    X_train_encoded[num_cechy] = scaler.fit_transform(X_train_encoded[num_cechy])
    X_test_encoded[num_cechy] = scaler.transform(X_test_encoded[num_cechy])

    return X_train_encoded, X_test_encoded, y_train, y_test

