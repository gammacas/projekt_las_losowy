import pandas as pd
import matplotlib.pyplot as plt

from data_handler import KOLUMNY, URL_TRAIN, URL_TEST

def plot_top_attacks() -> None:
    print("\n[INFO] Pobieranie danych do wykresów...")
    df_train = pd.read_csv(URL_TRAIN, header=None, names=KOLUMNY)
    df_test = pd.read_csv(URL_TEST, header=None, names=KOLUMNY)
    
    train_attacks = df_train[df_train['label'] != 'normal']['label']
    test_attacks = df_test[df_test['label'] != 'normal']['label']
    
    top_train = train_attacks.value_counts().head(10)
    top_test = test_attacks.value_counts().head(10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))


    top_train.plot(kind='bar', color='crimson', ax=axes[0])
    axes[0].set_title('Top 10 najczęstszych ataków - Zbiór Treningowy', fontsize=14)
    axes[0].set_ylabel('Liczba wystąpień')
    axes[0].set_xlabel('Nazwa ataku')
    axes[0].tick_params(axis='x', rotation=45)

    # Wykres dla zbioru testowego
    top_test.plot(kind='bar', color='darkorange', ax=axes[1])
    axes[1].set_title('Top 10 najczęstszych ataków - Zbiór Testowy', fontsize=14)
    axes[1].set_ylabel('Liczba wystąpień')
    axes[1].set_xlabel('Nazwa ataku')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def print_descriptive_statistics() -> None:
    print("\n--- Statystyki opisowe dla kluczowych cech numerycznych (Zbiór Treningowy) ---")
    features_to_analyze = ['src_bytes', 'dst_bytes', 'duration', 'count', 'dst_host_srv_count']
    
    df_train = pd.read_csv(URL_TRAIN, header=None, names=KOLUMNY)
    
    stats_df = pd.DataFrame()
    for feature in features_to_analyze:
        stats_df[feature] = [
            df_train[feature].mean(),
            df_train[feature].median(),
            df_train[feature].var(),
            df_train[feature].quantile(0.25),
            df_train[feature].quantile(0.75)
        ]
        
    stats_df.index = ['Średnia', 'Mediana', 'Wariancja', 'Kwartyl dolny (Q1)', 'Kwartyl górny (Q3)']
    print(stats_df.round(2).to_string())
    print("--------------------------------------------------------------------------------\n")

if __name__ == "__main__":

    print_descriptive_statistics()
    plot_top_attacks()