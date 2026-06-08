import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier  # Ta linijka naprawia Twój NameError

# Importujemy nasze własne moduły lokalne
from data_handler import load_and_preprocess_data, URL_TRAIN, URL_TEST
from models import train_baseline_model, train_tuned_random_forest

def evaluate_pipeline(model_name: str, model: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = model.predict(X_test)
    print(f"\n--- Wyniki ewaluacji dla: {model_name} ---")
    print(f"Dokładność (Accuracy):        {accuracy_score(y_test, predictions):.4f}")
    print(f"Precyzja (Precision):         {precision_score(y_test, predictions):.4f}")
    print(f"Czułość / Pełność (Recall):   {recall_score(y_test, predictions):.4f}")
    print(f"F1-Score:                     {f1_score(y_test, predictions):.4f}")

def plot_results(rf_model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    print("\n[4/4] Generowanie wizualizacji ")
    
    rf_predictions = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, rf_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ruch Normalny', 'Cyberatak'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Macierz Błędów - Zoptymalizowany Las Losowy")
    plt.show()

    importances = rf_model.feature_importances_
    forest_importances = pd.Series(importances, index=X_test.columns).sort_values(ascending=False)
    top_10_features = forest_importances.head(10)

    plt.figure(figsize=(10, 6))
    top_10_features.plot.bar(color='teal')
    plt.title("Top 10 kluczowych cech ruchu sieciowego (Random Forest)")
    plt.ylabel("Stopień ważności cechy (Gini Impurity)")
    plt.xlabel("Metryki sieciowe")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data(URL_TRAIN, URL_TEST)
    
    baseline_tree = train_baseline_model(X_train, y_train)
    optimized_rf = train_tuned_random_forest(X_train, y_train)
    
    evaluate_pipeline("Drzewo Decyzyjne (Baseline)", baseline_tree, X_test, y_test)
    evaluate_pipeline("Zoptymalizowany Las Losowy", optimized_rf, X_test, y_test)
    
    plot_results(optimized_rf, X_test, y_test)