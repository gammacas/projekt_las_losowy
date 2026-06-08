import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:

    print("\n [2/4] Trenowanie modelu bazowego (Decision Tree)...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

def train_tuned_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:

    print("[3/4] model Lasu Losowego przy użyciu GridSearchCV...")
    rf_base = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Najlepsze odnalezione hiperparametry: {grid_search.best_params_}")
    return grid_search.best_estimator_