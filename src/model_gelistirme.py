import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

def load_data():
    """Özellik mühendisliği yapılmış veriyi yükle"""
    df = pd.read_csv("data/processed/titanic_features.csv")
    return df

def prepare_features(df):
    """Modelleme için özellikleri hazırla"""
    # Hedef değişken
    y = df['survived']
    
    # Özellikler (sayısal olanları seç)
    feature_columns = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'is_alone']
    
    # Kategorik özellikler de dahil edilebilir
    if 'sex' in df.columns:
        feature_columns.append('sex')
    if 'embarked' in df.columns:
        feature_columns.append('embarked')
    if 'age_group' in df.columns:
        feature_columns.append('age_group')
    if 'fare_group' in df.columns:
        feature_columns.append('fare_group')
    if 'title' in df.columns:
        feature_columns.append('title')
    
    # Mevcut sütunları filtrele
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features].fillna(df[available_features].mean())
    
    print(f"Kullanılan özellikler: {available_features}")
    
    return X, y, available_features

def train_models(X_train, y_train):
    """Farklı modelleri eğit"""
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(random_state=42, probability=True),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'naive_bayes': GaussianNB()
    }
    
    trained_models = {}
    
    # Özellik ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    for name, model in models.items():
        print(f"{name} modeli eğitiliyor...")
        
        # SVM ve Logistic Regression için ölçeklendirilmiş veri kullan
        if name in ['svm', 'logistic_regression']:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        
        trained_models[name] = model
    
    # Scaler'ı da kaydet
    trained_models['scaler'] = scaler
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Modelleri değerlendir"""
    results = {}
    scaler = models.pop('scaler')  # Scaler'ı modeller sözlüğünden çıkar
    
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        print(f"{name} modeli değerlendiriliyor...")
        
        # Uygun veri setini seç
        if name in ['svm', 'logistic_regression']:
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"{name} - Doğruluk: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    # Scaler'ı geri ekle
    models['scaler'] = scaler
    
    return results

def save_models_and_results(models, results, feature_names):
    """Modelleri ve sonuçları kaydet"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Modelleri kaydet
    for name, model in models.items():
        joblib.dump(model, f"models/{name}.pkl")
    
    # Sonuçları kaydet
    with open("results/model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Özellik isimlerini kaydet
    with open("results/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    
    # En iyi modeli belirle (ROC-AUC'ye göre)
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    best_model = models[best_model_name]
    joblib.dump(best_model, "models/best_model.pkl")
    
    # En iyi model bilgisini kaydet
    best_model_info = {
        'best_model': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'roc_auc': results[best_model_name]['roc_auc'],
        'features_used': feature_names
    }
    
    with open("results/best_model_info.json", "w") as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"\nEn iyi model: {best_model_name}")
    print(f"Doğruluk: {results[best_model_name]['accuracy']:.4f}")
    print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

def main():
    print("Model geliştirme başlatılıyor...")
    
    # Veriyi yükle
    df = load_data()
    
    # Özellikleri hazırla
    X, y, feature_names = prepare_features(df)
    
    # Eğitim ve test setlerine böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    print(f"Hedef değişken dağılımı - Eğitim: {y_train.value_counts().to_dict()}")
    print(f"Hedef değişken dağılımı - Test: {y_test.value_counts().to_dict()}")
    
    # Modelleri eğit
    trained_models = train_models(X_train, y_train)
    
    # Modelleri değerlendir
    results = evaluate_models(trained_models, X_test, y_test)
    
    # Sonuçları kaydet
    save_models_and_results(trained_models, results, feature_names)
    
    print("\nModel geliştirme tamamlandı!")

if __name__ == "__main__":
    main()