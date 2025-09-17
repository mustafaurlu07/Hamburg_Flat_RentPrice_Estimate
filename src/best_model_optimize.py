import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import time

def load_data():
    """Özellik mühendisliği yapılmış veriyi yükle"""
    df = pd.read_csv("data/processed/titanic_features.csv")
    return df

def prepare_features(df):
    """Modelleme için özellikleri hazırla"""
    y = df['survived']
    
    feature_columns = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'is_alone']
    
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
    
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features].fillna(df[available_features].mean())
    
    return X, y, available_features

def identify_best_model():
    """En iyi modeli belirle"""
    
    # Önceki sonuçları kontrol et
    if os.path.exists("results/model_results.json"):
        with open("results/model_results.json", "r") as f:
            results = json.load(f)
        
        # ROC-AUC'ye göre en iyi modeli bul
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        best_score = results[best_model_name]['roc_auc']
        
        print(f"📊 Önceki sonuçlara göre en iyi model: {best_model_name}")
        print(f"🎯 ROC-AUC skoru: {best_score:.4f}")
        
        return best_model_name, best_score
    else:
        print("❌ Önceki model sonuçları bulunamadı!")
        return None, None

def optimize_random_forest_detailed(X, y, cv_folds=5):
    """Random Forest için detaylı optimizasyon"""
    print("🌲 Random Forest detaylı optimizasyonu...")
    
    # Geniş parametre aralığı
    param_grid = {
        'n_estimators': [100, 200, 300, 500, 700],
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'max_leaf_nodes': [None, 50, 100, 200]
    }
    
    # İlk aşama: RandomizedSearch
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_grid, n_iter=100, cv=cv_folds,
        scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42
    )
    
    start_time = time.time()
    random_search.fit(X, y)
    phase1_time = time.time() - start_time
    
    print(f"⚡ Random Search tamamlandı: {phase1_time:.2f}s")
    print(f"🎯 En iyi skor: {random_search.best_score_:.4f}")
    
    # İkinci aşama: En iyi parametreler etrafında GridSearch
    best_params = random_search.best_params_
    
    # En iyi parametreler etrafında dar aralık
    refined_grid = {}
    for param, value in best_params.items():
        if param == 'n_estimators':
            refined_grid[param] = [max(50, value-100), value, value+100]
        elif param == 'max_depth' and value is not None:
            refined_grid[param] = [max(3, value-2), value, value+2]
        elif param == 'min_samples_split':
            refined_grid[param] = [max(2, value-2), value, value+2]
        elif param == 'min_samples_leaf':
            refined_grid[param] = [max(1, value-1), value, value+1]
        else:
            refined_grid[param] = [value]  # Sabit tut
    
    print("🔍 Refined Grid Search başlatılıyor...")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), refined_grid,
        cv=cv_folds, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X, y)
    phase2_time = time.time() - start_time
    
    print(f"⚡ Grid Search tamamlandı: {phase2_time:.2f}s")
    print(f"🎯 Final skor: {grid_search.best_score_:.4f}")
    
    total_time = phase1_time + phase2_time
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, total_time

def optimize_gradient_boosting_detailed(X, y, cv_folds=5):
    """Gradient Boosting için detaylı optimizasyon"""
    print("🚀 Gradient Boosting detaylı optimizasyonu...")
    
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 5, 7, 9, 12],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'subsample': [0.8, 0.9, 0.95, 1.0],
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        gb, param_grid, n_iter=80, cv=cv_folds,
        scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42
    )
    
    start_time = time.time()
    random_search.fit(X, y)
    optimization_time = time.time() - start_time
    
    print(f"⚡ Optimizasyon tamamlandı: {optimization_time:.2f}s")
    print(f"🎯 En iyi skor: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_, optimization_time

def optimize_logistic_regression_detailed(X, y, cv_folds=5):
    """Logistic Regression için detaylı optimizasyon"""
    print("📈 Logistic Regression detaylı optimizasyonu...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'max_iter': [1000, 2000, 3000],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    lr = LogisticRegression(random_state=42)
    random_search = RandomizedSearchCV(
        lr, param_grid, n_iter=100, cv=cv_folds,
        scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42
    )
    
    start_time = time.time()
    random_search.fit(X_scaled, y)
    optimization_time = time.time() - start_time
    
    print(f"⚡ Optimizasyon tamamlandı: {optimization_time:.2f}s")
    print(f"🎯 En iyi skor: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_, scaler, optimization_time

def final_evaluation(model, X, y, model_name, scaler=None, cv_folds=10):
    """Final değerlendirme (daha fazla CV fold ile)"""
    print(f"\n🎯 {model_name} final değerlendirmesi ({cv_folds} fold CV)...")
    
    if scaler:
        X_eval = scaler.transform(X)
    else:
        X_eval = X
    
    # Daha detaylı cross-validation
    cv_accuracy = cross_val_score(model, X_eval, y, cv=cv_folds, scoring='accuracy')
    cv_roc_auc = cross_val_score(model, X_eval, y, cv=cv_folds, scoring='roc_auc')
    cv_precision = cross_val_score(model, X_eval, y, cv=cv_folds, scoring='precision')
    cv_recall = cross_val_score(model, X_eval, y, cv=cv_folds, scoring='recall')
    cv_f1 = cross_val_score(model, X_eval, y, cv=cv_folds, scoring='f1')
    
    results = {
        'cv_accuracy_mean': float(cv_accuracy.mean()),
        'cv_accuracy_std': float(cv_accuracy.std()),
        'cv_roc_auc_mean': float(cv_roc_auc.mean()),
        'cv_roc_auc_std': float(cv_roc_auc.std()),
        'cv_precision_mean': float(cv_precision.mean()),
        'cv_precision_std': float(cv_precision.std()),
        'cv_recall_mean': float(cv_recall.mean()),
        'cv_recall_std': float(cv_recall.std()),
        'cv_f1_mean': float(cv_f1.mean()),
        'cv_f1_std': float(cv_f1.std())
    }
    
    print(f"📊 Final Sonuçlar:")
    print(f"   Accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std()*2:.4f})")
    print(f"   ROC-AUC:  {cv_roc_auc.mean():.4f} (±{cv_roc_auc.std()*2:.4f})")
    print(f"   Precision: {cv_precision.mean():.4f} (±{cv_precision.std()*2:.4f})")
    print(f"   Recall:   {cv_recall.mean():.4f} (±{cv_recall.std()*2:.4f})")
    print(f"   F1-Score: {cv_f1.mean():.4f} (±{cv_f1.std()*2:.4f})")
    
    return results

def main():
    print("🎯 En İyi Model Detaylı Optimizasyonu")
    print("="*50)
    
    # Veriyi yükle
    df = load_data()
    X, y, feature_names = prepare_features(df)
    
    print(f"📊 Veri boyutu: {X.shape}")
    print(f"🔧 Özellikler: {feature_names}")
    
    # En iyi modeli belirle
    best_model_name, baseline_score = identify_best_model()
    
    if not best_model_name:
        print("❌ Önce temel modelleri çalıştırın!")
        return
    
    cv_folds = 5
    optimized_model = None
    best_params = None
    optimization_time = None
    scaler = None
    
    # En iyi modeli detaylı optimize et
    if best_model_name == 'random_forest':
        optimized_model, best_params, best_score, optimization_time = optimize_random_forest_detailed(X, y, cv_folds)
    elif best_model_name == 'gradient_boosting':
        optimized_model, best_params, best_score, optimization_time = optimize_gradient_boosting_detailed(X, y, cv_folds)
    elif best_model_name == 'logistic_regression':
        optimized_model, best_params, best_score, scaler, optimization_time = optimize_logistic_regression_detailed(X, y, cv_folds)
    else:
        print(f"❌ {best_model_name} için detaylı optimizasyon henüz desteklenmiyor!")
        return
    
    print(f"\n🚀 Optimizasyon Sonuçları:")
    print(f"   Baseline: {baseline_score:.4f}")
    print(f"   Optimized: {best_score:.4f}")
    print(f"   Gelişim: {((best_score - baseline_score) / baseline_score * 100):+.2f}%")
    
    # Final değerlendirme
    final_results = final_evaluation(optimized_model, X, y, best_model_name, scaler, cv_folds=10)
    
    # Sonuçları dosyaya kaydet
    os.makedirs("models/detailed_optimized", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    if scaler:
        joblib.dump({'model': optimized_model, 'scaler': scaler}, 
                   f"models/detailed_optimized/{best_model_name}_detailed.pkl")
    else:
        joblib.dump(optimized_model, f"models/detailed_optimized/{best_model_name}_detailed.pkl")
    
    # Sonuç raporu
    detailed_results = {
        'model_name': best_model_name,
        'baseline_score': baseline_score,
        'optimized_score': best_score,
        'improvement_percent': (best_score - baseline_score) / baseline_score * 100,
        'optimization_time': optimization_time,
        'best_parameters': best_params,
        'final_evaluation': final_results,
        'features_used': feature_names
    }
    
    with open("results/detailed_optimization_report.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n✅ Detaylı optimizasyon tamamlandı!")
    print(f"📁 Model kaydedildi: models/detailed_optimized/{best_model_name}_detailed.pkl")
    print(f"📊 Rapor: results/detailed_optimization_report.json")

if __name__ == "__main__":
    main()