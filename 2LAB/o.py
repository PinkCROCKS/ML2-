import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Стиль графиков
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Загрузка данных
train_df = pd.read_csv('datasets/train_c.csv')
test_df = pd.read_csv('datasets/test_c.csv')

# Очистка данных
clean_train = train_df.dropna(subset=['LoanApproved']).copy()

# Анализ целевой переменной
plt.figure(figsize=(10, 6))
loan_counts = clean_train['LoanApproved'].value_counts()
colors = ['#FF6B6B', '#4ECDC4']
plt.bar(['Отказ', 'Одобрение'], loan_counts.values, color=colors, edgecolor='black')
plt.title('Распределение заявок на кредит', fontsize=16, fontweight='bold')
plt.ylabel('Количество заявок', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Всего заявок: {len(clean_train)}")
print(f"Одобрено: {loan_counts[1]} ({loan_counts[1]/len(clean_train)*100:.1f}%)")
print(f"Отклонено: {loan_counts[0]} ({loan_counts[0]/len(clean_train)*100:.1f}%)")

# Подготовка признаков
X_data = clean_train.drop(['LoanApproved', 'ApplicationDate'], axis=1, errors='ignore')
y_target = clean_train['LoanApproved'].values

test_processed = test_df.drop(['ApplicationDate'], axis=1, errors='ignore')

# Сохраняем ID для submission
if 'ID' in test_processed.columns:
    submission_ids = test_processed['ID'].values
    test_processed = test_processed.drop('ID', axis=1)
else:
    submission_ids = np.arange(len(test_processed))

if 'ID' in X_data.columns:
    X_data = X_data.drop('ID', axis=1)

# Обработка пропущенных значений
numeric_features = X_data.select_dtypes(include=[np.number]).columns
categorical_features = X_data.select_dtypes(include=['object']).columns

for col in numeric_features:
    median_val = X_data[col].median()
    X_data[col] = X_data[col].fillna(median_val)
    if col in test_processed.columns:
        test_processed[col] = test_processed[col].fillna(median_val)

for col in categorical_features:
    mode_val = X_data[col].mode()[0] if not X_data[col].mode().empty else 'Unknown'
    X_data[col] = X_data[col].fillna(mode_val)
    if col in test_processed.columns:
        test_processed[col] = test_processed[col].fillna(mode_val)

# Кодирование категориальных признаков
if len(categorical_features) > 0:
    X_data = pd.get_dummies(X_data, columns=categorical_features, drop_first=True)
    test_processed = pd.get_dummies(test_processed, columns=categorical_features, drop_first=True)
    
    # Выравнивание столбцов
    common_cols = X_data.columns.intersection(test_processed.columns)
    X_data = X_data[common_cols]
    test_processed = test_processed[common_cols]

# Разделение данных
X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_target, test_size=0.25, random_state=42, stratify=y_target
)

print(f"\nРазмеры данных:")
print(f"Обучающая выборка: {X_train.shape}")
print(f"Валидационная выборка: {X_val.shape}")
print(f"Тестовая выборка: {test_processed.shape}")

# Собственные реализации метрик
def calculate_precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def calculate_recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calculate_f1(y_true, y_pred):
    prec = calculate_precision(y_true, y_pred)
    rec = calculate_recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

# Реализация ансамблевого классификатора
class EnsembleModel:
    def __init__(self, base_model=None, n_models=10, sample_ratio=0.8, feature_ratio=0.8, seed=42):
        self.base_model = base_model if base_model else DecisionTreeClassifier(max_depth=5)
        self.n_models = n_models
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.seed = seed
        self.models = []
        self.feature_indices = []
        
    def fit(self, X, y):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        
        sample_count = int(n_samples * self.sample_ratio)
        feature_count = int(n_features * self.feature_ratio)
        
        for i in range(self.n_models):
            sample_idx = np.random.choice(n_samples, size=sample_count, replace=True)
            feature_idx = np.random.choice(n_features, size=feature_count, replace=False)
            
            model = DecisionTreeClassifier(max_depth=5, random_state=self.seed + i)
            model.fit(X[sample_idx][:, feature_idx], y[sample_idx])
            
            self.models.append(model)
            self.feature_indices.append(feature_idx)
        
        return self
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], 2))
        
        for model, feat_idx in zip(self.models, self.feature_indices):
            predictions += model.predict_proba(X[:, feat_idx])
        
        final_probs = predictions / self.n_models
        return np.argmax(final_probs, axis=1), final_probs[:, 1]

# Обучение ансамблевой модели
ensemble = EnsembleModel(n_models=30, sample_ratio=0.8, feature_ratio=0.7, seed=42)
y_pred_ensemble, y_proba_ensemble = ensemble.fit(X_train.values, y_train).predict(X_val.values)

# Обучение градиентного бустинга
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_val)
y_proba_gb = gb_model.predict_proba(X_val)[:, 1]

# Сравнение моделей
print("\n" + "="*60)
print("Сравнение качества моделей")
print("="*60)

results_comparison = pd.DataFrame({
    'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'],
    'Ансамбль (самост.)': [
        accuracy_score(y_val, y_pred_ensemble),
        calculate_precision(y_val, y_pred_ensemble),
        calculate_recall(y_val, y_pred_ensemble),
        calculate_f1(y_val, y_pred_ensemble),
        roc_auc_score(y_val, y_proba_ensemble)
    ],
    'Градиентный бустинг (sklearn)': [
        accuracy_score(y_val, y_pred_gb),
        precision_score(y_val, y_pred_gb),
        recall_score(y_val, y_pred_gb),
        f1_score(y_val, y_pred_gb),
        roc_auc_score(y_val, y_proba_gb)
    ]
})

print(results_comparison.round(4))

# Выбор лучшей модели
if roc_auc_score(y_val, y_proba_gb) > roc_auc_score(y_val, y_proba_ensemble):
    print("\nВыбрана модель градиентного бустинга")
    final_model = gb_model
    use_gb = True
else:
    print("\nВыбрана ансамблевая модель")
    final_model = ensemble
    use_gb = False

# Обучение финальной модели на всех данных
if use_gb:
    final_model.fit(X_data, y_target)
    test_predictions = final_model.predict(test_processed)
else:
    final_model.fit(X_data.values, y_target)
    test_predictions, _ = final_model.predict(test_processed.values)

# Создание submission файла
submission_result = pd.DataFrame({
    'ID': submission_ids,
    'LoanApproved': test_predictions
})

submission_result.to_csv('credit_predictions.csv', index=False)

print(f"\nФайл с предсказаниями сохранен: credit_predictions.csv")
print(f"Размер файла: {submission_result.shape}")
print(f"Распределение предсказаний:")
print(submission_result['LoanApproved'].value_counts())

# Визуализация результатов
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Распределение предсказаний
pred_counts = submission_result['LoanApproved'].value_counts()
axes[0].bar(['Отказ', 'Одобрение'], pred_counts.values, color=['#FF9AA2', '#A0E7E5'], edgecolor='black')
axes[0].set_title('Распределение предсказаний', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Количество', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(pred_counts.values):
    axes[0].text(i, v + max(pred_counts.values)*0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')

# ROC кривая для валидации
if use_gb:
    val_proba = final_model.predict_proba(X_val)[:, 1]
else:
    _, val_proba = final_model.predict(X_val.values)

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_val, val_proba)
roc_auc = roc_auc_score(y_val, val_proba)

axes[1].plot(fpr, tpr, color='#6A0572', linewidth=2, label=f'ROC кривая (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайная модель')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('Качество модели на валидации', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\nАнализ завершен. Файл с предсказаниями готов к отправке.")