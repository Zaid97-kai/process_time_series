import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_excel('Dataset.xlsx')
print(f"Размер исходного датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")

# Целевые переменные для прогнозирования
target_columns = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']

# 1. Разделение временного ряда на сегменты длиной 20 записей
segment_length = 20
num_segments = len(df) // segment_length
segments = []

for i in range(num_segments):
    start_idx = i * segment_length
    end_idx = (i + 1) * segment_length
    segment = df.iloc[start_idx:end_idx].copy()
    segments.append(segment)

print(f"Создано {len(segments)} сегментов по {segment_length} записей каждый")

# 2. Создание Excel файла с сегментами на разных листах
with pd.ExcelWriter('segmented_time_series.xlsx') as writer:
    for i, segment in enumerate(segments):
        segment.to_excel(writer, sheet_name=f'Segment_{i+1}', index=False)

# 3. Вычисление временных характеристик для каждого сегмента
def calculate_temporal_features(segment_df, feature_columns=['F', 'F1', 'F6']):
    """
    Вычисляет временные характеристики для сегмента
    """
    features = {}
    
    for col in feature_columns:
        if col in segment_df.columns:
            data = segment_df[col].values
            
            # Базовые статистики для каждой колонки
            prefix = f"{col}_"
            features[prefix + 'min'] = np.min(data)
            features[prefix + 'max'] = np.max(data)
            features[prefix + 'mean'] = np.mean(data)
            features[prefix + 'median'] = np.median(data)
            features[prefix + 'variance'] = np.var(data)
            features[prefix + 'skewness'] = pd.Series(data).skew()
            features[prefix + 'kurtosis'] = pd.Series(data).kurtosis()
            features[prefix + 'autocorr'] = pd.Series(data).autocorr()
            features[prefix + 'area'] = np.trapz(data)
            features[prefix + 'trend'] = np.polyfit(range(len(data)), data, 1)[0]
    
    return features

# Создание DataFrame с характеристиками
features_list = []
feature_columns_for_stats = ['F', 'F1', 'F6']

for i, segment in enumerate(segments):
    features = calculate_temporal_features(segment, feature_columns_for_stats)
    features['segment_id'] = i + 1
    features_list.append(features)

features_df = pd.DataFrame(features_list)
id_col = ['segment_id']
other_cols = [col for col in features_df.columns if col != 'segment_id']
features_df = features_df[id_col + other_cols]

# Сохранение характеристик в Excel
features_df.to_excel('temporal_features.xlsx', index=False)
print(f"Временные характеристики сохранены в 'temporal_features.xlsx'")

# 4. Кластеризация с использованием KMeans
X = features_df.drop('segment_id', axis=1).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Определение оптимального числа кластеров
inertia = []
K_range = range(1, min(11, len(X_scaled)))
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Построение графика метода локтя
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()

# Автоматическое определение оптимального k
inertia_diff = np.diff(inertia)
if len(inertia_diff) > 1:
    inertia_diff_ratio = inertia_diff[1:] / inertia_diff[:-1]
    optimal_k = np.argmin(inertia_diff_ratio) + 2
else:
    optimal_k = 3
optimal_k = max(2, min(optimal_k, 5))

print(f"Используется {optimal_k} кластеров для KMeans")

# Кластеризация
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
features_df['cluster'] = cluster_labels
features_df.to_excel('temporal_features_with_clusters.xlsx', index=False)

print("\nРаспределение сегментов по кластерам:")
print(features_df['cluster'].value_counts().sort_index())

# 5. Подготовка данных для LSTM (многомерный прогноз)
def prepare_multivariate_lstm_data(segment_df, feature_columns, target_columns, sequence_length=10):
    """
    Подготавливает многомерные данные для обучения LSTM
    """
    X_data = segment_df[feature_columns].values
    y_data = segment_df[target_columns].values
    
    X, y = [], []
    
    for i in range(len(X_data) - sequence_length):
        X.append(X_data[i:i+sequence_length])
        y.append(y_data[i+sequence_length])
    
    if len(X) == 0:
        return np.array([]), np.array([]), [], []
    
    X = np.array(X)
    y = np.array(y)
    
    # Нормализация данных
    X_scalers = []
    y_scalers = []
    
    # Нормализация признаков
    X_scaled = np.zeros_like(X)
    for feature_idx in range(X.shape[2]):
        feature_scaler = MinMaxScaler()
        feature_data = X[:, :, feature_idx].reshape(-1, 1)
        feature_scaled = feature_scaler.fit_transform(feature_data).reshape(X.shape[0], X.shape[1])
        X_scaled[:, :, feature_idx] = feature_scaled
        X_scalers.append(feature_scaler)
    
    # Нормализация целевых переменных
    y_scaled = np.zeros_like(y)
    for target_idx in range(y.shape[1]):
        target_scaler = MinMaxScaler()
        target_data = y[:, target_idx].reshape(-1, 1)
        target_scaled = target_scaler.fit_transform(target_data).flatten()
        y_scaled[:, target_idx] = target_scaled
        y_scalers.append(target_scaler)
    
    return X_scaled, y_scaled, X_scalers, y_scalers

# 6. Создание и обучение LSTM моделей для каждого кластера
lstm_models = {}
lstm_scalers_X = {}
lstm_scalers_y = {}

# Выбираем признаки для LSTM (упрощенный набор)
lstm_feature_columns = ['e1', 'e6', 'F', 'F1', 'F6']

for cluster_id in range(optimal_k):
    print(f"\n--- Обучение LSTM для кластера {cluster_id} ---")
    
    # Получаем индексы сегментов в текущем кластере
    cluster_segment_ids = features_df[features_df['cluster'] == cluster_id]['segment_id'].values
    cluster_segment_ids = [int(id) - 1 for id in cluster_segment_ids]
    
    if len(cluster_segment_ids) == 0:
        print(f"Кластер {cluster_id} пустой, пропускаем")
        continue
    
    # Собираем все данные из сегментов кластера
    X_all, y_all = [], []
    X_scalers_list, y_scalers_list = [], []
    
    for seg_id in cluster_segment_ids:
        if seg_id < len(segments):
            segment = segments[seg_id]
            X_seg, y_seg, X_scalers, y_scalers = prepare_multivariate_lstm_data(
                segment, 
                lstm_feature_columns,
                target_columns
            )
            
            if len(X_seg) > 0:
                X_all.append(X_seg)
                y_all.append(y_seg)
                X_scalers_list.append(X_scalers)
                y_scalers_list.append(y_scalers)
    
    if len(X_all) == 0:
        print(f"Нет данных для обучения кластера {cluster_id}")
        continue
    
    # Объединяем данные
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)
    
    print(f"Размер данных для обучения: X={X_all.shape}, y={y_all.shape}")
    
    # Создание модели LSTM
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_all.shape[1], X_all.shape[2])),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(target_columns))
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Обучение модели
    history = model.fit(
        X_all, y_all,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    # Сохранение модели
    model.save(f'lstm_model_cluster_{cluster_id}.keras')
    lstm_models[cluster_id] = model
    
    # Сохраняем скалеры из первого сегмента
    if X_scalers_list:
        lstm_scalers_X[cluster_id] = X_scalers_list[0]
        lstm_scalers_y[cluster_id] = y_scalers_list[0]
    
    # Построение графиков обучения - РАЗДЕЛЬНЫЕ ГРАФИКИ
    # График потерь
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - Cluster {cluster_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Прогноз на тестовых данных
    if len(X_all) > 10:
        test_idx = min(10, len(X_all))
        X_test = X_all[:test_idx]
        y_true = y_all[:test_idx]
        
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Обратное преобразование масштаба
        y_pred = np.zeros_like(y_pred_scaled)
        y_true_original = np.zeros_like(y_true)
        
        for target_idx in range(y_pred_scaled.shape[1]):
            y_pred[:, target_idx] = y_scalers_list[0][target_idx].inverse_transform(
                y_pred_scaled[:, target_idx].reshape(-1, 1)
            ).flatten()
            y_true_original[:, target_idx] = y_scalers_list[0][target_idx].inverse_transform(
                y_true[:, target_idx].reshape(-1, 1)
            ).flatten()
        
        # Графики прогнозов для каждой целевой переменной
        for i, target_col in enumerate(target_columns):
            plt.figure(figsize=(10, 6))
            plt.plot(y_true_original[:, i], label='True Values', marker='o', linewidth=2, markersize=6)
            plt.plot(y_pred[:, i], label='Predictions', marker='s', linewidth=2, markersize=6)
            plt.title(f'Predictions vs True Values - {target_col} (Cluster {cluster_id})')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'predictions_{target_col}_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Сводный график всех прогнозов
        plt.figure(figsize=(15, 10))
        n_targets = len(target_columns)
        n_cols = 3
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        for i, target_col in enumerate(target_columns):
            plt.subplot(n_rows, n_cols, i+1)
            plt.plot(y_true_original[:, i], label='True', marker='o', markersize=4, alpha=0.7)
            plt.plot(y_pred[:, i], label='Pred', marker='s', markersize=4, alpha=0.7)
            plt.title(f'{target_col}', fontsize=10)
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'All Predictions - Cluster {cluster_id}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'all_predictions_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Модель для кластера {cluster_id} обучена и сохранена")

# 7. Классификация нового временного ряда и прогнозирование
def classify_and_predict(new_segment_df, features_df, kmeans_model, scaler, lstm_models, lstm_scalers_X, lstm_scalers_y, 
                         feature_columns, target_columns, sequence_length=10):
    """
    Классифицирует новый временной ряд и делает прогноз с помощью соответствующей LSTM модели
    """
    # 1. Вычисление характеристик для нового ряда
    new_features = calculate_temporal_features(new_segment_df, feature_columns_for_stats)
    
    # 2. Подготовка данных для классификации
    feature_names = [col for col in features_df.columns if col not in ['segment_id', 'cluster']]
    X_new = np.array([new_features.get(feat, 0) for feat in feature_names]).reshape(1, -1)
    
    # 3. Масштабирование и классификация
    X_new_scaled = scaler.transform(X_new)
    cluster_label = kmeans_model.predict(X_new_scaled)[0]
    
    print(f"Новый ряд отнесен к кластеру: {cluster_label}")
    
    # 4. Проверка наличия модели для этого кластера
    if cluster_label not in lstm_models:
        print(f"Модель для кластера {cluster_label} не найдена")
        return None, None, cluster_label
    
    # 5. Подготовка данных для LSTM
    if len(new_segment_df) < sequence_length:
        print(f"Недостаточно данных для прогноза. Нужно минимум {sequence_length} точек")
        return None, None, cluster_label
    
    # Берем последние sequence_length точек
    X_data = new_segment_df[feature_columns].values[-sequence_length:]
    
    # Масштабирование признаков
    X_scaled = np.zeros((1, sequence_length, len(feature_columns)))
    for feature_idx in range(len(feature_columns)):
        feature_scaler = lstm_scalers_X[cluster_label][feature_idx]
        feature_data = X_data[:, feature_idx].reshape(-1, 1)
        feature_scaled = feature_scaler.transform(feature_data).flatten()
        X_scaled[0, :, feature_idx] = feature_scaled
    
    # 6. Прогнозирование
    y_pred_scaled = lstm_models[cluster_label].predict(X_scaled, verbose=0)
    
    # Обратное преобразование масштаба
    y_pred = np.zeros(len(target_columns))
    for target_idx in range(len(target_columns)):
        y_pred[target_idx] = lstm_scalers_y[cluster_label][target_idx].inverse_transform(
            y_pred_scaled[:, target_idx].reshape(-1, 1)
        ).flatten()[0]
    
    return y_pred, X_data[-1], cluster_label

# Пример использования для классификации и прогноза
print("\n--- Пример классификации и прогноза для нового ряда ---")

# Создаем тестовый сегмент (можно взять существующий)
test_segment_idx = 5
if test_segment_idx < len(segments):
    new_segment = segments[test_segment_idx]
    
    prediction, last_values, cluster = classify_and_predict(
        new_segment, 
        features_df, 
        kmeans, 
        scaler, 
        lstm_models, 
        lstm_scalers_X,
        lstm_scalers_y,
        lstm_feature_columns,
        target_columns
    )
    
    if prediction is not None:
        print(f"\nПрогнозируемые значения:")
        for i, target_col in enumerate(target_columns):
            print(f"  {target_col}: {prediction[i]:.6f}")
        
        # Визуализация прогнозов
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График временного ряда
        axes[0, 0].plot(new_segment['F'].values, label='F', marker='o', markersize=3)
        axes[0, 0].axvline(x=len(new_segment)-10, color='r', linestyle='--', label='Prediction Window')
        axes[0, 0].set_title(f'Time Series F (Cluster {cluster})')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # График прогнозов vs последние известные значения
        x_pos = np.arange(len(target_columns))
        width = 0.35
        last_known = [new_segment[col].iloc[-1] for col in target_columns]
        
        axes[0, 1].bar(x_pos - width/2, last_known, width, label='Last Known', alpha=0.7)
        axes[0, 1].bar(x_pos + width/2, prediction, width, label='Predicted', alpha=0.7)
        axes[0, 1].set_xlabel('Target Variables')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title(f'Predictions vs Last Known Values (Cluster {cluster})')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(target_columns, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, axis='y')
        
        # График относительной ошибки
        errors = []
        for i, col in enumerate(target_columns):
            if last_known[i] != 0:
                error = abs(prediction[i] - last_known[i]) / abs(last_known[i]) * 100
            else:
                error = abs(prediction[i]) * 100
            errors.append(error)
        
        colors = ['green' if err < 20 else 'orange' if err < 50 else 'red' for err in errors]
        axes[1, 0].bar(target_columns, errors, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=20, color='r', linestyle='--', label='20% threshold')
        axes[1, 0].set_xlabel('Target Variables')
        axes[1, 0].set_ylabel('Relative Error (%)')
        axes[1, 0].set_title('Prediction Relative Error')
        axes[1, 0].set_xticklabels(target_columns, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis='y')
        
        # График нескольких целевых переменных
        for i, col in enumerate(target_columns[:3]):
            axes[1, 1].plot(new_segment[col].values, label=col, alpha=0.7)
        axes[1, 1].set_title('Target Variables in Segment')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle(f'Prediction Results for Test Segment (Cluster {cluster})', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('new_series_multivariate_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nГрафик сохранен как 'new_series_multivariate_prediction.png'")

# 8. Дополнительная визуализация характеристик кластеров
plt.figure(figsize=(15, 10))
key_features = ['F_mean', 'F_variance', 'F_trend', 'F1_mean']

for i, feature in enumerate(key_features, 1):
    if feature in features_df.columns:
        plt.subplot(2, 2, i)
        for cluster_id in range(optimal_k):
            cluster_data = features_df[features_df['cluster'] == cluster_id][feature]
            if len(cluster_data) > 0:
                plt.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster_id}', bins=10)
        plt.title(f'Distribution of {feature} by Cluster')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Сводный отчет по кластерам
print("\n" + "="*60)
print("СВОДНЫЙ ОТЧЕТ ПО КЛАСТЕРАМ")
print("="*60)
print(f"Всего сегментов: {len(segments)}")
print(f"Число кластеров: {optimal_k}")
print("\nРаспределение сегментов по кластерам:")
cluster_stats = features_df['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_stats.items():
    percentage = count/len(segments)*100 if len(segments) > 0 else 0
    print(f"  Кластер {cluster_id}: {count} сегментов ({percentage:.1f}%)")

print(f"\nОбучено LSTM моделей: {len(lstm_models)}")
print(f"Прогнозируемые переменные: {', '.join(target_columns)}")

# 10. Создание отчета в Excel
report_data = {
    'Metric': [
        'Total Segments',
        'Number of Clusters',
        'Segment Length',
        'LSTM Sequence Length',
        'Target Variables',
        'LSTM Models Trained'
    ],
    'Value': [
        len(segments),
        optimal_k,
        segment_length,
        10,
        len(target_columns),
        len(lstm_models)
    ]
}

cluster_report = []
for cluster_id in range(optimal_k):
    count = len(features_df[features_df['cluster'] == cluster_id])
    percentage = count/len(segments)*100 if len(segments) > 0 else 0
    has_model = 'Yes' if cluster_id in lstm_models else 'No'
    cluster_report.append({
        'Cluster ID': cluster_id,
        'Segments Count': count,
        'Percentage (%)': round(percentage, 1),
        'LSTM Model': has_model
    })

report_df1 = pd.DataFrame(report_data)
report_df2 = pd.DataFrame(cluster_report)

with pd.ExcelWriter('analysis_report.xlsx') as writer:
    report_df1.to_excel(writer, sheet_name='Summary', index=False)
    report_df2.to_excel(writer, sheet_name='Cluster Distribution', index=False)
    features_df.to_excel(writer, sheet_name='Segment Features', index=False)

print("\n" + "="*60)
print("СОЗДАННЫЕ ФАЙЛЫ")
print("="*60)
files_categories = {
    "Данные": [
        "segmented_time_series.xlsx",
        "temporal_features.xlsx", 
        "temporal_features_with_clusters.xlsx",
        "analysis_report.xlsx"
    ],
    "Графики": [
        "elbow_method.png",
        "cluster_feature_distributions.png",
        "new_series_multivariate_prediction.png"
    ],
    "Модели LSTM": [f"lstm_model_cluster_{cluster_id}.keras" for cluster_id in lstm_models.keys()],
    "Графики обучения": []
}

# Добавляем графики обучения
for cluster_id in lstm_models.keys():
    files_categories["Графики обучения"].append(f"loss_cluster_{cluster_id}.png")
    files_categories["Графики обучения"].append(f"all_predictions_cluster_{cluster_id}.png")
    for target_col in target_columns:
        files_categories["Графики обучения"].append(f"predictions_{target_col}_cluster_{cluster_id}.png")

for category, files in files_categories.items():
    print(f"\n{category}:")
    for file in files[:10]:  # Показываем первые 10 файлов в каждой категории
        print(f"  - {file}")
    if len(files) > 10:
        print(f"  ... и еще {len(files) - 10} файлов")

print("\n" + "="*60)
print("СКРИПТ УСПЕШНО ВЫПОЛНЕН!")
print("="*60)