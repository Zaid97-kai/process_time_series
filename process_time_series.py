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

# ================================================
# ФУНКЦИЯ СГЛАЖИВАНИЯ ДАННЫХ
# ================================================
def smooth_data(df, window_size=3, method='moving_average'):
    """
    Сглаживание данных для уменьшения шума и скачков
    """
    df_smoothed = df.copy()
    
    # Список колонок для сглаживания (исключаем целевые переменные)
    columns_to_smooth = [col for col in df.columns if col not in target_columns]
    
    for col in columns_to_smooth:
        try:
            if method == 'moving_average':
                # Простое скользящее среднее
                df_smoothed[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
            
            elif method == 'savitzky_golay':
                # Фильтр Савицкого-Голея (требует scipy)
                try:
                    from scipy.signal import savgol_filter
                    df_smoothed[col] = savgol_filter(df[col], window_size, 2)
                except ImportError:
                    print("Для метода savitzky_golay требуется scipy. Используется moving_average.")
                    df_smoothed[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
            
            elif method == 'exponential':
                # Экспоненциальное сглаживание
                df_smoothed[col] = df[col].ewm(span=window_size, adjust=False).mean()
            
            # Для целевых переменных используем меньший уровень сглаживания
            if col in target_columns:
                df_smoothed[col] = df[col].rolling(window=2, center=True, min_periods=1).mean()
        
        except Exception as e:
            print(f"Ошибка при сглаживании колонки {col}: {e}")
            df_smoothed[col] = df[col]
    
    # Заполняем NaN значения (если остались)
    df_smoothed = df_smoothed.fillna(method='ffill').fillna(method='bfill')
    
    return df_smoothed

# Визуализация сглаживания для ключевых переменных
def visualize_smoothing(original_df, smoothed_df, columns_to_show=['F', 'F1', 'F6', 'a0'], save_path='smoothing_comparison.png'):
    """
    Визуализация сравнения оригинальных и сглаженных данных
    """
    n_plots = min(len(columns_to_show), 4)  # Максимум 4 графика
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4*n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, col in enumerate(columns_to_show[:n_plots]):
        if col in original_df.columns and col in smoothed_df.columns:
            axes[idx].plot(original_df[col].values, label='Original', alpha=0.7, linewidth=1)
            axes[idx].plot(smoothed_df[col].values, label='Smoothed', alpha=0.9, linewidth=1.5)
            axes[idx].set_title(f'Сравнение сглаживания: {col}', fontsize=12)
            axes[idx].set_xlabel('Индекс')
            axes[idx].set_ylabel('Значение')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сравнения сглаживания сохранен как '{save_path}'")

# Применяем сглаживание
print("\n--- Применение сглаживания данных ---")
df_smoothed = smooth_data(df, window_size=3, method='moving_average')

# Визуализируем результат сглаживания
visualize_smoothing(df, df_smoothed)

# ================================================
# ПРОДОЛЖЕНИЕ ОСНОВНОГО КОДА СО СГЛАЖЕННЫМИ ДАННЫМИ
# ================================================

# 1. Разделение ВРЕМЕННОГО РЯДА на сегменты длиной 25 записей (используем сглаженные данные)
segment_length = 12
num_segments = len(df_smoothed) // segment_length
segments = []

for i in range(num_segments):
    start_idx = i * segment_length
    end_idx = (i + 1) * segment_length
    segment = df_smoothed.iloc[start_idx:end_idx].copy()
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
lstm_mape_results = {}  # Для хранения результатов MAPE
lstm_prediction_details = {}  # Для хранения деталей прогнозов

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
        batch_size=32,
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
    
    # Прогноз на тестовых данных и расчет MAPE
    if len(X_all) > 10:
        test_idx = min(20, len(X_all))
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
        
        # Расчет ошибок
        errors = y_pred - y_true_original
        absolute_errors = np.abs(errors)
        
        # Расчет MAPE для каждой целевой переменной
        mape_values = []
        for target_idx in range(len(target_columns)):
            true_vals = y_true_original[:, target_idx]
            pred_vals = y_pred[:, target_idx]
            
            # Избегаем деления на ноль
            mask = true_vals != 0
            if np.any(mask):
                mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100
            else:
                mape = np.mean(np.abs(pred_vals)) * 100
            mape_values.append(mape)
        
        lstm_mape_results[cluster_id] = mape_values
        
        # Сохраняем детали прогнозов для кластера
        lstm_prediction_details[cluster_id] = {
            'y_true': y_true_original,
            'y_pred': y_pred,
            'errors': errors,
            'absolute_errors': absolute_errors
        }
        
        # 1. Графики прогнозов для каждой целевой переменной
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
        
        # 2. Графики ошибок для каждой целевой переменной
        for i, target_col in enumerate(target_columns):
            plt.figure(figsize=(10, 6))
            plt.plot(errors[:, i], label='Error (Pred - True)', marker='o', linewidth=2, markersize=6, color='red')
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            plt.fill_between(range(len(errors[:, i])), 0, errors[:, i], 
                            where=errors[:, i] > 0, alpha=0.3, color='green', label='Overestimation')
            plt.fill_between(range(len(errors[:, i])), 0, errors[:, i], 
                            where=errors[:, i] < 0, alpha=0.3, color='red', label='Underestimation')
            plt.title(f'Prediction Errors - {target_col} (Cluster {cluster_id})')
            plt.xlabel('Sample Index')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'errors_{target_col}_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Сводный график абсолютных ошибок для всех переменных
        plt.figure(figsize=(15, 10))
        n_targets = len(target_columns)
        n_cols = 3
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        for i, target_col in enumerate(target_columns):
            plt.subplot(n_rows, n_cols, i+1)
            plt.bar(range(len(absolute_errors[:, i])), absolute_errors[:, i], 
                   color='skyblue', alpha=0.7, label='Absolute Error')
            plt.axhline(y=np.mean(absolute_errors[:, i]), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(absolute_errors[:, i]):.4f}')
            plt.title(f'{target_col} - Absolute Errors', fontsize=10)
            plt.xlabel('Sample')
            plt.ylabel('Absolute Error')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Absolute Prediction Errors - Cluster {cluster_id}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'absolute_errors_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Гистограмма распределения ошибок
        plt.figure(figsize=(12, 8))
        
        for i, target_col in enumerate(target_columns):
            plt.subplot(3, 3, i+1)
            plt.hist(errors[:, i], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            plt.axvline(x=np.mean(errors[:, i]), color='green', linestyle='-', 
                       linewidth=2, label=f'Mean: {np.mean(errors[:, i]):.4f}')
            plt.title(f'{target_col}', fontsize=9)
            plt.xlabel('Error')
            plt.ylabel('Frequency')
            plt.legend(fontsize=7)
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Error Distribution - Cluster {cluster_id}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'error_distribution_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Сводный график всех прогнозов
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

# 7. Создание подробных таблиц с прогнозами и ошибками
print("\n--- Создание таблиц с детальными прогнозами ---")

for cluster_id in lstm_prediction_details.keys():
    if cluster_id in lstm_prediction_details:
        details = lstm_prediction_details[cluster_id]
        y_true = details['y_true']
        y_pred = details['y_pred']
        errors = details['errors']
        absolute_errors = details['absolute_errors']
        
        # Создаем отдельные таблицы для каждой целевой переменной
        for target_idx, target_col in enumerate(target_columns):
            # Создаем DataFrame для текущей целевой переменной
            prediction_table = pd.DataFrame({
                'Sample_Index': range(len(y_true[:, target_idx])),
                'True_Value': y_true[:, target_idx],
                'Predicted_Value': y_pred[:, target_idx],
                'Error': errors[:, target_idx],
                'Absolute_Error': absolute_errors[:, target_idx],
                'Relative_Error_%': np.where(
                    y_true[:, target_idx] != 0,
                    np.abs(errors[:, target_idx]) / np.abs(y_true[:, target_idx]) * 100,
                    np.abs(errors[:, target_idx]) * 100  # Если истинное значение равно 0
                )
            })
            
            # Добавляем статистику
            stats_row = {
                'Sample_Index': 'Statistics',
                'True_Value': np.mean(y_true[:, target_idx]),
                'Predicted_Value': np.mean(y_pred[:, target_idx]),
                'Error': np.mean(errors[:, target_idx]),
                'Absolute_Error': np.mean(absolute_errors[:, target_idx]),
                'Relative_Error_%': np.mean(prediction_table['Relative_Error_%'].iloc[:-1])
            }
            
            # Создаем DataFrame со статистикой и объединяем
            stats_df = pd.DataFrame([stats_row])
            prediction_table = pd.concat([prediction_table, stats_df], ignore_index=True)
            
            # Сохраняем в Excel
            filename = f'prediction_details_cluster_{cluster_id}_{target_col}.xlsx'
            prediction_table.to_excel(filename, index=False)
            print(f"  Таблица для {target_col} (кластер {cluster_id}) сохранена в {filename}")
        
        # Создаем сводную таблицу для всех переменных
        summary_table = pd.DataFrame()
        for target_idx, target_col in enumerate(target_columns):
            col_summary = {
                'Target_Variable': target_col,
                'Mean_True': np.mean(y_true[:, target_idx]),
                'Mean_Predicted': np.mean(y_pred[:, target_idx]),
                'Mean_Error': np.mean(errors[:, target_idx]),
                'Std_Error': np.std(errors[:, target_idx]),
                'MAE': np.mean(absolute_errors[:, target_idx]),
                'MAPE_%': lstm_mape_results[cluster_id][target_idx] if cluster_id in lstm_mape_results else 0,
                'Max_Absolute_Error': np.max(absolute_errors[:, target_idx]),
                'Min_Absolute_Error': np.min(absolute_errors[:, target_idx])
            }
            summary_table = pd.concat([summary_table, pd.DataFrame([col_summary])], ignore_index=True)
        
        summary_filename = f'prediction_summary_cluster_{cluster_id}.xlsx'
        summary_table.to_excel(summary_filename, index=False)
        print(f"  Сводная таблица для кластера {cluster_id} сохранена в {summary_filename}")

# 8. Создание графика MAPE для всех кластеров и целевых переменных
print("\n--- Создание графика MAPE ---")

# Проверяем, есть ли данные для построения графика MAPE
if lstm_mape_results:
    plt.figure(figsize=(15, 8))
    
    # Подготовка данных для группированного столбчатого графика
    cluster_ids = list(lstm_mape_results.keys())
    x = np.arange(len(target_columns))
    width = 0.8 / len(cluster_ids)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
    
    for idx, cluster_id in enumerate(cluster_ids):
        mape_values = lstm_mape_results[cluster_id]
        plt.bar(x + idx * width - (len(cluster_ids) - 1) * width / 2, 
                mape_values, width, label=f'Cluster {cluster_id}', color=colors[idx], alpha=0.8)
        
        # Добавление значений на столбцы
        for j, value in enumerate(mape_values):
            plt.text(x[j] + idx * width - (len(cluster_ids) - 1) * width / 2, 
                    value + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Target Variables', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.title('Mean Absolute Percentage Error (MAPE) by Cluster and Target Variable', fontsize=14, pad=20)
    plt.xticks(x, target_columns, rotation=45, ha='right')
    plt.legend(title='Clusters', loc='upper right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, max([max(vals) for vals in lstm_mape_results.values()]) * 1.2 if any(lstm_mape_results.values()) else 50)
    
    # Добавляем горизонтальные линии для уровней качества прогноза
    plt.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Excellent (<10%)')
    plt.axhline(y=20, color='yellow', linestyle='--', alpha=0.5, label='Good (<20%)')
    plt.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Fair (<30%)')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Poor (<50%)')
    
    plt.tight_layout()
    plt.savefig('mape_by_cluster_and_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("График MAPE сохранен как 'mape_by_cluster_and_target.png'")
    
    # Дополнительный график: средний MAPE по кластерам
    plt.figure(figsize=(12, 6))
    
    avg_mape_by_cluster = {}
    for cluster_id, mape_values in lstm_mape_results.items():
        avg_mape_by_cluster[cluster_id] = np.mean(mape_values)
    
    # Сортируем кластеры по среднему MAPE
    sorted_clusters = sorted(avg_mape_by_cluster.items(), key=lambda x: x[1])
    clusters_sorted = [c[0] for c in sorted_clusters]
    avg_mape_sorted = [c[1] for c in sorted_clusters]
    
    bars = plt.bar(range(len(clusters_sorted)), avg_mape_sorted, 
                   color=plt.cm.viridis(np.linspace(0, 0.8, len(clusters_sorted))))
    
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Average MAPE (%)', fontsize=12)
    plt.title('Average MAPE by Cluster', fontsize=14, pad=20)
    plt.xticks(range(len(clusters_sorted)), [f'Cluster {c}' for c in clusters_sorted])
    
    # Добавляем значения на столбцы
    for idx, value in enumerate(avg_mape_sorted):
        plt.text(idx, value + 1, f'{value:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Цветовая градация по качеству прогноза
    ax = plt.gca()
    ax.axhspan(0, 10, alpha=0.2, color='green', label='Excellent')
    ax.axhspan(10, 20, alpha=0.2, color='yellow', label='Good')
    ax.axhspan(20, 30, alpha=0.2, color='orange', label='Fair')
    ax.axhspan(30, 50, alpha=0.2, color='red', label='Poor')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('average_mape_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("График среднего MAPE сохранен как 'average_mape_by_cluster.png'")

# 9. Классификация нового временного ряда и прогнозирование
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

# 10. Сводный отчет по кластерам
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

# 11. Создание отчета в Excel
report_data = {
    'Metric': [
        'Total Segments',
        'Number of Clusters',
        'Segment Length',
        'LSTM Sequence Length',
        'Target Variables',
        'LSTM Models Trained',
        'Data Smoothing Method',
        'Smoothing Window Size'
    ],
    'Value': [
        len(segments),
        optimal_k,
        segment_length,
        10,
        len(target_columns),
        len(lstm_models),
        'moving_average',
        3
    ]
}

cluster_report = []
for cluster_id in range(optimal_k):
    count = len(features_df[features_df['cluster'] == cluster_id])
    percentage = count/len(segments)*100 if len(segments) > 0 else 0
    has_model = 'Yes' if cluster_id in lstm_models else 'No'
    
    # Добавляем информацию о MAPE, если есть
    if cluster_id in lstm_mape_results:
        avg_mape = np.mean(lstm_mape_results[cluster_id])
        mape_info = f"{avg_mape:.2f}%"
    else:
        mape_info = "N/A"
    
    cluster_report.append({
        'Cluster ID': cluster_id,
        'Segments Count': count,
        'Percentage (%)': round(percentage, 1),
        'LSTM Model': has_model,
        'Avg MAPE': mape_info
    })

report_df1 = pd.DataFrame(report_data)
report_df2 = pd.DataFrame(cluster_report)

with pd.ExcelWriter('analysis_report.xlsx') as writer:
    report_df1.to_excel(writer, sheet_name='Summary', index=False)
    report_df2.to_excel(writer, sheet_name='Cluster Distribution', index=False)
    features_df.to_excel(writer, sheet_name='Segment Features', index=False)

# 12. Сохранение результатов MAPE в отдельный Excel файл
if lstm_mape_results:
    mape_data = []
    for cluster_id, mape_values in lstm_mape_results.items():
        for i, target_col in enumerate(target_columns):
            if i < len(mape_values):
                mape_data.append({
                    'Cluster': cluster_id,
                    'Target Variable': target_col,
                    'MAPE (%)': round(mape_values[i], 2)
                })
    
    mape_df = pd.DataFrame(mape_data)
    
    # Добавляем сводную статистику
    summary_stats = []
    for cluster_id in lstm_mape_results.keys():
        if cluster_id in lstm_mape_results:
            mape_vals = lstm_mape_results[cluster_id]
            summary_stats.append({
                'Cluster': cluster_id,
                'Min MAPE (%)': round(np.min(mape_vals), 2),
                'Max MAPE (%)': round(np.max(mape_vals), 2),
                'Avg MAPE (%)': round(np.mean(mape_vals), 2),
                'Std MAPE (%)': round(np.std(mape_vals), 2)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    
    with pd.ExcelWriter('mape_results.xlsx') as writer:
        mape_df.to_excel(writer, sheet_name='Detailed MAPE', index=False)
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
    
    print("\nРезультаты MAPE сохранены в 'mape_results.xlsx'")

# 13. Сохранение сглаженных данных
df_smoothed.to_excel('smoothed_dataset.xlsx', index=False)
print("Сглаженные данные сохранены в 'smoothed_dataset.xlsx'")

print("\n" + "="*60)
print("СОЗДАННЫЕ ФАЙЛЫ")
print("="*60)
files_categories = {
    "Данные": [
        "segmented_time_series.xlsx",
        "temporal_features.xlsx", 
        "temporal_features_with_clusters.xlsx",
        "analysis_report.xlsx",
        "mape_results.xlsx",
        "smoothed_dataset.xlsx"
    ],
    "Графики": [
        "smoothing_comparison.png",
        "mape_by_cluster_and_target.png",
        "average_mape_by_cluster.png"
    ],
    "Таблицы прогнозов": [],
    "Модели LSTM": [f"lstm_model_cluster_{cluster_id}.keras" for cluster_id in lstm_models.keys()],
    "Графики обучения и ошибок": []
}

# Добавляем таблицы прогнозов
for cluster_id in lstm_prediction_details.keys():
    files_categories["Таблицы прогнозов"].append(f'prediction_summary_cluster_{cluster_id}.xlsx')
    for target_col in target_columns:
        files_categories["Таблицы прогнозов"].append(f'prediction_details_cluster_{cluster_id}_{target_col}.xlsx')

# Добавляем графики обучения и ошибок
for cluster_id in lstm_models.keys():
    files_categories["Графики обучения и ошибок"].append(f'loss_cluster_{cluster_id}.png')
    files_categories["Графики обучения и ошибок"].append(f'all_predictions_cluster_{cluster_id}.png')
    files_categories["Графики обучения и ошибок"].append(f'absolute_errors_cluster_{cluster_id}.png')
    files_categories["Графики обучения и ошибок"].append(f'error_distribution_cluster_{cluster_id}.png')
    for target_col in target_columns:
        files_categories["Графики обучения и ошибок"].append(f'predictions_{target_col}_cluster_{cluster_id}.png')
        files_categories["Графики обучения и ошибок"].append(f'errors_{target_col}_cluster_{cluster_id}.png')

for category, files in files_categories.items():
    print(f"\n{category}:")
    # Для таблиц показываем только первые 5 файлов
    max_files = 5 if category == "Таблицы прогнозов" else 10
    for file in files[:max_files]:
        print(f"  - {file}")
    if len(files) > max_files:
        print(f"  ... и еще {len(files) - max_files} файлов")

print("\n" + "="*60)
print("ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ И ТАБЛИЦЫ:")
print("="*60)
print("1. Графики ошибок для каждой целевой переменной:")
print("   - errors_[target]_cluster_[id].png - график ошибок прогноза")
print("   - absolute_errors_cluster_[id].png - график абсолютных ошибок")
print("   - error_distribution_cluster_[id].png - гистограмма распределения ошибок")
print("\n2. Детальные таблицы прогнозов:")
print("   - prediction_details_cluster_[id]_[target].xlsx - таблица с прогнозами и ошибками")
print("   - prediction_summary_cluster_[id].xlsx - сводная статистика по кластеру")

print("\n" + "="*60)
print("СКРИПТ УСПЕШНО ВЫПОЛНЕН!")
print("="*60)