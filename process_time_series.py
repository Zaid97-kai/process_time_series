import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Импорт для LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Импорт для визуализации
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def calculate_time_series_features(segment, segment_id, segment_start_idx, target_columns):
    """
    Вычисляет временные характеристики для сегмента временного ряда.
    Теперь характеристики вычисляются отдельно для каждого выходного параметра.
    """
    if len(segment) < 5:
        return None
    
    features = {
        'Сегмент_ID': segment_id,
        'Начальный_индекс': segment_start_idx,
        'Длина_сегмента': len(segment),
    }
    
    # Для каждого выходного параметра вычисляем характеристики
    for target_col in target_columns:
        time_series = segment[target_col].values
        
        # Проверяем на NaN
        if np.isnan(time_series).any():
            # Заменяем NaN на среднее значение
            time_series = np.nan_to_num(time_series, nan=np.nanmean(time_series))
        
        # Базовые статистики
        features[f'{target_col}_Минимум'] = np.min(time_series)
        features[f'{target_col}_Максимум'] = np.max(time_series)
        features[f'{target_col}_Среднее'] = np.mean(time_series)
        features[f'{target_col}_Медиана'] = np.median(time_series)
        features[f'{target_col}_Дисперсия'] = np.var(time_series)
        features[f'{target_col}_Стандартное_отклонение'] = np.std(time_series)
        features[f'{target_col}_Размах'] = np.max(time_series) - np.min(time_series)
        
        # Коэффициенты асимметрии и эксцесса
        if len(time_series) > 2 and np.std(time_series) > 0:
            features[f'{target_col}_Коэффициент_асимметрии'] = stats.skew(time_series)
        else:
            features[f'{target_col}_Коэффициент_асимметрии'] = 0
            
        if len(time_series) > 3 and np.std(time_series) > 0:
            features[f'{target_col}_Коэффициент_эксцесса'] = stats.kurtosis(time_series)
        else:
            features[f'{target_col}_Коэффициент_эксцесса'] = 0
        
        # Автокорреляция (лаг 1)
        if len(time_series) > 1 and np.std(time_series) > 0:
            autocorr = pd.Series(time_series).autocorr(lag=1)
            features[f'{target_col}_Автокорреляция_лаг1'] = autocorr if not np.isnan(autocorr) else 0
        else:
            features[f'{target_col}_Автокорреляция_лаг1'] = 0
        
        # Площадь под графиком
        features[f'{target_col}_Площадь_под_графиком'] = np.trapz(time_series)
        
        # Наклон тренда
        x = np.arange(len(time_series))
        if len(time_series) > 1 and np.var(time_series) > 0:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_series)
                features[f'{target_col}_Наклон_тренда'] = slope
                features[f'{target_col}_R_квадрат_тренда'] = r_value**2
            except:
                features[f'{target_col}_Наклон_тренда'] = 0
                features[f'{target_col}_R_квадрат_тренда'] = 0
        else:
            features[f'{target_col}_Наклон_тренда'] = 0
            features[f'{target_col}_R_квадрат_тренда'] = 0
        
        # Квартили и IQR
        if len(time_series) >= 4:
            q1 = np.percentile(time_series, 25)
            q3 = np.percentile(time_series, 75)
            features[f'{target_col}_Q1'] = q1
            features[f'{target_col}_Q3'] = q3
            features[f'{target_col}_IQR'] = q3 - q1
        else:
            features[f'{target_col}_Q1'] = 0
            features[f'{target_col}_Q3'] = 0
            features[f'{target_col}_IQR'] = 0
        
        # Количество пиков
        try:
            peaks, _ = find_peaks(time_series)
            features[f'{target_col}_Количество_пиков'] = len(peaks)
        except:
            features[f'{target_col}_Количество_пиков'] = 0
        
        # Коэффициент вариации
        if np.mean(time_series) != 0 and np.std(time_series) > 0:
            features[f'{target_col}_Коэффициент_вариации'] = np.std(time_series) / np.mean(time_series)
        else:
            features[f'{target_col}_Коэффициент_вариации'] = 0
        
        # Среднеквадратичное значение
        features[f'{target_col}_Среднеквадратичное'] = np.sqrt(np.mean(time_series**2))
    
    return features

def perform_kmeans_clustering(features_df, n_clusters=3):
    """
    Выполняет кластеризацию KMeans с обработкой NaN значений.
    """
    print("\n" + "=" * 60)
    print("КЛАСТЕРИЗАЦИЯ K-MEANS")
    print("=" * 60)
    
    # Выбираем только числовые характеристики для кластеризации
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    # Исключаем идентификаторы и индексы
    exclude_cols = ['Начальный_индекс', 'Длина_сегмента']
    cluster_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = features_df[cluster_cols].values
    
    print(f"Размерность данных для кластеризации: {X.shape}")
    print(f"Используемые характеристики: {len(cluster_cols)}")
    
    # Проверяем на NaN
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"Обнаружено NaN значений: {nan_count}")
        print("Заполняем NaN средними значениями...")
        
        # Заполняем NaN средними значениями по колонкам
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    # Нормализация данных (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Автоматический выбор количества кластеров с помощью метода локтя
    if n_clusters == 'auto':
        print("\nОпределение оптимального количества кластеров (метод локтя)...")
        inertias = []
        max_clusters = min(10, len(X_scaled))
        
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Находим "локоть" - точку, где уменьшение инерции замедляется
        diffs = np.diff(inertias)
        diff_diffs = np.diff(diffs)
        if len(diff_diffs) > 0:
            n_clusters = np.argmax(diff_diffs) + 2
        else:
            n_clusters = 3
        
        print(f"Оптимальное количество кластеров: {n_clusters}")
    
    # Применение KMeans
    print(f"\nВыполнение кластеризации KMeans с {n_clusters} кластерами...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Добавляем метки кластеров в DataFrame
    features_df['Кластер'] = cluster_labels
    
    # Статистика по кластерам
    print(f"\nРезультаты кластеризации:")
    print(f"Количество кластеров: {n_clusters}")
    print(f"Количество точек в каждом кластере:")
    
    for cluster_id in sorted(set(cluster_labels)):
        count = list(cluster_labels).count(cluster_id)
        print(f"  Кластер {cluster_id}: {count} сегментов ({count/len(cluster_labels)*100:.1f}%)")
    
    # Вычисляем инерцию (сумма квадратов расстояний до центроидов)
    inertia = kmeans.inertia_
    print(f"\nИнерция (within-cluster sum of squares): {inertia:.2f}")
    
    return features_df, cluster_labels, scaler, cluster_cols, kmeans

def prepare_cross_segment_data(train_segment, test_segment, input_columns, output_columns, sequence_length=10):
    """
    Подготавливает данные для обучения на одном сегменте и тестирования на другом.
    """
    # Проверяем на NaN
    train_data = train_segment.fillna(train_segment.mean())
    test_data = test_segment.fillna(test_segment.mean())
    
    # Разделяем входные и выходные данные для обучения
    X_train_data = train_data[input_columns].values
    y_train_data = train_data[output_columns].values
    
    # Разделяем входные и выходные данные для тестирования
    X_test_data = test_data[input_columns].values
    y_test_data = test_data[output_columns].values
    
    # Нормализация данных (отдельно для обучения и тестирования для реалистичности)
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    
    X_train_scaled = X_scaler.fit_transform(X_train_data)
    y_train_scaled = y_scaler.fit_transform(y_train_data)
    
    # Для тестовых данных используем те же скейлеры
    X_test_scaled = X_scaler.transform(X_test_data)
    y_test_scaled = y_scaler.transform(y_test_data)
    
    # Создание последовательностей для обучения
    X_train_seq, y_train_seq = [], []
    for i in range(len(X_train_scaled) - sequence_length):
        X_train_seq.append(X_train_scaled[i:i+sequence_length])
        y_train_seq.append(y_train_scaled[i+sequence_length])
    
    # Создание последовательностей для тестирования
    X_test_seq, y_test_seq = [], []
    for i in range(len(X_test_scaled) - sequence_length):
        X_test_seq.append(X_test_scaled[i:i+sequence_length])
        y_test_seq.append(y_test_scaled[i+sequence_length])
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        return None, None, None, None, None, None
    
    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)
    X_test_seq = np.array(X_test_seq)
    y_test_seq = np.array(y_test_seq)
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, X_scaler, y_scaler

def build_improved_lstm_model(input_shape, output_dim, units=64, dropout_rate=0.3, l2_reg=0.001):
    """
    Строит улучшенную многомерную модель LSTM для повышения точности.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True, kernel_regularizer=l2(l2_reg), 
             recurrent_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        LSTM(units//2, return_sequences=True, kernel_regularizer=l2(l2_reg),
             recurrent_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        LSTM(units//4, return_sequences=False, kernel_regularizer=l2(l2_reg),
             recurrent_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate/2),
        
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        
        Dense(output_dim)
    ])
    
    # Используем адаптивный learning rate
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def train_and_test_cross_segment(cluster_data, cluster_id, input_columns, output_columns, 
                                sequence_length=10, epochs=100):
    """
    Обучает улучшенную модель на одном сегменте и тестирует на другом в том же кластере.
    """
    print(f"\n{'='*60}")
    print(f"КЛАСТЕР {cluster_id}: ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ")
    print(f"{'='*60}")
    
    segment_ids = list(cluster_data.keys())
    
    if len(segment_ids) < 2:
        print(f"❌ В кластере {cluster_id} недостаточно сегментов для кросс-сегментного обучения (требуется минимум 2)")
        return None, None, None, None
    
    # Выбираем сегменты для обучения и тестирования
    train_seg_id = segment_ids[0]
    test_seg_id = segment_ids[1]
    
    print(f"🎯 СТРАТЕГИЯ:")
    print(f"   Обучение на сегменте: {train_seg_id}")
    print(f"   Тестирование на сегменте: {test_seg_id}")
    
    train_segment = cluster_data[train_seg_id]
    test_segment = cluster_data[test_seg_id]
    
    # Подготовка данных
    prepared_data = prepare_cross_segment_data(
        train_segment, test_segment, input_columns, output_columns, sequence_length
    )
    
    if prepared_data[0] is None:
        print(f"❌ Не удалось подготовить данные для обучения и тестирования")
        return None, None, None, None
    
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = prepared_data
    
    if len(X_train) < 1:
        print(f"❌ Недостаточно данных для обучения")
        return None, None, None, None
    
    if len(X_test) < 1:
        print(f"❌ Недостаточно данных для тестирования")
        return None, None, None, None
    
    print(f"📊 ДАННЫЕ:")
    print(f"   Размер обучающей выборки: {len(X_train)} последовательностей")
    print(f"   Размер тестовой выборки: {len(X_test)} последовательностей")
    
    # Построение улучшенной модели
    model = build_improved_lstm_model(
        input_shape=(sequence_length, len(input_columns)),
        output_dim=len(output_columns)
    )
    
    # Callbacks для улучшения обучения
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=0.0001,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=0
    )
    
    print("🏋️ НАЧИНАЕМ ОБУЧЕНИЕ МОДЕЛИ...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Проверяем, была ли остановка обучения
    if len(history.history['loss']) < epochs:
        print(f"   Обучение остановлено на эпохе {len(history.history['loss'])} (ранняя остановка)")
    
    # Прогнозирование на тестовых данных
    print("🔮 ВЫПОЛНЯЕМ ПРОГНОЗИРОВАНИЕ...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_original = y_scaler.inverse_transform(y_test)
    
    # Расчет метрик для каждого выходного параметра
    metrics = {
        'Кластер': cluster_id,
        'Сегмент_обучения': train_seg_id,
        'Сегмент_тестирования': test_seg_id,
        'Длина_обучающей_выборки': len(X_train),
        'Длина_тестовой_выборки': len(X_test),
        'Эпохи_обучения': len(history.history['loss'])
    }
    
    param_metrics = {}
    for i, output_col in enumerate(output_columns):
        mse = mean_squared_error(y_test_original[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test_original[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original[:, i], y_pred[:, i])
        
        param_metrics[f'{output_col}_MSE'] = mse
        param_metrics[f'{output_col}_RMSE'] = rmse
        param_metrics[f'{output_col}_MAE'] = mae
        param_metrics[f'{output_col}_R2'] = r2
    
    # Средние метрики по всем параметрам
    metrics['Средний_MSE'] = np.mean([param_metrics[f'{col}_MSE'] for col in output_columns])
    metrics['Средний_RMSE'] = np.mean([param_metrics[f'{col}_RMSE'] for col in output_columns])
    metrics['Средний_MAE'] = np.mean([param_metrics[f'{col}_MAE'] for col in output_columns])
    metrics['Средний_R2'] = np.mean([param_metrics[f'{col}_R2'] for col in output_columns])
    
    # Объединяем все метрики
    all_metrics = {**metrics, **param_metrics}
    
    print(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО:")
    print(f"   Средний R² = {metrics['Средний_R2']:.4f}")
    print(f"   Средний MSE = {metrics['Средний_MSE']:.8f}")
    print(f"   Средний MAE = {metrics['Средний_MAE']:.8f}")
    
    return {
        'y_test': y_test_original,
        'y_pred': y_pred,
        'history': history.history,
        'model': model,
        'train_segment': train_seg_id,
        'test_segment': test_seg_id,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }, all_metrics, train_seg_id, test_seg_id

def create_detailed_prediction_plots(cluster_id, predictions, metrics, output_columns, train_seg_id, test_seg_id):
    """
    Создает детальные графики сравнения прогнозов и истинных значений.
    """
    import os
    cluster_dir = f"cluster_{cluster_id}_results"
    os.makedirs(cluster_dir, exist_ok=True)
    
    y_test = predictions['y_test']
    y_pred = predictions['y_pred']
    
    # 1. Графики для каждого параметра отдельно
    for i, output_col in enumerate(output_columns):
        plt.figure(figsize=(14, 8))
        
        # Основной график сравнения
        plt.subplot(2, 2, 1)
        time_steps = range(len(y_test[:, i]))
        
        plt.plot(time_steps, y_test[:, i], 'b-', linewidth=2, alpha=0.7, label='Истинные значения')
        plt.plot(time_steps, y_pred[:, i], 'r--', linewidth=2, alpha=0.7, label='Прогнозы')
        plt.fill_between(time_steps, y_test[:, i], y_pred[:, i], alpha=0.2, color='gray', label='Ошибка')
        
        plt.title(f'{output_col}\nКластер {cluster_id}', fontsize=14, fontweight='bold')
        plt.xlabel('Временной шаг', fontsize=12)
        plt.ylabel('Значение', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # График ошибок
        plt.subplot(2, 2, 2)
        errors = y_test[:, i] - y_pred[:, i]
        
        plt.plot(time_steps, errors, 'g-', linewidth=1.5, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.fill_between(time_steps, errors, 0, alpha=0.2, color='green')
        
        plt.title(f'Ошибки прогнозирования\nMSE={metrics[f"{output_col}_MSE"]:.6f}, MAE={metrics[f"{output_col}_MAE"]:.6f}', 
                 fontsize=12)
        plt.xlabel('Временной шаг', fontsize=10)
        plt.ylabel('Ошибка', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Гистограмма ошибок
        plt.subplot(2, 2, 3)
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.title('Распределение ошибок', fontsize=12)
        plt.xlabel('Ошибка', fontsize=10)
        plt.ylabel('Частота', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Диаграмма рассеяния
        plt.subplot(2, 2, 4)
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, s=30)
        
        # Линия идеального прогноза
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
        
        plt.title(f'Диаграмма рассеяния\nR²={metrics[f"{output_col}_R2"]:.4f}', fontsize=12)
        plt.xlabel('Истинные значения', fontsize=10)
        plt.ylabel('Прогнозы', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Детальный анализ прогнозов: {output_col}\n'
                    f'Обучение на сегменте: {train_seg_id}, Тестирование на сегменте: {test_seg_id}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{cluster_dir}/detailed_prediction_{output_col}_cluster_{cluster_id}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Сводный график всех параметров
    plt.figure(figsize=(16, 10))
    
    n_params = len(output_columns)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    for i, output_col in enumerate(output_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        
        time_steps = range(len(y_test[:, i]))
        
        plt.plot(time_steps, y_test[:, i], 'b-', linewidth=1.5, alpha=0.7, label='Истинные')
        plt.plot(time_steps, y_pred[:, i], 'r--', linewidth=1.5, alpha=0.7, label='Прогнозы')
        
        plt.title(f'{output_col}\nR²={metrics[f"{output_col}_R2"]:.4f}', fontsize=11)
        plt.xlabel('Временной шаг', fontsize=9)
        plt.ylabel('Значение', fontsize=9)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'СВОДКА ПРОГНОЗОВ ПО ВСЕМ ПАРАМЕТРАМ - КЛАСТЕР {cluster_id}\n'
                f'Обучение на сегменте: {train_seg_id} | Тестирование на сегменте: {test_seg_id}', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{cluster_dir}/summary_predictions_cluster_{cluster_id}.png", 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. График обучения
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(predictions['history']['loss'], 'b-', label='Ошибка обучения', linewidth=2)
    plt.plot(predictions['history']['val_loss'], 'r-', label='Ошибка валидации', linewidth=2)
    plt.title('Кривая обучения', fontsize=14, fontweight='bold')
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Ошибка (MSE)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions['history']['mae'], 'g-', label='MAE обучения', linewidth=2)
    plt.plot([m for m in predictions['history']['val_mae'] if not np.isnan(m)], 'orange', 
             label='MAE валидации', linewidth=2)
    plt.title('Средняя абсолютная ошибка', fontsize=14, fontweight='bold')
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'ПРОЦЕСС ОБУЧЕНИЯ - КЛАСТЕР {cluster_id}\n'
                f'Средний R² = {metrics["Средний_R2"]:.4f}', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{cluster_dir}/training_history_cluster_{cluster_id}.png", 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Созданы детальные графики для кластера {cluster_id}")

def save_cross_segment_results(cluster_id, predictions, metrics, features_df, output_columns, 
                              train_seg_id, test_seg_id):
    """
    Сохраняет результаты кросс-сегментного обучения и тестирования.
    """
    import os
    cluster_dir = f"cluster_{cluster_id}_results"
    os.makedirs(cluster_dir, exist_ok=True)
    
    # 1. Сохраняем метрики в Excel
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_excel(f"{cluster_dir}/cross_segment_metrics_cluster_{cluster_id}.xlsx", index=False)
    
    # 2. Сохраняем прогнозы в Excel
    if predictions:
        with pd.ExcelWriter(f"{cluster_dir}/cross_segment_predictions_cluster_{cluster_id}.xlsx") as writer:
            # Создаем DataFrame с прогнозами для всех выходных параметров
            pred_dfs = []
            for i, col in enumerate(output_columns):
                temp_df = pd.DataFrame({
                    f'{col}_Фактические': predictions['y_test'][:, i],
                    f'{col}_Прогнозные': predictions['y_pred'][:, i],
                    f'{col}_Ошибка': predictions['y_test'][:, i] - predictions['y_pred'][:, i],
                    f'{col}_Абсолютная_ошибка': np.abs(predictions['y_test'][:, i] - predictions['y_pred'][:, i]),
                    f'{col}_Относительная_ошибка_%': 100 * np.abs(predictions['y_test'][:, i] - predictions['y_pred'][:, i]) / 
                                                  np.abs(predictions['y_test'][:, i] + 1e-10)
                })
                pred_dfs.append(temp_df)
            
            # Объединяем все прогнозы в один DataFrame
            pred_df = pd.concat(pred_dfs, axis=1)
            pred_df.to_excel(writer, sheet_name="predictions", index=False)
            
            # Добавляем информацию о сегментах
            seg_info = pd.DataFrame({
                'Параметр': ['Кластер', 'Сегмент обучения', 'Сегмент тестирования', 
                           'Размер обучающей выборки', 'Размер тестовой выборки',
                           'Эпохи обучения', 'Средний R²', 'Средний MSE'],
                'Значение': [cluster_id, train_seg_id, test_seg_id,
                           metrics['Длина_обучающей_выборки'], metrics['Длина_тестовой_выборки'],
                           metrics['Эпохи_обучения'], metrics['Средний_R2'], metrics['Средний_MSE']]
            })
            seg_info.to_excel(writer, sheet_name="segment_info", index=False)
    
    # 3. Сохраняем информацию о сегментах в кластере
    cluster_segments = features_df[features_df['Кластер'] == cluster_id]
    cluster_segments.to_excel(f"{cluster_dir}/segments_info_cluster_{cluster_id}.xlsx", index=False)
    
    # 4. Создаем детальные графики
    create_detailed_prediction_plots(cluster_id, predictions, metrics, output_columns, 
                                   train_seg_id, test_seg_id)
    
    print(f"✓ Результаты кросс-сегментного анализа для кластера {cluster_id} сохранены")
    
    return metrics_df

def create_comprehensive_analysis(all_cluster_results, output_columns, features_df):
    """
    Создает комплексный анализ результатов всех кластеров.
    """
    print("\n" + "=" * 60)
    print("КОМПЛЕКСНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    # 1. Сводная таблица
    summary_data = []
    
    for cluster_id, results in all_cluster_results.items():
        if not results:
            continue
            
        summary = {
            'Кластер': cluster_id,
            'Сегмент_обучения': results['metrics']['Сегмент_обучения'],
            'Сегмент_тестирования': results['metrics']['Сегмент_тестирования'],
            'Эпохи_обучения': results['metrics']['Эпохи_обучения'],
            'Средний_R2': results['metrics']['Средний_R2'],
            'Средний_MSE': results['metrics']['Средний_MSE'],
            'Средний_RMSE': results['metrics']['Средний_RMSE'],
            'Средний_MAE': results['metrics']['Средний_MAE']
        }
        
        # Добавляем метрики для каждого параметра
        for col in output_columns:
            summary[f'{col}_R2'] = results['metrics'][f'{col}_R2']
            summary[f'{col}_MSE'] = results['metrics'][f'{col}_MSE']
        
        summary_data.append(summary)
    
    if not summary_data:
        print("Нет данных для создания анализа")
        return None
    
    # Создаем DataFrame сводной таблицы
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Средний_R2', ascending=False)
    summary_df.to_excel("comprehensive_analysis_summary.xlsx", index=False)
    
    # 2. Визуализация результатов
    create_comprehensive_visualizations(all_cluster_results, summary_df, features_df, output_columns)
    
    return summary_df

def create_comprehensive_visualizations(all_cluster_results, summary_df, features_df, output_columns):
    """
    Создает комплексные визуализации результатов.
    """
    try:
        # 1. Сравнение кластеров по точности
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # График 1: Средний R2 по кластерам
        ax = axes[0, 0]
        clusters = summary_df['Кластер'].astype(str)
        r2_values = summary_df['Средний_R2'].values
        
        bars = ax.bar(clusters, r2_values, color=plt.cm.viridis(np.linspace(0, 1, len(clusters))))
        ax.set_title('Средний R² по кластерам', fontsize=14, fontweight='bold')
        ax.set_xlabel('Кластер', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, r2 in zip(bars, r2_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
        
        # График 2: Тепловая карта R2 по параметрам и кластерам
        ax = axes[0, 1]
        r2_matrix = []
        
        for cluster_id in summary_df['Кластер']:
            if cluster_id in all_cluster_results:
                cluster_r2 = []
                for col in output_columns:
                    cluster_r2.append(all_cluster_results[cluster_id]['metrics'][f'{col}_R2'])
                r2_matrix.append(cluster_r2)
        
        if r2_matrix:
            r2_matrix = np.array(r2_matrix)
            im = ax.imshow(r2_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            
            ax.set_xticks(np.arange(len(output_columns)))
            ax.set_xticklabels(output_columns, rotation=45, ha='right')
            ax.set_yticks(np.arange(len(summary_df)))
            ax.set_yticklabels(summary_df['Кластер'].astype(str))
            
            # Добавляем значения в ячейки
            for i in range(len(summary_df)):
                for j in range(len(output_columns)):
                    text = ax.text(j, i, f'{r2_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title('R² по параметрам и кластерам', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='R² score')
        
        # График 3: Распределение сегментов по кластерам
        ax = axes[1, 0]
        cluster_counts = features_df['Кластер'].value_counts().sort_index()
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_counts)))
        wedges, texts, autotexts = ax.pie(cluster_counts.values, labels=cluster_counts.index.astype(str),
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        ax.set_title('Распределение сегментов по кластерам', fontsize=14, fontweight='bold')
        
        # График 4: Сравнение MSE по кластерам
        ax = axes[1, 1]
        mse_values = summary_df['Средний_MSE'].values
        
        ax.bar(clusters, mse_values, color=plt.cm.plasma(np.linspace(0, 1, len(clusters))))
        ax.set_title('Средний MSE по кластерам', fontsize=14, fontweight='bold')
        ax.set_xlabel('Кластер', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        plt.suptitle('КОМПЛЕКСНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ И ПРОГНОЗИРОВАНИЯ', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('comprehensive_analysis_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. График сравнения параметров
        plt.figure(figsize=(14, 8))
        
        param_avg_r2 = []
        param_avg_mse = []
        
        for col in output_columns:
            r2_values = []
            mse_values = []
            for cluster_id, results in all_cluster_results.items():
                if results:
                    r2_values.append(results['metrics'][f'{col}_R2'])
                    mse_values.append(results['metrics'][f'{col}_MSE'])
            
            if r2_values:
                param_avg_r2.append(np.mean(r2_values))
                param_avg_mse.append(np.mean(mse_values))
        
        x = np.arange(len(output_columns))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # График R2
        bars1 = ax1.bar(x - width/2, param_avg_r2, width, label='Средний R²', color='skyblue', alpha=0.7)
        ax1.set_xlabel('Параметр', fontsize=12)
        ax1.set_ylabel('Средний R²', fontsize=12, color='skyblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(output_columns, rotation=45, ha='right')
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # График MSE на второй оси Y
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, param_avg_mse, width, label='Средний MSE', color='salmon', alpha=0.7)
        ax2.set_ylabel('Средний MSE', fontsize=12, color='salmon')
        ax2.tick_params(axis='y', labelcolor='salmon')
        ax2.set_yscale('log')
        
        # Объединяем легенды
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('СРАВНЕНИЕ КАЧЕСТВА ПРОГНОЗОВ ПО ПАРАМЕТРАМ', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('parameter_comparison_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Созданы комплексные визуализации результатов")
        
    except Exception as e:
        print(f"Ошибка при создании визуализаций: {e}")

def main():
    """
    Главная функция - полный анализ с улучшенными моделями LSTM.
    """
    print("=" * 80)
    print("МНОГОМЕРНЫЙ АНАЛИЗ С УЛУЧШЕННЫМИ МОДЕЛЯМИ LSTM")
    print("=" * 80)
    print("🎯 СТРАТЕГИЯ: Обучение на одном сегменте кластера, тестирование на другом")
    print("📈 УЛУЧШЕНИЯ: Более глубокая архитектура, регуляризация, оптимизация")
    print("=" * 80)
    
    # Конфигурация
    file_path = 'Dataset.xlsx'
    segment_length = 20
    
    # Определяем входные и выходные параметры
    # Входные параметры
    input_columns = ['kγ', 'kβ', 'α0', 'lψ', 'V0', 'LWx', 'ω*', 'e1', 'e6', 'F1', 'F6', 'F']
    
    # Выходные параметры (прогнозируемые)
    output_columns = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    
    try:
        # ===== 1. ЗАГРУЗКА И ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА =====
        print("\n1. ЗАГРУЗКА И ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА")
        print("-" * 50)
        
        df = pd.read_excel(file_path)
        print(f"✓ Загружено {len(df)} записей")
        print(f"✓ Входные параметры: {', '.join(input_columns[:5])}...")
        print(f"✓ Выходные параметры: {', '.join(output_columns)}")
        
        # Проверяем наличие всех колонок
        missing_input = [col for col in input_columns if col not in df.columns]
        missing_output = [col for col in output_columns if col not in df.columns]
        
        if missing_input:
            print(f"❌ Отсутствуют входные параметры: {missing_input}")
            # Попробуем использовать все колонки кроме выходных
            all_columns = df.columns.tolist()
            input_columns = [col for col in all_columns if col not in output_columns]
            print(f"Используем все остальные колонки как входные: {len(input_columns)} параметров")
        
        if missing_output:
            print(f"❌ Отсутствуют выходные параметры: {missing_output}")
            return
        
        # Проверяем на NaN
        nan_counts = df[input_columns + output_columns].isnull().sum()
        total_nan = nan_counts.sum()
        if total_nan > 0:
            print(f"\nОбнаружено NaN значений: {total_nan}")
            print("Заполняем NaN средними значениями...")
            df[input_columns + output_columns] = df[input_columns + output_columns].fillna(
                df[input_columns + output_columns].mean()
            )
        
        # ===== 2. СЕГМЕНТАЦИЯ И ВЫЧИСЛЕНИЕ ХАРАКТЕРИСТИК =====
        print("\n2. СЕГМЕНТАЦИЯ И ВЫЧИСЛЕНИЕ ХАРАКТЕРИСТИК")
        print("-" * 50)
        
        segments = []
        all_features = []
        segment_data_dict = {}
        
        segment_counter = 0
        for i in range(0, len(df), segment_length):
            segment = df.iloc[i:i + segment_length]
            if len(segment) >= 10:  # Увеличиваем минимальную длину для LSTM
                segment_counter += 1
                seg_id = f"Сегмент_{segment_counter:03d}"
                segments.append(segment)
                
                # Вычисляем характеристики для выходных параметров
                features = calculate_time_series_features(segment, seg_id, i, output_columns)
                if features:
                    all_features.append(features)
                
                # Сохраняем полные данные сегмента для LSTM
                segment_data_dict[seg_id] = segment[input_columns + output_columns]
        
        if not all_features:
            print("❌ Не удалось вычислить характеристики для сегментов")
            return
            
        features_df = pd.DataFrame(all_features)
        print(f"✓ Создано {len(segments)} сегментов")
        print(f"✓ Вычислено {len(all_features)} наборов характеристик")
        
        # Проверяем features_df на NaN
        nan_in_features = features_df.isnull().sum().sum()
        if nan_in_features > 0:
            print(f"\nОбнаружено NaN в характеристиках: {nan_in_features}")
            print("Заполняем средними значениями...")
            features_df = features_df.fillna(features_df.mean())
        
        # ===== 3. КЛАСТЕРИЗАЦИЯ K-MEANS =====
        print("\n3. КЛАСТЕРИЗАЦИЯ K-MEANS")
        print("-" * 50)
        
        # Используем автоматическое определение количества кластеров
        features_df, cluster_labels, scaler, cluster_cols, kmeans = perform_kmeans_clustering(
            features_df, n_clusters='auto'
        )
        
        # Сохраняем результаты кластеризации
        features_df.to_excel("improved_clustering_results.xlsx", index=False)
        print("✓ Результаты кластеризации сохранены в 'improved_clustering_results.xlsx'")
        
        # ===== 4. КРОСС-СЕГМЕНТНОЕ ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ =====
        print("\n4. КРОСС-СЕГМЕНТНОЕ ОБУЧЕНИЕ С УЛУЧШЕННЫМИ МОДЕЛЯМИ")
        print("-" * 50)
        print("🎯 УЛУЧШЕННАЯ АРХИТЕКТУРА:")
        print("   • 3-слойная LSTM сеть")
        print("   • Batch Normalization")
        print("   • L2 регуляризация")
        print("   • ReduceLROnPlateau callback")
        print("   • Более глубокие слои")
        print("-" * 50)
        
        # Группируем сегменты по кластерам
        cluster_segments = {}
        for seg_id, segment_data in segment_data_dict.items():
            # Находим кластер для этого сегмента
            seg_features = features_df[features_df['Сегмент_ID'] == seg_id]
            if not seg_features.empty:
                cluster_id = seg_features['Кластер'].iloc[0]
                if cluster_id not in cluster_segments:
                    cluster_segments[cluster_id] = {}
                cluster_segments[cluster_id][seg_id] = segment_data
        
        print(f"\nРАСПРЕДЕЛЕНИЕ СЕГМЕНТОВ ПО КЛАСТЕРАМ:")
        for cluster_id in sorted(cluster_segments.keys()):
            seg_count = len(cluster_segments[cluster_id])
            print(f"  Кластер {cluster_id}: {seg_count} сегментов")
        
        # Обучаем и тестируем модели для каждого кластера
        all_cluster_results = {}
        
        trained_clusters = 0
        for cluster_id, cluster_data in cluster_segments.items():
            if len(cluster_data) >= 2:  # Только кластеры с минимум 2 сегментами
                
                predictions, metrics, train_seg_id, test_seg_id = train_and_test_cross_segment(
                    cluster_data, cluster_id, input_columns, output_columns,
                    sequence_length=20, epochs=100
                )
                
                if predictions and metrics:
                    all_cluster_results[cluster_id] = {
                        'predictions': predictions,
                        'metrics': metrics,
                        'train_segment': train_seg_id,
                        'test_segment': test_seg_id
                    }
                    trained_clusters += 1
                    
                    # Сохраняем результаты для кластера
                    metrics_df = save_cross_segment_results(
                        cluster_id, predictions, metrics, features_df, output_columns,
                        train_seg_id, test_seg_id
                    )
                else:
                    print(f"\n❌ Не удалось обучить модель для кластера {cluster_id}")
            else:
                print(f"\n⚠️  Кластер {cluster_id}: Пропущен (требуется минимум 2 сегмента, доступно: {len(cluster_data)})")
        
        if trained_clusters == 0:
            print("\n❌ Не удалось обучить ни одной модели LSTM")
            return
        
        # ===== 5. КОМПЛЕКСНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ =====
        print("\n5. КОМПЛЕКСНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("-" * 50)
        
        # Создаем комплексный анализ
        summary_df = create_comprehensive_analysis(all_cluster_results, output_columns, features_df)
        
        if summary_df is not None:
            # Выводим статистику
            print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
            print(f"   Обучено кластеров: {trained_clusters}")
            print(f"   Средний R² по всем кластерам: {summary_df['Средний_R2'].mean():.4f}")
            print(f"   Средний MSE по всем кластерам: {summary_df['Средний_MSE'].mean():.8f}")
            print(f"   Лучший кластер (R²={summary_df['Средний_R2'].max():.4f}): Кластер {summary_df['Средний_R2'].idxmax()}")
            print(f"   Худший кластер (R²={summary_df['Средний_R2'].min():.4f}): Кластер {summary_df['Средний_R2'].idxmin()}")
        
        # ===== 6. СОХРАНЕНИЕ ИТОГОВЫХ РЕЗУЛЬТАТОВ =====
        print("\n6. СОХРАНЕНИЕ ИТОГОВЫХ РЕЗУЛЬТАТОВ")
        print("-" * 50)
        
        # Сохраняем все метрики в один файл
        all_metrics_list = []
        for cluster_id, results in all_cluster_results.items():
            all_metrics_list.append(results['metrics'])
        
        if all_metrics_list:
            all_metrics_df = pd.DataFrame(all_metrics_list)
            all_metrics_df.to_excel("all_improved_metrics.xlsx", index=False)
            print("✓ Все метрики сохранены в 'all_improved_metrics.xlsx'")
        
        # Сохраняем информацию о моделях
        models_info = {
            'Всего_кластеров': len(cluster_segments),
            'Обучено_кластеров': trained_clusters,
            'Стратегия_обучения': 'Кросс-сегментное с улучшенными моделями',
            'Входные_параметры': input_columns,
            'Выходные_параметры': output_columns,
            'Архитектура_LSTM': {
                'layers': 'LSTM(64)-BN-Dropout-LSTM(32)-BN-Dropout-LSTM(16)-BN-Dropout-Dense(64)-BN-Dropout-Dense(32)-BN-Dense(7)',
                'regularization': 'L2 regularization',
                'optimizer': 'Adam with learning rate scheduling',
                'callbacks': 'EarlyStopping, ReduceLROnPlateau'
            },
            'Параметры_обучения': {
                'sequence_length': 10,
                'epochs': 100,
                'batch_size': 32,
                'validation_split': 0.2
            }
        }
        
        import json
        with open('improved_analysis_config.json', 'w', encoding='utf-8') as f:
            json.dump(models_info, f, indent=2, ensure_ascii=False)
        
        print("✓ Конфигурация анализа сохранена в 'improved_analysis_config.json'")
        
        # ===== 7. ФИНАЛЬНЫЙ ОТЧЕТ =====
        print("\n7. ФИНАЛЬНЫЙ ОТЧЕТ")
        print("-" * 50)
        
        print("\n" + "=" * 80)
        print("УЛУЧШЕННЫЙ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
        print("=" * 80)
        
        print(f"\n🎯 РЕЗУЛЬТАТЫ:")
        print(f"   📊 Обучено кластеров: {trained_clusters}")
        print(f"   📈 Средняя точность (R²): {summary_df['Средний_R2'].mean():.4f}" if summary_df is not None else "")
        print(f"   📉 Средняя ошибка (MSE): {summary_df['Средний_MSE'].mean():.8f}" if summary_df is not None else "")
        
        print(f"\n📁 СОЗДАННЫЕ ФАЙЛЫ:")
        print(f"   1. improved_clustering_results.xlsx - результаты кластеризации")
        print(f"   2. all_improved_metrics.xlsx - все метрики моделей")
        print(f"   3. comprehensive_analysis_summary.xlsx - сводная таблица")
        print(f"   4. improved_analysis_config.json - конфигурация")
        print(f"   5. comprehensive_analysis_visualization.png - визуализация")
        print(f"   6. parameter_comparison_analysis.png - сравнение параметров")
        print(f"   7. Папки cluster_X_results/ - детальные результаты по кластерам")
        
        print(f"\n📂 В ПАПКАХ CLUSTER_X_RESULTS/ СОДЕРЖАТСЯ:")
        print(f"   • Детальные графики прогнозов для каждого параметра")
        print(f"   • Графики обучения и ошибок")
        print(f"   • Таблицы с метриками и прогнозами")
        print(f"   • Информация о сегментах обучения и тестирования")
        
        print(f"\n📈 УЛУЧШЕНИЯ ТОЧНОСТИ:")
        print(f"   • Более глубокая архитектура LSTM (3 слоя)")
        print(f"   • Batch Normalization для стабилизации обучения")
        print(f"   • L2 регуляризация для предотвращения переобучения")
        print(f"   • Адаптивный learning rate (ReduceLROnPlateau)")
        print(f"   • Ранняя остановка для оптимального времени обучения")
        
    except FileNotFoundError:
        print(f"\n❌ ОШИБКА: Файл '{file_path}' не найден!")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Проверка доступности TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow версия: {tf.__version__}")
        gpu_devices = tf.config.list_physical_devices('GPU')
        gpu_available = len(gpu_devices) > 0
        print(f"GPU доступен: {gpu_available}")
        if gpu_available:
            print(f"Используется GPU: {gpu_devices[0]}")
    except ImportError:
        print("❌ TensorFlow не установлен. Установите: pip install tensorflow")
        exit(1)
    
    main()