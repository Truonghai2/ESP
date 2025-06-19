import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Load training data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Tải dữ liệu và thêm đặc trưng fire_sensor_status
    # fire_sensor_status: 1 = no fire, 0 = fire detected
    X = np.array([[sample['temperature'], sample['humidity'], 
                   sample['gas_value'], sample['dust_value'], sample['fire_sensor_status']] 
                  for sample in data['data']])
    y = np.array([sample['label'] for sample in data['data']])
    
    return X, y

# Create and train model
def create_model(input_shape):
    """Tạo mô hình AI cho ESP32 với Flame Sensor"""
    model = Sequential([
        # Lớp đầu vào với regularization L2
        Dense(32, activation='relu', input_shape=input_shape,
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        
        # Lớp ẩn 1 với regularization L2
        Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        
        # Lớp ẩn 2 với regularization L2
        Dense(8, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        
        # Lớp đầu ra
        Dense(3, activation='softmax')  # 3 classes: NO_FIRE, COOKING_FIRE, DANGEROUS_FIRE
    ])
    
    # Compile model với learning rate thấp hơn và thêm gradient clipping
    optimizer = Adam(
        learning_rate=0.0005,  # Giảm learning rate
        clipnorm=1.0  # Thêm gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Vẽ biểu đồ quá trình training"""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Đã lưu biểu đồ lịch sử huấn luyện vào training_history.png")

def plot_confusion_matrix(y_true, y_pred):
    """Vẽ ma trận nhầm lẫn"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NO_FIRE', 'COOKING_FIRE', 'DANGEROUS_FIRE'],
                yticklabels=['NO_FIRE', 'COOKING_FIRE', 'DANGEROUS_FIRE'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Đã lưu ma trận nhầm lẫn vào confusion_matrix.png")

def plot_feature_distributions(X, y):
    """Vẽ phân bố của các đặc trưng theo nhãn"""
    df = np.hstack((X, y.reshape(-1, 1)))
    df = pd.DataFrame(df, columns=['temperature', 'humidity', 'gas_value', 'dust_value', 'fire_sensor_status', 'label'])
    
    plt.figure(figsize=(20, 15))
    
    plt.subplot(3, 2, 1)
    sns.countplot(data=df, x='label')
    plt.title('Phân bố nhãn')
    
    plt.subplot(3, 2, 2)
    sns.histplot(data=df, x='temperature', hue='label', multiple="stack")
    plt.title('Phân bố nhiệt độ')
    
    plt.subplot(3, 2, 3)
    sns.histplot(data=df, x='humidity', hue='label', multiple="stack")
    plt.title('Phân bố độ ẩm')
    
    plt.subplot(3, 2, 4)
    sns.histplot(data=df, x='gas_value', hue='label', multiple="stack")
    plt.title('Phân bố khí gas')
    
    plt.subplot(3, 2, 5)
    sns.histplot(data=df, x='dust_value', hue='label', multiple="stack")
    plt.title('Phân bố bụi')

    plt.subplot(3, 2, 6)
    sns.countplot(data=df, x='fire_sensor_status', hue='label')
    plt.title('Phân bố trạng thái cảm biến lửa')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    print("Đã lưu biểu đồ phân bố đặc trưng vào feature_distributions.png")

def plot_correlation_matrix(X):
    """Vẽ ma trận tương quan giữa các đặc trưng"""
    df = pd.DataFrame(X, columns=['temperature', 'humidity', 'gas_value', 'dust_value', 'fire_sensor_status'])
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    print("Đã lưu ma trận tương quan vào correlation_matrix.png")

def main():
    # Tìm file dữ liệu training mới nhất
    training_files = glob.glob("mock_training_data_*.json")
    if not training_files:
        print("Không tìm thấy file dữ liệu training!")
        return
    
    latest_file = max(training_files, key=os.path.getctime)
    print(f"Đang sử dụng file dữ liệu: {latest_file}")
    
    # Load dữ liệu
    with open(latest_file, 'r') as f:
        data = json.load(f)['data']
    
    # Chuyển đổi dữ liệu
    X = np.array([[d['temperature'], d['humidity'], d['gas_value'], d['dust_value'], d['fire_sensor_status']] for d in data])
    
    # Chuyển đổi nhãn từ chuỗi sang số
    label_map = {'NO_FIRE': 0, 'COOKING_FIRE': 1, 'DANGEROUS_FIRE': 2}
    y = np.array([label_map[d['label']] for d in data])
    
    print(f"Kích thước dữ liệu: {X.shape}")
    print(f"Phân bố nhãn: {np.bincount(y)}")
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nMin/Max of Scaled Training Data:")
    for i, col in enumerate(['temperature', 'humidity', 'gas_value', 'dust_value', 'fire_sensor_status']):
        print(f"  {col}: Min={X_train_scaled[:, i].min():.4f}, Max={X_train_scaled[:, i].max():.4f}")

    print("\nMin/Max of Scaled Test Data:")
    for i, col in enumerate(['temperature', 'humidity', 'gas_value', 'dust_value', 'fire_sensor_status']):
        print(f"  {col}: Min={X_test_scaled[:, i].min():.4f}, Max={X_test_scaled[:, i].max():.4f}")
    
    # Lưu tham số chuẩn hóa
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)
    
    # Tạo và train model
    model = create_model(input_shape=(5,))
    
    # Tính toán class weights để cân bằng dữ liệu
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Early stopping để tránh overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # Tăng patience
        restore_best_weights=True,
        min_delta=0.0005
    )
    
    # Reduce learning rate khi model không cải thiện
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        min_delta=0.0005
    )
    
    # Train model
    callbacks = [early_stopping, reduce_lr]
    
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2
    )
    
    # Đánh giá model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['NO_FIRE', 'COOKING_FIRE', 'DANGEROUS_FIRE']))
    
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_distributions(X, y)
    plot_correlation_matrix(X)

    # Chuyển đổi model sang TensorFlow Lite for Microcontrollers
    # Cần một hàm để tạo tập dữ liệu đại diện cho quantization
    def representative_dataset_gen():
        for i in range(min(100, X_train_scaled.shape[0])):  # Use first 100 samples for calibration
            yield [X_train_scaled[i:i+1].astype(np.float32)]

    # Sử dụng TF Lite Converter với int8 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # Lưu model TensorFlow Lite
    with open('fire_detection_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Đã lưu model TensorFlow Lite (int8) vào fire_detection_model.tflite")

    # Test the TFLite model in Python
    print("\n=== TESTING TFLITE MODEL IN PYTHON ===")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite Input Details:")
    print(input_details)
    print("TFLite Output Details:")
    print(output_details)

    tflite_y_pred = []
    for i, x in enumerate(X_test_scaled):
        # Quantize input data for int8 model
        input_scale = input_details[0]['quantization_parameters']['scales'][0]
        input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
        input_quantized = np.round(x / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_quantized, axis=0))
        
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Dequantize output
        output_scale = output_details[0]['quantization_parameters']['scales'][0]
        output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
        output_dequant = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        tflite_y_pred.append(np.argmax(output_dequant))
    
    tflite_y_pred = np.array(tflite_y_pred)
    print("\nClassification Report (TFLite Model - int8):")
    print(classification_report(y_test, tflite_y_pred, target_names=['NO_FIRE', 'COOKING_FIRE', 'DANGEROUS_FIRE']))
    
    print("\n=== TFLITE MODEL TESTING COMPLETE ===")

if __name__ == "__main__":
    main() 