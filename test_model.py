import json
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import lite


def load_scaler_params():
    """Load tham số chuẩn hóa từ file"""
    try:
        with open('scaler_params.json', 'r') as f:
            params = json.load(f)
        return np.array(params['mean']), np.array(params['scale'])
    except FileNotFoundError:
        print("Không tìm thấy file scaler_params.json")
        return None, None

def validate_input_data(temp, humidity, gas, dust, fire_status):
    """Validate input data ranges"""
    if not (0 <= temp <= 200):
        raise ValueError(f"Temperature out of range: {temp}")
    if not (0 <= humidity <= 100):
        raise ValueError(f"Humidity out of range: {humidity}")
    if not (0 <= gas <= 1000):
        raise ValueError(f"Gas value out of range: {gas}")
    if not (0 <= dust <= 300):
        raise ValueError(f"Dust value out of range: {dust}")
    if fire_status not in [0, 1]:
        raise ValueError(f"Fire status must be 0 or 1: {fire_status}")

def normalize_features(features, mean, scale):
    """Normalize features using StandardScaler parameters"""
    if mean is None or scale is None:
        raise ValueError("Scaler parameters not loaded")
    return (features - mean) / scale

def test_scenarios():
    """Test các tình huống thực tế"""
    # Load model
    model_path = "fire_detection_model.tflite"
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file model {model_path}")
        return
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Lấy thông tin input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Print output tensor details for debugging
        print(f"Output tensor type: {output_details[0]['dtype']}")
        print(f"Output tensor scale: {output_details[0]['quantization_parameters']['scales'][0]}")
        print(f"Output tensor zero_point: {output_details[0]['quantization_parameters']['zero_points'][0]}")
        
        # Load tham số chuẩn hóa
        mean, scale = load_scaler_params()
        if mean is None or scale is None:
            return
        
        # Các tình huống test
        scenarios = [
            {
                "name": "Phòng điều hòa bình thường",
                "values": [26.0, 55.0, 50.0, 15.0, 0.0]
            },
            {
                "name": "Phòng bếp khi nấu ăn nhẹ",
                "values": [35.0, 65.0, 150.0, 25.0, 1.0]
            },
            {
                "name": "Phòng bếp khi chiên rán",
                "values": [45.0, 70.0, 300.0, 40.0, 1.0]
            },
            {
                "name": "Phòng bếp khi nướng",
                "values": [60.0, 75.0, 400.0, 50.0, 1.0]
            },
            {
                "name": "Cháy nhỏ do chập điện",
                "values": [85.0, 40.0, 600.0, 100.0, 1.0]
            },
            {
                "name": "Cháy rác",
                "values": [120.0, 30.0, 800.0, 200.0, 1.0]
            },
            {
                "name": "Cháy lớn do xăng dầu",
                "values": [180.0, 20.0, 1000.0, 300.0, 1.0]
            },
            {
                "name": "Phòng có người hút thuốc",
                "values": [28.0, 50.0, 250.0, 80.0, 0.0]
            },
            {
                "name": "Phòng bị ngập nước",
                "values": [30.0, 95.0, 100.0, 10.0, 0.0]
            },
            {
                "name": "Phòng có máy lạnh bị rò gas",
                "values": [24.0, 45.0, 500.0, 20.0, 0.0]
            },
            {
                "name": "Phòng có nhiều người (CO2 cao)",
                "values": [28.0, 60.0, 350.0, 30.0, 0.0]
            },
            {
                "name": "Phòng có máy photocopy (Ozone, bụi cao)",
                "values": [27.0, 45.0, 200.0, 60.0, 0.0]
            },
            {
                "name": "Phòng có sơn mới (VOCs cao)",
                "values": [26.0, 50.0, 450.0, 25.0, 0.0]
            },
            {
                "name": "Cháy âm ỉ, nhiệt độ thấp",
                "values": [55.0, 60.0, 500.0, 100.0, 1.0]
            },
            {
                "name": "Khói nhiều, nhiệt độ trung bình",
                "values": [70.0, 50.0, 750.0, 200.0, 1.0]
            },
            {
                "name": "Cháy vật liệu dễ cháy (quần áo)",
                "values": [90.0, 45.0, 500.0, 250.0, 1.0]
            },
            {
                "name": "Cháy mới khởi phát (nhiệt độ chậm tăng)",
                "values": [60.0, 50.0, 650.0, 180.0, 1.0]
            }
        ]
        
        print("\n=== TESTING REAL-WORLD SCENARIOS ===\n")
        
        for scenario in scenarios:
            print(f"Testing scenario: {scenario['name']}")
            print("-" * 50)
            
            try:
                # Validate input data
                validate_input_data(*scenario['values'])
                
                # In giá trị input
                print("Input values:")
                print(f"  Temperature: {scenario['values'][0]:.1f}°C")
                print(f"  Humidity: {scenario['values'][1]:.1f}%")
                print(f"  Gas: {scenario['values'][2]:.1f} ppm")
                print(f"  Dust: {scenario['values'][3]:.1f} µg/m³")
                print(f"  Fire Sensor: {scenario['values'][4]:.1f}")
                print()
                
                # Chuẩn hóa dữ liệu
                input_data_norm = (np.array(scenario['values']) - mean) / scale
                print("  Normalized input:", [f"{v:.4f}" for v in input_data_norm])
                # Quantize input data if the model expects INT8
                if input_details[0]['dtype'] == np.int8:
                    input_scale = input_details[0]['quantization_parameters']['scales'][0]
                    input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
                    input_data_quant = np.round(input_data_norm / input_scale + input_zero_point).astype(np.int8)
                    print("  Quantized input (int8):", input_data_quant.tolist())
                else:
                    input_data_quant = input_data_norm.astype(np.float32)
                input_data = np.expand_dims(input_data_quant, axis=0)
                
                # Dự đoán
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # In ra output thô (int8)
                if output_details[0]['dtype'] == np.int8:
                    print("  Raw output (int8):", output_data[0].tolist())
                    output_scale = output_details[0]['quantization_parameters']['scales'][0]
                    output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
                    output_dequant = (output_data[0].astype(np.float32) - output_zero_point) * output_scale
                    print("  Dequantized output:", [f"{v:.4f}" for v in output_dequant])
                else:
                    print("  Output (float32):", output_data[0].tolist())
                    output_dequant = output_data[0]
                
                # Lấy kết quả
                prediction = np.argmax(output_dequant)
                confidence = output_dequant[prediction] * 100
                
                # Chuyển đổi nhãn
                label_map = {0: 'NO_FIRE', 1: 'COOKING_FIRE', 2: 'DANGEROUS_FIRE'}
                predicted_class = label_map[prediction]
                
                print("Prediction:")
                print(f"  Class: {predicted_class}")
                print(f"  Confidence: {confidence:.2f}%")
                print()
                
            except ValueError as e:
                print(f"Error in scenario {scenario['name']}: {str(e)}")
                continue
            
    except Exception as e:
        print(f"Lỗi khi chạy model: {str(e)}")

if __name__ == "__main__":
    test_scenarios() 