import json
import random
from datetime import datetime, timedelta

import numpy as np


def generate_mock_data(num_samples=1000):
    """Tạo dữ liệu giả lập cho training"""
    data = []
    
    # Tạo dữ liệu cho các tình huống khác nhau
    scenarios = {
        'NO_FIRE': {
            'temp_range': (20, 35),
            'humidity_range': (40, 70),
            'gas_range': (0, 300),  
            'dust_range': (10, 80),  
            'fire_sensor': 1  # 1 means no fire
        },
        'COOKING_FIRE': {
            'temp_range': (35, 65),
            'humidity_range': (60, 80),
            'gas_range': (150, 500),
            'dust_range': (30, 100),
            'fire_sensor': 0  # 0 means fire detected
        },
        'DANGEROUS_FIRE': {
            'temp_range': (65, 200),
            'humidity_range': (20, 60),
            'gas_range': (500, 1000),
            'dust_range': (100, 300),
            'fire_sensor': 0  # 0 means fire detected
        }
    }
    
    # Tạo dữ liệu cho mỗi tình huống
    for label, params in scenarios.items():
        for _ in range(num_samples // len(scenarios)):
            data.append({
                'temperature': random.uniform(*params['temp_range']),
                'humidity': random.uniform(*params['humidity_range']),
                'gas_value': random.uniform(*params['gas_range']),
                'dust_value': random.uniform(*params['dust_range']),
                'fire_sensor_status': params['fire_sensor'],
                'label': label
            })
    
    # Thêm các trường hợp đặc biệt với trọng số cao
    special_cases = [
        # Phòng điều hòa bình thường
        {'temperature': 26, 'humidity': 55, 'gas_value': 50, 'dust_value': 15, 'fire_sensor_status': 1, 'label': 'NO_FIRE'},
        # Phòng bếp khi nấu ăn nhẹ
        {'temperature': 35, 'humidity': 65, 'gas_value': 150, 'dust_value': 25, 'fire_sensor_status': 0, 'label': 'COOKING_FIRE'},
        # Phòng bếp khi chiên rán
        {'temperature': 45, 'humidity': 70, 'gas_value': 300, 'dust_value': 40, 'fire_sensor_status': 0, 'label': 'COOKING_FIRE'},
        # Phòng bếp khi nướng
        {'temperature': 60, 'humidity': 75, 'gas_value': 400, 'dust_value': 50, 'fire_sensor_status': 0, 'label': 'COOKING_FIRE'},
        # Phòng có người hút thuốc
        {'temperature': 28, 'humidity': 50, 'gas_value': 250, 'dust_value': 80, 'fire_sensor_status': 1, 'label': 'NO_FIRE'},
        # Phòng có máy lạnh bị rò gas
        {'temperature': 24, 'humidity': 45, 'gas_value': 500, 'dust_value': 20, 'fire_sensor_status': 1, 'label': 'NO_FIRE'},
        # Phòng có nhiều người
        {'temperature': 28, 'humidity': 60, 'gas_value': 350, 'dust_value': 30, 'fire_sensor_status': 1, 'label': 'NO_FIRE'},
        # Phòng có sơn mới
        {'temperature': 26, 'humidity': 50, 'gas_value': 450, 'dust_value': 25, 'fire_sensor_status': 1, 'label': 'NO_FIRE'},
        # Cháy nhỏ do chập điện
        {'temperature': 85, 'humidity': 40, 'gas_value': 600, 'dust_value': 100, 'fire_sensor_status': 0, 'label': 'DANGEROUS_FIRE'},
        # Cháy rác
        {'temperature': 120, 'humidity': 30, 'gas_value': 800, 'dust_value': 200, 'fire_sensor_status': 0, 'label': 'DANGEROUS_FIRE'},
        # Cháy lớn do xăng dầu
        {'temperature': 180, 'humidity': 20, 'gas_value': 1000, 'dust_value': 300, 'fire_sensor_status': 0, 'label': 'DANGEROUS_FIRE'}
    ]
    
    # Thêm các trường hợp đặc biệt vào dữ liệu
    for case in special_cases:
        for _ in range(100):  # Tăng trọng số lên 100
            data.append(case)
    
    # Xáo trộn dữ liệu
    random.shuffle(data)
    
    # Lưu dữ liệu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mock_training_data_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({'data': data}, f, indent=2)
    
    print(f"Đã tạo {len(data)} mẫu dữ liệu và lưu vào {filename}")
    
    # Phân tích dữ liệu
    analyze_mock_data(data)
    
    return data

def analyze_mock_data(data):
    """Phân tích dữ liệu training"""
    print("\n=== PHÂN TÍCH DỮ LIỆU ===")
    
    # Phân tích theo nhãn
    labels = [d['label'] for d in data]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nPhân bố nhãn:")
    for label, count in label_counts.items():
        print(f"{label}: {count} mẫu ({count/len(data)*100:.1f} %)")
    
    # Phân tích theo từng nhãn
    for label in label_counts.keys():
        label_data = [d for d in data if d['label'] == label]
        print(f"\nThống kê cho {label}:")
        print(f"Số lượng mẫu: {len(label_data)}")
        
        # Tính giá trị trung bình và độ lệch chuẩn
        temp_mean = sum(d['temperature'] for d in label_data) / len(label_data)
        temp_std = (sum((d['temperature'] - temp_mean) ** 2 for d in label_data) / len(label_data)) ** 0.5
        
        humidity_mean = sum(d['humidity'] for d in label_data) / len(label_data)
        humidity_std = (sum((d['humidity'] - humidity_mean) ** 2 for d in label_data) / len(label_data)) ** 0.5
        
        gas_mean = sum(d['gas_value'] for d in label_data) / len(label_data)
        gas_std = (sum((d['gas_value'] - gas_mean) ** 2 for d in label_data) / len(label_data)) ** 0.5
        
        dust_mean = sum(d['dust_value'] for d in label_data) / len(label_data)
        dust_std = (sum((d['dust_value'] - dust_mean) ** 2 for d in label_data) / len(label_data)) ** 0.5
        
        print(f"Nhiệt độ: {temp_mean:.1f}°C ± {temp_std:.1f}°C")
        print(f"Độ ẩm: {humidity_mean:.1f}% ± {humidity_std:.1f}%")
        print(f"Gas: {gas_mean:.1f} ppm ± {gas_std:.1f} ppm")
        print(f"Bụi: {dust_mean:.1f} µg/m³ ± {dust_std:.1f} µg/m³")
        
        # Phân tích Flame Sensor
        fire_sensor_count = sum(1 for d in label_data if d['fire_sensor_status'] == 0)  # 0 means fire detected
        print(f"Flame Sensor báo có lửa: {fire_sensor_count} mẫu ({fire_sensor_count/len(label_data)*100:.1f} %)")

if __name__ == "__main__":
    generate_mock_data() 