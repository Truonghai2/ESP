#ifndef ONLINE_LEARNING_H
#define ONLINE_LEARNING_H

#include <Arduino.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>

// Neural network structure
const int HIDDEN_LAYER1_SIZE = 32;
const int HIDDEN_LAYER2_SIZE = 16;
const int HIDDEN_LAYER3_SIZE = 8;

// Seasonal adaptation
struct SeasonalStats {
    float temp_mean;
    float temp_std;
    float humidity_mean;
    float humidity_std;
    float gas_mean;
    float gas_std;
    float dust_mean;
    float dust_std;
    int sample_count;
};

// Cấu trúc dữ liệu cho một mẫu học
struct LearningSample {
    float temperature;
    float humidity;
    float gas;
    float dust;
    int flame;
    int label;  // 0: NO_FIRE, 1: COOKING_FIRE, 2: DANGEROUS_FIRE
    unsigned long timestamp;
    int season;  // 0: Winter, 1: Spring, 2: Summer, 3: Fall
};

// Cấu trúc dữ liệu cho mô hình
struct Model {
    // Layer 1: Input -> Hidden1
    float weights1[5][32];  // 5 features x 32 neurons
    float bias1[32];
    float m1[5][32];  // First moment for Adam
    float v1[5][32];  // Second moment for Adam
    float m_bias1[32];
    float v_bias1[32];
    
    // Layer 2: Hidden1 -> Hidden2
    float weights2[32][16];  // 32 neurons x 16 neurons
    float bias2[16];
    float m2[32][16];
    float v2[32][16];
    float m_bias2[16];
    float v_bias2[16];
    
    // Layer 3: Hidden2 -> Hidden3
    float weights3[16][8];  // 16 neurons x 8 neurons
    float bias3[8];
    float m3[16][8];
    float v3[16][8];
    float m_bias3[8];
    float v_bias3[8];
    
    // Layer 4: Hidden3 -> Output
    float weights4[8][3];  // 8 neurons x 3 classes
    float bias4[3];
    float m4[8][3];
    float v4[8][3];
    float m_bias4[3];
    float v_bias4[3];
    
    float learning_rate;
    int sample_count;
    float accuracy;
    float best_loss;
    int patience_counter;
    int t;  // Time step for Adam
    
    // Seasonal adaptation
    SeasonalStats seasonal_stats[4];  // Stats for each season
    float adaptation_rate;  // Rate at which seasonal stats are updated
};

class OnlineLearner {
private:
    static const int BUFFER_SIZE = 50;  // Số mẫu tối đa lưu trong bộ nhớ
    static const int FEATURE_COUNT = 5;  // Số đặc trưng
    static const int CLASS_COUNT = 3;    // Số lớp
    
    LearningSample samples[BUFFER_SIZE];
    int sample_index;
    Model model;
    
    // Hàm tiện ích
    float relu(float x);
    float relu_derivative(float x);
    void normalize_features(float* features);
    void save_model_to_flash();
    void load_model_from_flash();
    
    // Seasonal adaptation functions
    int get_current_season();
    void update_seasonal_stats(const LearningSample& sample);
    void adapt_to_season(float* features);
    
public:
    OnlineLearner();
    
    // Thêm mẫu học mới
    void add_sample(float temperature, float humidity, float gas, float dust, int flame, int label);
    
    // Huấn luyện mô hình với mẫu mới
    void train();
    
    // Dự đoán với mô hình hiện tại
    int predict(float temperature, float humidity, float gas, float dust, int flame);
    
    // Lưu mô hình
    void save_model();
    
    // Tải mô hình
    void load_model();
    
    // Lấy thông tin mô hình
    float get_accuracy();
    int get_sample_count();
    
    // In thông tin mô hình
    void print_model_info();
    
    // Seasonal adaptation
    void update_seasonal_adaptation();
    void print_seasonal_stats();
};

#endif