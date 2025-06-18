#include "online_learning.h"
#include "model_config.h"

// Adam optimizer parameters
const float BETA1 = 0.9f;
const float BETA2 = 0.999f;
const float EPSILON = 1e-7f;
const float GRADIENT_CLIP = 1.0f;
const float L2_REGULARIZATION = 0.01f;

// Early stopping parameters
const int PATIENCE = 5;
const float MIN_DELTA = 0.0005f;

// Time constants for season calculation
const unsigned long MILLIS_PER_DAY = 24UL * 60UL * 60UL * 1000UL;
const unsigned long MILLIS_PER_MONTH = 30UL * MILLIS_PER_DAY;

OnlineLearner::OnlineLearner() {
    // Initialize model parameters
    model.learning_rate = 0.01f;
    model.sample_count = 0;
    model.accuracy = 0.0f;
    model.best_loss = 1e10f;
    model.patience_counter = 0;
    model.t = 0;
    model.adaptation_rate = 0.1f;  // 10% adaptation rate
    
    // Initialize seasonal stats
    for (int i = 0; i < 4; i++) {
        model.seasonal_stats[i].sample_count = 0;
        model.seasonal_stats[i].temp_mean = 25.0f;  // Default room temperature
        model.seasonal_stats[i].temp_std = 5.0f;
        model.seasonal_stats[i].humidity_mean = 50.0f;  // Default humidity
        model.seasonal_stats[i].humidity_std = 10.0f;
        model.seasonal_stats[i].gas_mean = 0.0f;
        model.seasonal_stats[i].gas_std = 1.0f;
        model.seasonal_stats[i].dust_mean = 0.0f;
        model.seasonal_stats[i].dust_std = 1.0f;
    }
    
    // Initialize weights with Xavier/Glorot initialization
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            model.weights1[i][j] = (random(1000) / 500.0f - 1.0f) * sqrt(2.0f / (5 + HIDDEN_LAYER1_SIZE));
            model.m1[i][j] = 0.0f;
            model.v1[i][j] = 0.0f;
        }
    }
    
    // Khởi tạo SPIFFS nếu chưa có
    if(!SPIFFS.begin(true)) {
        Serial.println("SPIFFS Mount Failed");
        return;
    }
    
    // Tải mô hình nếu có
    load_model();
}

int OnlineLearner::get_current_season() {
    // Get current month (1-12) using unsigned long to avoid overflow
    unsigned long current_time = millis();
    int month = ((current_time / MILLIS_PER_MONTH) % 12) + 1;
    
    // Determine season based on month
    if (month >= 3 && month <= 5) return 1;  // Spring
    if (month >= 6 && month <= 8) return 2;  // Summer
    if (month >= 9 && month <= 11) return 3;  // Fall
    return 0;  // Winter
}

void OnlineLearner::update_seasonal_stats(const LearningSample& sample) {
    int season = sample.season;
    SeasonalStats& stats = model.seasonal_stats[season];
    
    // Update count
    stats.sample_count++;
    
    // Update means using exponential moving average
    float alpha = model.adaptation_rate;
    
    // Update means with clipping to prevent extreme values
    float new_temp_mean = (1 - alpha) * stats.temp_mean + alpha * sample.temperature;
    float new_humidity_mean = (1 - alpha) * stats.humidity_mean + alpha * sample.humidity;
    float new_gas_mean = (1 - alpha) * stats.gas_mean + alpha * sample.gas;
    float new_dust_mean = (1 - alpha) * stats.dust_mean + alpha * sample.dust;
    
    // Clip means to reasonable ranges
    stats.temp_mean = constrain(new_temp_mean, 0.0f, 100.0f);
    stats.humidity_mean = constrain(new_humidity_mean, 0.0f, 100.0f);
    stats.gas_mean = constrain(new_gas_mean, 0.0f, 1000.0f);
    stats.dust_mean = constrain(new_dust_mean, 0.0f, 500.0f);
    
    // Update standard deviations with minimum values
    float temp_diff = sample.temperature - stats.temp_mean;
    float humidity_diff = sample.humidity - stats.humidity_mean;
    float gas_diff = sample.gas - stats.gas_mean;
    float dust_diff = sample.dust - stats.dust_mean;
    
    // Calculate new standard deviations with minimum values
    float new_temp_std = sqrt((1 - alpha) * stats.temp_std * stats.temp_std + alpha * temp_diff * temp_diff);
    float new_humidity_std = sqrt((1 - alpha) * stats.humidity_std * stats.humidity_std + alpha * humidity_diff * humidity_diff);
    float new_gas_std = sqrt((1 - alpha) * stats.gas_std * stats.gas_std + alpha * gas_diff * gas_diff);
    float new_dust_std = sqrt((1 - alpha) * stats.dust_std * stats.dust_std + alpha * dust_diff * dust_diff);
    
    // Set minimum standard deviations to prevent division by very small numbers
    stats.temp_std = max(new_temp_std, 5.0f);  // At least 5°C
    stats.humidity_std = max(new_humidity_std, 5.0f);  // At least 5%
    stats.gas_std = max(new_gas_std, 50.0f);  // At least 50 ppm
    stats.dust_std = max(new_dust_std, 25.0f);  // At least 25 µg/m³
    
    // Debug print
    if(stats.sample_count % 10 == 0) {  // Print every 10 samples
        Serial.printf("\nSeasonal Stats Update (Season %d):\n", season);
        Serial.printf("Temperature: %.1f°C ± %.1f°C\n", stats.temp_mean, stats.temp_std);
        Serial.printf("Humidity: %.1f%% ± %.1f%%\n", stats.humidity_mean, stats.humidity_std);
        Serial.printf("Gas: %.1f ppm ± %.1f ppm\n", stats.gas_mean, stats.gas_std);
        Serial.printf("Dust: %.1f µg/m³ ± %.1f µg/m³\n", stats.dust_mean, stats.dust_std);
    }
}

void OnlineLearner::adapt_to_season(float* features) {
    int current_season = get_current_season();
    const SeasonalStats& stats = model.seasonal_stats[current_season];
    
    // First normalize using seasonal statistics to get values in reasonable range
    float temp_normalized = (features[0] - stats.temp_mean) / (stats.temp_std + EPSILON);
    float humidity_normalized = (features[1] - stats.humidity_mean) / (stats.humidity_std + EPSILON);
    float gas_normalized = (features[2] - stats.gas_mean) / (stats.gas_std + EPSILON);
    float dust_normalized = (features[3] - stats.dust_mean) / (stats.dust_std + EPSILON);
    
    // Clip normalized values to prevent extreme values
    temp_normalized = constrain(temp_normalized, -3.0f, 3.0f);
    humidity_normalized = constrain(humidity_normalized, -3.0f, 3.0f);
    gas_normalized = constrain(gas_normalized, -3.0f, 3.0f);
    dust_normalized = constrain(dust_normalized, -3.0f, 3.0f);
    
    // Then apply global normalization to match TensorFlow's scale
    features[0] = (temp_normalized - INPUT_MEAN[0]) * INPUT_SCALE[0];
    features[1] = (humidity_normalized - INPUT_MEAN[1]) * INPUT_SCALE[1];
    features[2] = (gas_normalized - INPUT_MEAN[2]) * INPUT_SCALE[2];
    features[3] = (dust_normalized - INPUT_MEAN[3]) * INPUT_SCALE[3];
    features[4] = features[4] * INPUT_SCALE[4];  // Flame sensor
    
    // Debug print
    Serial.println("Seasonal Normalization Debug:");
    Serial.printf("Raw values: [%.1f, %.1f, %.1f, %.1f, %d]\n", 
                 features[0], features[1], features[2], features[3], (int)features[4]);
    Serial.printf("Seasonal normalized: [%.4f, %.4f, %.4f, %.4f]\n",
                 temp_normalized, humidity_normalized, gas_normalized, dust_normalized);
    Serial.printf("Final normalized: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                 features[0], features[1], features[2], features[3], features[4]);
}

void OnlineLearner::add_sample(float temperature, float humidity, float gas, float dust, int flame, int label) {
    // Create new sample
    LearningSample sample;
    sample.temperature = temperature;
    sample.humidity = humidity;
    sample.gas = gas;
    sample.dust = dust;
    sample.flame = flame;
    sample.label = label;
    sample.timestamp = millis();
    sample.season = get_current_season();
    
    // Update seasonal statistics
    update_seasonal_stats(sample);
    
    // Add to buffer
    samples[sample_index] = sample;
    sample_index = (sample_index + 1) % BUFFER_SIZE;
    model.sample_count++;
    
    // Train model with new sample
    train();
}

void OnlineLearner::train() {
    if(model.sample_count < 2) return;
    
    float features[FEATURE_COUNT];
    float hidden1[HIDDEN_LAYER1_SIZE];
    float hidden2[HIDDEN_LAYER2_SIZE];
    float hidden3[HIDDEN_LAYER3_SIZE];
    float output[CLASS_COUNT];
    float gradients[CLASS_COUNT];
    int correct_predictions = 0;
    float total_loss = 0.0;
    
    // Huấn luyện với tất cả mẫu trong buffer
    for(int i = 0; i < model.sample_count; i++) {
        // Chuẩn bị features
        features[0] = samples[i].temperature;
        features[1] = samples[i].humidity;
        features[2] = samples[i].gas;
        features[3] = samples[i].dust;
        features[4] = samples[i].flame;
        
        // Chuẩn hóa features
        normalize_features(features);
        
        // Forward pass
        // Layer 1: Input -> Hidden1
        for(int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            hidden1[j] = model.bias1[j];
            for(int k = 0; k < FEATURE_COUNT; k++) {
                hidden1[j] += features[k] * model.weights1[k][j];
            }
            hidden1[j] = relu(hidden1[j]);
        }
        
        // Layer 2: Hidden1 -> Hidden2
        for(int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            hidden2[j] = model.bias2[j];
            for(int k = 0; k < HIDDEN_LAYER1_SIZE; k++) {
                hidden2[j] += hidden1[k] * model.weights2[k][j];
            }
            hidden2[j] = relu(hidden2[j]);
        }
        
        // Layer 3: Hidden2 -> Hidden3
        for(int j = 0; j < HIDDEN_LAYER3_SIZE; j++) {
            hidden3[j] = model.bias3[j];
            for(int k = 0; k < HIDDEN_LAYER2_SIZE; k++) {
                hidden3[j] += hidden2[k] * model.weights3[k][j];
            }
            hidden3[j] = relu(hidden3[j]);
        }
        
        // Layer 4: Hidden3 -> Output
        for(int j = 0; j < CLASS_COUNT; j++) {
            output[j] = model.bias4[j];
            for(int k = 0; k < HIDDEN_LAYER3_SIZE; k++) {
                output[j] += hidden3[k] * model.weights4[k][j];
            }
        }
        
        // Softmax
        float sum = 0;
        for(int j = 0; j < CLASS_COUNT; j++) {
            output[j] = exp(output[j]);
            sum += output[j];
        }
        for(int j = 0; j < CLASS_COUNT; j++) {
            output[j] /= sum;
        }
        
        // Tính loss và gradients
        float loss = 0;
        for(int j = 0; j < CLASS_COUNT; j++) {
            loss -= (j == samples[i].label ? log(output[j]) : 0);
            gradients[j] = output[j] - (j == samples[i].label ? 1.0 : 0.0);
        }
        total_loss += loss;
        
        // Backpropagation với Adam optimizer
        model.t++;
        float lr_t = model.learning_rate * sqrt(1 - pow(BETA2, model.t)) / (1 - pow(BETA1, model.t));
        
        // Layer 4 gradients
        float grad_hidden3[HIDDEN_LAYER3_SIZE] = {0};
        for(int j = 0; j < HIDDEN_LAYER3_SIZE; j++) {
            for(int k = 0; k < CLASS_COUNT; k++) {
                float grad = gradients[k] * hidden3[j];
                // Adam update
                model.m4[j][k] = BETA1 * model.m4[j][k] + (1 - BETA1) * grad;
                model.v4[j][k] = BETA2 * model.v4[j][k] + (1 - BETA2) * grad * grad;
                float m_hat = model.m4[j][k] / (1 - pow(BETA1, model.t));
                float v_hat = model.v4[j][k] / (1 - pow(BETA2, model.t));
                model.weights4[j][k] -= lr_t * m_hat / (sqrt(v_hat) + EPSILON);
            }
        }
        
        // Layer 3 gradients
        float grad_hidden2[HIDDEN_LAYER2_SIZE] = {0};
        for(int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
            for(int k = 0; k < HIDDEN_LAYER3_SIZE; k++) {
                float grad = grad_hidden3[k] * relu_derivative(hidden3[k]) * hidden2[j];
                model.m3[j][k] = BETA1 * model.m3[j][k] + (1 - BETA1) * grad;
                model.v3[j][k] = BETA2 * model.v3[j][k] + (1 - BETA2) * grad * grad;
                float m_hat = model.m3[j][k] / (1 - pow(BETA1, model.t));
                float v_hat = model.v3[j][k] / (1 - pow(BETA2, model.t));
                model.weights3[j][k] -= lr_t * m_hat / (sqrt(v_hat) + EPSILON);
            }
        }
        
        // Layer 2 gradients
        float grad_hidden1[HIDDEN_LAYER1_SIZE] = {0};
        for(int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            for(int k = 0; k < HIDDEN_LAYER2_SIZE; k++) {
                float grad = grad_hidden2[k] * relu_derivative(hidden2[k]) * hidden1[j];
                model.m2[j][k] = BETA1 * model.m2[j][k] + (1 - BETA1) * grad;
                model.v2[j][k] = BETA2 * model.v2[j][k] + (1 - BETA2) * grad * grad;
                float m_hat = model.m2[j][k] / (1 - pow(BETA1, model.t));
                float v_hat = model.v2[j][k] / (1 - pow(BETA2, model.t));
                model.weights2[j][k] -= lr_t * m_hat / (sqrt(v_hat) + EPSILON);
            }
        }
        
        // Layer 1 gradients
        for(int j = 0; j < FEATURE_COUNT; j++) {
            for(int k = 0; k < HIDDEN_LAYER1_SIZE; k++) {
                float grad = grad_hidden1[k] * relu_derivative(hidden1[k]) * features[j];
                model.m1[j][k] = BETA1 * model.m1[j][k] + (1 - BETA1) * grad;
                model.v1[j][k] = BETA2 * model.v1[j][k] + (1 - BETA2) * grad * grad;
                float m_hat = model.m1[j][k] / (1 - pow(BETA1, model.t));
                float v_hat = model.v1[j][k] / (1 - pow(BETA2, model.t));
                model.weights1[j][k] -= lr_t * m_hat / (sqrt(v_hat) + EPSILON);
            }
        }
        
        // Kiểm tra độ chính xác
        int predicted_class = 0;
        float max_prob = output[0];
        for(int j = 1; j < CLASS_COUNT; j++) {
            if(output[j] > max_prob) {
                max_prob = output[j];
                predicted_class = j;
            }
        }
        if(predicted_class == samples[i].label) {
            correct_predictions++;
        }
    }
    
    // Cập nhật độ chính xác
    model.accuracy = (float)correct_predictions / model.sample_count;
    
    // Early stopping
    float avg_loss = total_loss / model.sample_count;
    if(avg_loss < model.best_loss - MIN_DELTA) {
        model.best_loss = avg_loss;
        model.patience_counter = 0;
        save_model();  // Lưu model tốt nhất
    } else {
        model.patience_counter++;
        if(model.patience_counter >= PATIENCE) {
            // Reduce learning rate
            model.learning_rate *= 0.2;
            model.patience_counter = 0;
        }
    }
    
    // In thông tin debug
    Serial.println(F("\nOnline Learning Training Debug:"));
    Serial.print(F("Sample count: ")); Serial.println(model.sample_count);
    Serial.print(F("Accuracy: ")); Serial.println(model.accuracy);
    Serial.print(F("Loss: ")); Serial.println(avg_loss);
    Serial.print(F("Learning rate: ")); Serial.println(model.learning_rate);
}

int OnlineLearner::predict(float temperature, float humidity, float gas, float dust, int flame) {
    float features[FEATURE_COUNT] = {temperature, humidity, gas, dust, (float)flame};
    float hidden1[HIDDEN_LAYER1_SIZE];
    float hidden2[HIDDEN_LAYER2_SIZE];
    float hidden3[HIDDEN_LAYER3_SIZE];
    float output[CLASS_COUNT];
    
    // Chuẩn hóa features
    normalize_features(features);
    
    // Forward pass
    // Layer 1: Input -> Hidden1
    for(int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
        hidden1[j] = model.bias1[j];
        for(int k = 0; k < FEATURE_COUNT; k++) {
            hidden1[j] += features[k] * model.weights1[k][j];
        }
        hidden1[j] = relu(hidden1[j]);
    }
    
    // Layer 2: Hidden1 -> Hidden2
    for(int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
        hidden2[j] = model.bias2[j];
        for(int k = 0; k < HIDDEN_LAYER1_SIZE; k++) {
            hidden2[j] += hidden1[k] * model.weights2[k][j];
        }
        hidden2[j] = relu(hidden2[j]);
    }
    
    // Layer 3: Hidden2 -> Hidden3
    for(int j = 0; j < HIDDEN_LAYER3_SIZE; j++) {
        hidden3[j] = model.bias3[j];
        for(int k = 0; k < HIDDEN_LAYER2_SIZE; k++) {
            hidden3[j] += hidden2[k] * model.weights3[k][j];
        }
        hidden3[j] = relu(hidden3[j]);
    }
    
    // Layer 4: Hidden3 -> Output
    for(int j = 0; j < CLASS_COUNT; j++) {
        output[j] = model.bias4[j];
        for(int k = 0; k < HIDDEN_LAYER3_SIZE; k++) {
            output[j] += hidden3[k] * model.weights4[k][j];
        }
    }
    
    // Softmax
    float sum = 0;
    for(int j = 0; j < CLASS_COUNT; j++) {
        output[j] = exp(output[j]);
        sum += output[j];
    }
    for(int j = 0; j < CLASS_COUNT; j++) {
        output[j] /= sum;
    }
    
    // In thông tin debug
    Serial.println(F("Online Learning Debug:"));
    Serial.print(F("Normalized features: ["));
    for(int i = 0; i < FEATURE_COUNT; i++) {
        Serial.print(features[i], 4);
        if(i < FEATURE_COUNT-1) Serial.print(F(", "));
    }
    Serial.println(F("]"));
    
    Serial.print(F("Predictions: ["));
    for(int i = 0; i < CLASS_COUNT; i++) {
        Serial.print(output[i], 4);
        if(i < CLASS_COUNT-1) Serial.print(F(", "));
    }
    Serial.println(F("]"));
    
    // Tìm class có xác suất cao nhất
    int predicted_class = 0;
    float max_prob = output[0];
    for(int i = 1; i < CLASS_COUNT; i++) {
        if(output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}

void OnlineLearner::save_model() {
    save_model_to_flash();
}

void OnlineLearner::load_model() {
    load_model_from_flash();
}

float OnlineLearner::get_accuracy() {
    return model.accuracy;
}

int OnlineLearner::get_sample_count() {
    return model.sample_count;
}

void OnlineLearner::print_model_info() {
    Serial.println("\n=== Model Information ===");
    Serial.print("Sample Count: "); Serial.println(model.sample_count);
    Serial.print("Accuracy: "); Serial.println(model.accuracy);
    Serial.print("Learning Rate: "); Serial.println(model.learning_rate);
    Serial.println("Weights:");
    for(int i = 0; i < FEATURE_COUNT; i++) {
        Serial.print("  Feature "); Serial.print(i); Serial.print(": ");
        for(int j = 0; j < CLASS_COUNT; j++) {
            Serial.print(model.weights1[i][j], 4); Serial.print(F(" "));
        }
        Serial.println();
    }
    Serial.println("Bias:");
    for(int i = 0; i < CLASS_COUNT; i++) {
        Serial.print("  Class "); Serial.print(i); Serial.print(": ");
        Serial.println(model.bias1[i], 4);
    }
    Serial.println("=====================\n");
}

// Helper functions
float OnlineLearner::relu(float x) {
    return x > 0 ? x : 0;
}

float OnlineLearner::relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

void OnlineLearner::normalize_features(float* features) {
    // Apply seasonal adaptation first
    adapt_to_season(features);
    
    // Debug print
    Serial.println("Online Learning Debug:");
    Serial.print("Normalized features: [");
    for(int i = 0; i < FEATURE_COUNT; i++) {
        Serial.print(features[i], 4);
        if(i < FEATURE_COUNT - 1) Serial.print(", ");
    }
    Serial.println("]");
}

void OnlineLearner::save_model_to_flash() {
    File file = SPIFFS.open("/model.json", "w");
    if(!file) {
        Serial.println("Failed to open model file for writing");
        return;
    }
    
    StaticJsonDocument<1024> doc;
    
    // Lưu thông tin mô hình
    doc["learning_rate"] = model.learning_rate;
    doc["sample_count"] = model.sample_count;
    doc["accuracy"] = model.accuracy;
    
    // Lưu weights
    JsonArray weights = doc.createNestedArray("weights1");
    for(int i = 0; i < FEATURE_COUNT; i++) {
        JsonArray row = weights.createNestedArray();
        for(int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            row.add(model.weights1[i][j]);
        }
    }
    
    // Lưu bias
    JsonArray bias = doc.createNestedArray("bias1");
    for(int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        bias.add(model.bias1[i]);
    }
    
    // Ghi ra file
    if(serializeJson(doc, file) == 0) {
        Serial.println("Failed to write model to file");
    }
    
    file.close();
}

void OnlineLearner::load_model_from_flash() {
    if(!SPIFFS.exists("/model.json")) {
        Serial.println("No model file found");
        return;
    }
    
    File file = SPIFFS.open("/model.json", "r");
    if(!file) {
        Serial.println("Failed to open model file for reading");
        return;
    }
    
    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, file);
    if(error) {
        Serial.println("Failed to parse model file");
        file.close();
        return;
    }
    
    // Đọc thông tin mô hình
    model.learning_rate = doc["learning_rate"];
    model.sample_count = doc["sample_count"];
    model.accuracy = doc["accuracy"];
    
    // Đọc weights
    JsonArray weights = doc["weights1"];
    for(int i = 0; i < FEATURE_COUNT; i++) {
        JsonArray row = weights[i];
        for(int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
            model.weights1[i][j] = row[j];
        }
    }
    
    // Đọc bias
    JsonArray bias = doc["bias1"];
    for(int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
        model.bias1[i] = bias[i];
    }
    
    file.close();
}

void OnlineLearner::print_seasonal_stats() {
    const char* seasons[] = {"Winter", "Spring", "Summer", "Fall"};
    
    for (int i = 0; i < 4; i++) {
        const SeasonalStats& stats = model.seasonal_stats[i];
        Serial.printf("\nSeason: %s\n", seasons[i]);
        Serial.printf("Samples: %d\n", stats.sample_count);
        Serial.printf("Temperature: %.1f°C ± %.1f°C\n", stats.temp_mean, stats.temp_std);
        Serial.printf("Humidity: %.1f%% ± %.1f%%\n", stats.humidity_mean, stats.humidity_std);
        Serial.printf("Gas: %.2f ± %.2f\n", stats.gas_mean, stats.gas_std);
        Serial.printf("Dust: %.2f ± %.2f\n", stats.dust_mean, stats.dust_std);
    }
} 