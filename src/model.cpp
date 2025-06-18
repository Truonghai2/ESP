#include "model.h"
#include "model_config.h"
#include "model_data.h"
#include "online_learning.h"

// Global variables
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Define tensor arena with proper alignment
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Static resolver to save memory - only include needed ops
static tflite::MicroMutableOpResolver<4> resolver;

// Define global variables
extern int fireStatus;  // Changed to extern since it's defined in main.cpp

// Quantization parameters
float output_scale = 0.0f;
int output_zero_point = 0;

// Online learning instance
OnlineLearner online_learner;

// Logic rules for fire detection based on scaler_params.json
bool isNormalCondition(float temperature, float humidity, float gas, float dust, int flame) {
    // Calculate normalized values using mean and scale from scaler_params.json
    float norm_temp = (temperature - 64.900457318116) / 49.710347473000475;
    float norm_hum = (humidity - 52.5554747653331) / 15.828152218044682;
    float norm_gas = (gas - 425.6221939247098) / 271.8186561646269;
    float norm_dust = (dust - 92.90548528076565) / 84.50510204114954;
    float norm_flame = (flame - 0.6033353186420488) / 0.4892052860723701;

    // Print normalized values for debugging
    Serial.println(F("Normalized values:"));
    Serial.print(F("  Temperature: ")); Serial.println(norm_temp, 4);
    Serial.print(F("  Humidity: ")); Serial.println(norm_hum, 4);
    Serial.print(F("  Gas: ")); Serial.println(norm_gas, 4);
    Serial.print(F("  Dust: ")); Serial.println(norm_dust, 4);
    Serial.print(F("  Flame: ")); Serial.println(norm_flame, 4);

    // Check if values are within normal range
    // Using tighter thresholds for more accurate detection
    if (norm_temp < -0.5 && norm_temp > -1.5 &&    // Temperature slightly below mean
        norm_hum > 1.0 && norm_hum < 2.0 &&        // Humidity above mean
        norm_gas < -1.0 && norm_gas > -2.0 &&      // Gas well below mean
        norm_dust < -1.0 && norm_dust > -2.0 &&    // Dust well below mean
        norm_flame < -1.0 && norm_flame > -2.0) {  // Flame well below mean
        Serial.println(F("  Normal conditions detected"));
        return true;
    }
    
    Serial.println(F("  Abnormal conditions detected"));
    return false;
}

void initAIModel() {
    // Map the model into a usable data structure
    model = tflite::GetModel(fire_detection_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println(F("Model schema mismatch!"));
        return;
    }

    // Add only the ops we need
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddQuantize();

    // Build an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println(F("AllocateTensors() failed"));
        return;
    }

    // Get pointers to the model's input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Verify tensor dimensions
    if (input->dims->size != 2 || input->dims->data[1] != 5) {
        Serial.println(F("Invalid input tensor dimensions"));
        return;
    }

    // Get quantization parameters from model
    if (output->type == kTfLiteInt8) {
        output_scale = output->params.scale;
        output_zero_point = output->params.zero_point;
    }

    // Print model details for debugging
    Serial.println(F("Model initialized successfully"));
    Serial.print(F("Input type: "));
    Serial.println(input->type);
    Serial.print(F("Output type: "));
    Serial.println(output->type);
    Serial.print(F("Output scale: "));
    Serial.println(output_scale, 8);
    Serial.print(F("Output zero point: "));
    Serial.println(output_zero_point);
    
    // Print online learning model info
    online_learner.print_model_info();
}

void runModelTests() {
    Serial.println(F("\n=== Running Model Tests ==="));
    
    // Test cases from test_model.py
    const float test_inputs[][5] PROGMEM = {
        {26.0f, 55.0f, 50.0f, 15.0f, 0.0f},    // Phòng điều hòa bình thường
        {35.0f, 65.0f, 150.0f, 25.0f, 1.0f},   // Phòng bếp khi nấu ăn nhẹ
        {45.0f, 70.0f, 300.0f, 40.0f, 1.0f},   // Phòng bếp khi chiên rán
        {60.0f, 75.0f, 400.0f, 50.0f, 1.0f},   // Phòng bếp khi nướng
        {85.0f, 40.0f, 600.0f, 100.0f, 1.0f},  // Cháy nhỏ do chập điện
        {120.0f, 30.0f, 800.0f, 200.0f, 1.0f}, // Cháy rác
        {180.0f, 20.0f, 1000.0f, 300.0f, 1.0f} // Cháy lớn do xăng dầu
    };

    const char* test_descriptions[] = {
        "Phòng điều hòa bình thường",
        "Phòng bếp khi nấu ăn nhẹ",
        "Phòng bếp khi chiên rán",
        "Phòng bếp khi nướng",
        "Cháy nhỏ do chập điện",
        "Cháy rác",
        "Cháy lớn do xăng dầu"
    };

    for (int i = 0; i < 7; i++) {
        Serial.print(F("\nTest Case "));
        Serial.print(i + 1);
        Serial.print(F(": "));
        Serial.println(test_descriptions[i]);
        
        float temp = pgm_read_float(&test_inputs[i][0]);
        float hum = pgm_read_float(&test_inputs[i][1]);
        float gas = pgm_read_float(&test_inputs[i][2]);
        float dust = pgm_read_float(&test_inputs[i][3]);
        int flame = pgm_read_float(&test_inputs[i][4]);
        
        Serial.print(F("Input Values: "));
        Serial.print(F("Temp="));
        Serial.print(temp, 1);
        Serial.print(F("°C, Hum="));
        Serial.print(hum, 1);
        Serial.print(F("%, Gas="));
        Serial.print(gas, 1);
        Serial.print(F("ppm, Dust="));
        Serial.print(dust, 1);
        Serial.print(F("μg/m³, Flame="));
        Serial.println(flame);
        
        // Get prediction from TensorFlow model
        int tf_result = predictFireStatus(temp, hum, gas, dust, flame);
        
        // Add sample to online learning model first
        online_learner.add_sample(temp, hum, gas, dust, flame, tf_result);
        
        // Then get prediction from online learning model
        int online_result = online_learner.predict(temp, hum, gas, dust, flame);
        
        Serial.print(F("TensorFlow Result: "));
        switch (tf_result) {
            case 0:
                Serial.println(F("NO FIRE"));
                break;
            case 1:
                Serial.println(F("COOKING FIRE"));
                break;
            case 2:
                Serial.println(F("DANGEROUS FIRE"));
                break;
            default:
                Serial.println(F("ERROR"));
        }
        
        Serial.print(F("Online Learning Result: "));
        switch (online_result) {
            case 0:
                Serial.println(F("NO FIRE"));
                break;
            case 1:
                Serial.println(F("COOKING FIRE"));
                break;
            case 2:
                Serial.println(F("DANGEROUS FIRE"));
                break;
            default:
                Serial.println(F("ERROR"));
        }
        
        Serial.println(F("------------------------"));
    }
    
    Serial.println(F("=== Model Tests Completed ===\n"));
    
    // Print final online learning model info
    online_learner.print_model_info();
}

int predictFireStatus(float temperature, float humidity, float gas, float dust, int flame) {
    if (!interpreter || !input || !output) {
        Serial.println(F("Model not properly initialized!"));
        return -1;
    }

    // Input validation with ranges from test_model.py
    if (temperature < MIN_TEMP || temperature > MAX_TEMP || 
        humidity < MIN_HUMIDITY || humidity > MAX_HUMIDITY ||
        gas < MIN_GAS || gas > MAX_GAS ||
        dust < MIN_DUST || dust > MAX_DUST ||
        (flame != 0 && flame != 1)) {
        Serial.println(F("Invalid input values"));
        return -1;
    }

    // Print input values
    Serial.println(F("Input values:"));
    Serial.print(F("  Temperature: "));
    Serial.print(temperature, 1);
    Serial.println(F("°C"));
    Serial.print(F("  Humidity: "));
    Serial.print(humidity, 1);
    Serial.println(F("%"));
    Serial.print(F("  Gas: "));
    Serial.print(gas, 1);
    Serial.println(F(" ppm"));
    Serial.print(F("  Dust: "));
    Serial.print(dust, 1);
    Serial.println(F(" µg/m³"));
    Serial.print(F("  Fire Sensor: "));
    Serial.println(flame, 1);
    Serial.println();

    // Check if conditions are normal
    if (isNormalCondition(temperature, humidity, gas, dust, flame)) {
        Serial.println(F("  Normal conditions detected - NO FIRE"));
        return 0;  // Return NO FIRE for normal conditions
    }

    // Normalize inputs using parameters from scaler_params.json
    float scaled_inputs[5];
    scaled_inputs[0] = (temperature - INPUT_MEAN[0]) / INPUT_SCALE[0];
    scaled_inputs[1] = (humidity - INPUT_MEAN[1]) / INPUT_SCALE[1];
    scaled_inputs[2] = (gas - INPUT_MEAN[2]) / INPUT_SCALE[2];
    scaled_inputs[3] = (dust - INPUT_MEAN[3]) / INPUT_SCALE[3];
    scaled_inputs[4] = (flame - INPUT_MEAN[4]) / INPUT_SCALE[4];

    // Print normalized inputs
    Serial.print(F("  Normalized input: ["));
    for (int i = 0; i < 5; i++) {
        Serial.print(scaled_inputs[i], 4);
        if (i < 4) Serial.print(F(", "));
    }
    Serial.println(F("]"));

    // Handle quantization for int8 model
    if (input->type == kTfLiteInt8) {
        // Get quantization parameters
        float input_scale = input->params.scale;
        int input_zero_point = input->params.zero_point;
        
        // Quantize input data
        int8_t quantized_inputs[5];
        for (int i = 0; i < 5; i++) {
            quantized_inputs[i] = static_cast<int8_t>(
                std::round(scaled_inputs[i] / input_scale + input_zero_point)
            );
        }
        
        // Print quantized inputs
        Serial.print(F("  Quantized input (int8): ["));
        for (int i = 0; i < 5; i++) {
            Serial.print(quantized_inputs[i]);
            if (i < 4) Serial.print(F(", "));
        }
        Serial.println(F("]"));
        
        // Copy quantized inputs to tensor
        std::memcpy(input->data.int8, quantized_inputs, sizeof(quantized_inputs));
    } else {
        // For float32 model
        float* input_data = input->data.f;
        for (int i = 0; i < 5; i++) {
            input_data[i] = scaled_inputs[i];
        }
    }

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println(F("Invoke failed!"));
        return -1;
    }

    // Get the output values
    float output_data[3];
    
    // Handle quantization for int8 model
    if (output->type == kTfLiteInt8) {
        // Get raw int8 output
        int8_t raw_output[3];
        std::memcpy(raw_output, output->data.int8, sizeof(raw_output));
    
        // Print raw output
        Serial.print(F("  Raw output (int8): ["));
        for (int i = 0; i < 3; i++) {
            Serial.print(raw_output[i]);
            if (i < 2) Serial.print(F(", "));
        }
        Serial.println(F("]"));
    
        // Dequantize output using model parameters
        for (int i = 0; i < 3; i++) {
            output_data[i] = (raw_output[i] - output_zero_point) * output_scale;
        }
    } else {
        // For float32 model
        float* output_data_ptr = output->data.f;
        for (int i = 0; i < 3; i++) {
            output_data[i] = output_data_ptr[i];
        }
    }
    
    // Print dequantized output
    Serial.print(F("  Dequantized output: ["));
    for (int i = 0; i < 3; i++) {
        Serial.print(output_data[i], 4);
        if (i < 2) Serial.print(F(", "));
    }
    Serial.println(F("]"));
    
    // Find the class with highest probability
    int max_index = 0;
    float max_prob = output_data[0];
    for (int i = 1; i < 3; i++) {
        if (output_data[i] > max_prob) {
            max_prob = output_data[i];
            max_index = i;
        }
    }
    
    // Print prediction and confidence
    Serial.print(F("  Prediction: "));
    switch (max_index) {
        case 0:
            Serial.print(F("NO_FIRE"));
            break;
        case 1:
            Serial.print(F("COOKING_FIRE"));
            break;
        case 2:
            Serial.print(F("DANGEROUS_FIRE"));
            break;
    }
    Serial.print(F(" (Confidence: "));
    Serial.print(max_prob * 100, 2);  // Convert to percentage like Python
    Serial.println(F("%)"));
    
    // Return prediction based on highest probability
    return max_index;
} 