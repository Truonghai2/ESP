#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// Tensor arena size - optimized for ESP32
constexpr int kTensorArenaSize = 8 * 1024;  // 8KB for tensor arena

// Model optimization flags
#define TFLITE_USE_OPTIMIZED_ALLOCATOR 1
#define TFLITE_USE_SIMPLE_MEMORY_PLANNER 1
#define TFLITE_DISABLE_X86_NEON 1
#define TFLITE_DISABLE_GPU 1
#define TFLITE_DISABLE_NNAPI 1
#define TFLITE_DISABLE_RUY 1
#define TFLITE_DISABLE_GEMMLOWP 1
#define TFLITE_DISABLE_FLOAT 1  // Use int8 quantization

// Debug settings
#define TFLITE_DEBUG 0  // Disable debug prints in production

// Input normalization parameters
const float INPUT_MEAN[] = {64.900457318116f, 52.5554747653331f, 425.6221939247098f, 92.90548528076565f, 0.6033353186420488f};
const float INPUT_SCALE[] = {49.710347473000475f, 15.828152218044682f, 271.8186561646269f, 84.50510204114954f, 0.4892052860723701f};

// Model output classes
enum class FireClass {
    NO_FIRE = 0,
    COOKING_FIRE = 1,
    DANGEROUS_FIRE = 2
};

// Input validation ranges
constexpr float MIN_TEMP = 0.0f;
constexpr float MAX_TEMP = 200.0f;
constexpr float MIN_HUMIDITY = 0.0f;
constexpr float MAX_HUMIDITY = 100.0f;
constexpr float MIN_GAS = 0.0f;
constexpr float MAX_GAS = 1000.0f;
constexpr float MIN_DUST = 0.0f;
constexpr float MAX_DUST = 300.0f;

#endif // MODEL_CONFIG_H 