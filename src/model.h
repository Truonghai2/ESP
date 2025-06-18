#ifndef MODEL_H
#define MODEL_H

#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// Model data declarations
extern const unsigned char fire_detection_model_tflite[];
extern const unsigned int fire_detection_model_tflite_len;

// Global AI model variables
extern const tflite::Model* model;
extern tflite::MicroInterpreter* interpreter;
extern TfLiteTensor* input;
extern TfLiteTensor* output;

// Keep aligned to 16 bytes for CMSIS
alignas(16) extern uint8_t tensor_arena[];

// Khai báo biến toàn cục
extern int fireStatus;

// Model configuration
#define MODEL_VERSION "1.0.0"

// Sensor thresholds
#define TEMP_THRESHOLD_COOKING (35.0f)  // Nhiệt độ ngưỡng nấu ăn (từ test case 2)
#define TEMP_THRESHOLD_DANGER (85.0f)   // Nhiệt độ ngưỡng nguy hiểm (từ test case 5)
#define HUM_THRESHOLD (80.0f)          // Độ ẩm ngưỡng (giữ nguyên)
#define GAS_THRESHOLD_COOKING (150.0f)  // Khí gas ngưỡng nấu ăn (từ test case 2)
#define GAS_THRESHOLD_DANGER (600.0f)   // Khí gas ngưỡng nguy hiểm (từ test case 5)
#define DUST_THRESHOLD (100.0f)        // Bụi ngưỡng (từ test case 5)
#define FLAME_THRESHOLD (0)           // Ngưỡng cảm biến lửa (0 = phát hiện lửa)

// Fire status codes
#define FIRE_STATUS_NO_FIRE 0
#define FIRE_STATUS_COOKING 1
#define FIRE_STATUS_DANGER 2

// Function declarations
void initAIModel();
void runModelTests();
int predictFireStatus(float temp, float hum, float gas, float dust, int flame);
int analyzeFireStatus(float temp, float hum, float gas, float dust, int flame);

#endif // MODEL_H
