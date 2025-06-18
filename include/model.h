#ifndef MODEL_H
#define MODEL_H

#include <Arduino.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Function declarations
void initAIModel();
void runModelTests();
int predictFireStatus(float temperature, float humidity, float gas, float dust, int flame);

#endif // MODEL_H 