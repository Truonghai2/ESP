#ifndef DATA_COLLECTION_H
#define DATA_COLLECTION_H

#include <Arduino.h>
#include "config.h"

// Structure to store training data
struct TrainingSample {
    float temperature;
    float humidity;
    float gas_value;
    float dust_value;
    int fire_sensor_status;
    int label;  // 0: NO_FIRE, 1: COOKING_FIRE, 2: DANGEROUS_FIRE
    unsigned long timestamp;
};

// Function declarations
void addTrainingSample(float temp, float hum, float gas, float dust, int flame, int predicted_label);
void sendTrainingData(TrainingSample* samples, int count);

#endif // DATA_COLLECTION_H 