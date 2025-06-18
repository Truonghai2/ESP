#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>
#include <DHT.h>
#include "config.h"

// Global sensor readings
extern volatile float current_humidity;
extern volatile float current_temperature;
extern volatile float current_mq2_ppm;
extern volatile int current_flame_value;
extern volatile int current_mp02_value;

// Moving average for DHT
extern float humidity_readings[DHT_SAMPLE_COUNT];
extern float temp_readings[DHT_SAMPLE_COUNT];
extern int dht_index;

// Sensor objects
extern DHT dht;

// Sensor reading functions
void readDHT11();
void readMQ2();
void readMP2();
void readFlame();
void readAllSensors();

// Task functions
void dhtTask(void *pvParameters);
void mq2Task(void *pvParameters);
void flameSensorTask(void *pvParameters);
void mp02Task(void *pvParameters);
void sensorTask(void *pvParameters);

// Sensor initialization
void initSensors();

#endif // SENSORS_H