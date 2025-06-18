#include <Arduino.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "config.h"
#include "sensors.h"
#include "webserver.h"
#include "data_collection.h"
#include "model.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_system.h>
#include <esp_private/esp_clk.h>

// Brownout detector register
#define RTC_CNTL_BROWN_OUT_REG 0x3FF48084

// Task handles
TaskHandle_t sensorTaskHandle = NULL;
TaskHandle_t webServerTaskHandle = NULL;

// Global variables
int fireStatus = FIRE_STATUS_NO_FIRE;  // Initialize fire status

// Function to check power supply
void checkPowerSupply() {
    float voltage = analogRead(PINS::MP2_ANALOG) * (3.3 / 4095.0) * 2;  // Using MP2 analog pin for voltage reading
    Serial.print("Power supply voltage: ");
    Serial.print(voltage);
    Serial.println("V");
    
    if (voltage < 3.0) {
        Serial.println("WARNING: Low power supply voltage!");
        Serial.println("Please check power supply and connections.");
        delay(2000);
    }
}

// Task for handling web server
void webServerTask(void *pvParameters) {
    while (1) {
        // Check WiFi connection
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("WiFi connection lost. Reconnecting...");
            WiFi.reconnect();
            delay(5000);
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void setup() {
    // Disable brownout detector
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    // Initialize serial communication
    Serial.begin(115200);
    delay(1000);  // Give time for serial to initialize
    
    Serial.println("\nFire Early Warning System Starting...");
    
    // Check power supply
    checkPowerSupply();

    // Initialize sensors
    initSensors();

    // Initialize AI model
    initAIModel();

    // Connect to WiFi with timeout
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting to WiFi");
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nConnected to WiFi");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nFailed to connect to WiFi");
        Serial.println("Continuing in offline mode...");
    }

    // Initialize WebServer
    initWebServer();

    // Run model tests
    runModelTests();

    // Create tasks
    xTaskCreate(
        sensorTask,           // Task function
        "SensorTask",         // Task name
        4096,                // Stack size
        NULL,                // Task parameters
        1,                   // Task priority
        &sensorTaskHandle    // Task handle
    );

    xTaskCreate(
        webServerTask,     // Task function
        "WebServerTask",   // Task name
        4096,             // Stack size
        NULL,             // Task parameters
        1,                // Task priority
        &webServerTaskHandle // Task handle
    );
}

void loop() {
    // Main loop is empty as tasks handle everything
    vTaskDelay(pdMS_TO_TICKS(1000));
}