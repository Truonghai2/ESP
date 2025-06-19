#include "data_collection.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "config.h"

// Circular buffer for storing samples
TrainingSample collectedSamples[MAX_STORED_SAMPLES];
int sampleCount = 0;
int currentSampleIndex = 0;
unsigned long lastCollectionTime = 0;

void addTrainingSample(float temp, float hum, float gas, float dust, int flame, int predicted_label) {
    unsigned long currentTime = millis();
    
    Serial.println("\n=== Adding Training Sample ===");
    
    // Prepare a single-sample buffer
    TrainingSample sample = {
        temp,
        hum,
        gas,
        dust,
        flame,
        predicted_label,
        currentTime
    };
    
    // Send this sample immediately
    sendTrainingData(&sample, 1);
    Serial.println("===========================\n");
}

// Overload sendTrainingData to accept a pointer to sample(s) and count
void sendTrainingData(TrainingSample* samples, int count) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi not connected. Cannot send data.");
        return;
    }
    
    Serial.println("\n=== Sending Data to Server ===");
    Serial.print("Server URL: "); Serial.println(SERVER_URL);
    
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    
    // Create JSON document
    StaticJsonDocument<1024> doc;
    
    // Add device information
    JsonObject device_info = doc.createNestedObject("device_info");
    device_info["ip"] = WiFi.localIP().toString();
    device_info["mac"] = WiFi.macAddress();
    device_info["rssi"] = WiFi.RSSI();
    device_info["wifi_ssid"] = WIFI_SSID;
    
    // Add sensor data array
    JsonArray data = doc.createNestedArray("data");
    for (int i = 0; i < count; i++) {
        JsonObject s = data.createNestedObject();
        s["temperature"] = samples[i].temperature;
        s["humidity"] = samples[i].humidity;
        s["gas_value"] = samples[i].gas_value;
        s["dust_value"] = samples[i].dust_value;
        s["fire_sensor_status"] = samples[i].fire_sensor_status;
        s["ai_prediction"] = samples[i].label;
        s["timestamp"] = samples[i].timestamp;
    }
    
    String jsonString;
    serializeJson(doc, jsonString);
    Serial.println("Sending JSON data:");
    Serial.println(jsonString);
    
    int httpResponseCode = http.POST(jsonString);
    
    if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.println("HTTP Response code: " + String(httpResponseCode));
        Serial.println("Response: " + response);
        Serial.println("Data sent successfully");
    } else {
        Serial.print("Error sending data. HTTP Response code: ");
        Serial.println(httpResponseCode);
        Serial.print("Error message: ");
        Serial.println(http.errorToString(httpResponseCode));
    }
    
    http.end();
    Serial.println("===========================\n");
} 