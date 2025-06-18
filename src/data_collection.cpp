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
    
    // Check if enough time has passed since last collection
    if (currentTime - lastCollectionTime < MIN_COLLECTION_INTERVAL) {
        return;
    }
    
    // Add new sample
    collectedSamples[currentSampleIndex] = {
        temp,
        hum,
        gas,
        dust,
        flame,
        predicted_label,
        currentTime
    };
    
    currentSampleIndex = (currentSampleIndex + 1) % MAX_STORED_SAMPLES;
    if (sampleCount < MAX_STORED_SAMPLES) {
        sampleCount++;
    }
    
    lastCollectionTime = currentTime;
    
    // If we have enough samples, try to send them
    if (sampleCount >= MAX_STORED_SAMPLES) {
        sendTrainingData();
    }
}

void sendTrainingData() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi not connected. Cannot send data.");
        return;
    }
    
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    
    // Create JSON document
    StaticJsonDocument<4096> doc;
    JsonArray data = doc.createNestedArray("data");
    
    // Add all samples to JSON
    for (int i = 0; i < sampleCount; i++) {
        JsonObject sample = data.createNestedObject();
        sample["temperature"] = collectedSamples[i].temperature;
        sample["humidity"] = collectedSamples[i].humidity;
        sample["gas_value"] = collectedSamples[i].gas_value;
        sample["dust_value"] = collectedSamples[i].dust_value;
        sample["fire_sensor_status"] = collectedSamples[i].fire_sensor_status;
        sample["label"] = collectedSamples[i].label;
        sample["timestamp"] = collectedSamples[i].timestamp;
    }
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    int httpResponseCode = http.POST(jsonString);
    
    if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.println("HTTP Response code: " + String(httpResponseCode));
        Serial.println("Response: " + response);
        
        // If data was successfully sent, clear the buffer
        sampleCount = 0;
        currentSampleIndex = 0;
    } else {
        Serial.println("Error sending data: " + String(httpResponseCode));
    }
    
    http.end();
} 