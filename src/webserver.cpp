#include "webserver.h"
#include "sensors.h"
#include "model.h"
#include "config.h"
#include <ArduinoJson.h>
#include <SPIFFS.h>

extern float temperature;
extern float humidity;
extern float gas;
extern float dust;
extern int flame;
extern int fireStatus;

AsyncWebServer server(80);

const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE HTML>
<html>
<head>
  <title>Fire Detection System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 20px;
      background-color: #f0f0f0;
    }
    .container {
      max-width: 800px;
      margin: 0 auto;
      background-color: white;
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .sensor-value {
      margin: 10px 0;
      padding: 10px;
      background-color: #f8f9fa;
      border-radius: 5px;
    }
    .status {
      margin: 20px 0;
      padding: 15px;
      border-radius: 5px;
      text-align: center;
      font-weight: bold;
    }
    .safe {
      background-color: #d4edda;
      color: #155724;
    }
    .warning {
      background-color: #fff3cd;
      color: #856404;
    }
    .danger {
      background-color: #f8d7da;
      color: #721c24;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Fire Detection System</h1>
    <div class="sensor-value">
      <h2>Sensor Readings</h2>
      <p>Temperature: <span id="temperature">--</span>°C</p>
      <p>Humidity: <span id="humidity">--</span>%</p>
      <p>Gas Level: <span id="gas">--</span> ppm</p>
      <p>Dust Level: <span id="dust">--</span> μg/m³</p>
      <p>Flame Detected: <span id="flame">--</span></p>
    </div>
    <div id="status" class="status">
      <h2>Fire Status</h2>
      <p id="status-text">Checking...</p>
    </div>
  </div>
  <script>
    function updateValues() {
      fetch('/sensors')
        .then(response => response.json())
        .then(data => {
          document.getElementById('temperature').textContent = data.temperature.toFixed(1);
          document.getElementById('humidity').textContent = data.humidity.toFixed(1);
          document.getElementById('gas').textContent = data.gas.toFixed(1);
          document.getElementById('dust').textContent = data.dust.toFixed(1);
          document.getElementById('flame').textContent = data.flame === 1 ? 'No' : 'Yes';  // 1 means no fire, 0 means fire detected
          
          const statusDiv = document.getElementById('status');
          const statusText = document.getElementById('status-text');
          
          if (data.fireStatus === 1) {
            statusDiv.className = 'status danger';
            statusText.textContent = 'FIRE DETECTED!';
          } else if (data.fireStatus === 0) {
            statusDiv.className = 'status safe';
            statusText.textContent = 'No Fire Detected';
          } else {
            statusDiv.className = 'status warning';
            statusText.textContent = 'System Error';
          }
        });
    }
    
    setInterval(updateValues, 1000);
    updateValues();
  </script>
</body>
</html>
)rawliteral";

void initWebServer() {
    // Initialize SPIFFS
    if(!SPIFFS.begin(true)) {
        Serial.println("An error occurred while mounting SPIFFS");
        return;
    }

    // Route for root / web page
    server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
        request->send(SPIFFS, "/index.html", "text/html");
    });

    // Route for style.css
    server.on("/style.css", HTTP_GET, [](AsyncWebServerRequest *request){
        request->send(SPIFFS, "/style.css", "text/css");
    });

    // Route for script.js
    server.on("/script.js", HTTP_GET, [](AsyncWebServerRequest *request){
        request->send(SPIFFS, "/script.js", "text/javascript");
    });

    // Route for sensor data
    server.on("/sensor-data", HTTP_GET, [](AsyncWebServerRequest *request){
        StaticJsonDocument<200> doc;
        doc["temperature"] = current_temperature;
        doc["humidity"] = current_humidity;
        doc["gas"] = current_mq2_ppm;
        doc["dust"] = current_mp02_value;
        doc["flame"] = current_flame_value;
        doc["fireStatus"] = (current_flame_value == 0) ? 1 : 0;  // 1 if flame detected (value = 0), 0 if not (value = 1)

        String response;
        serializeJson(doc, response);
        request->send(200, "application/json", response);
    });

    // Start server
    server.begin();
    Serial.println("HTTP server started");
}

void handleWebServer() {
  // Nothing to do here as AsyncWebServer handles requests asynchronously
} 