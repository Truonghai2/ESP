#ifndef WEBSERVER_H
#define WEBSERVER_H

#define HTTP_GET 0
#define HTTP_POST 1
#define HTTP_DELETE 2
#define HTTP_PUT 3
#define HTTP_PATCH 4
#define HTTP_HEAD 5
#define HTTP_OPTIONS 6
#define HTTP_ANY 255

#include <Arduino.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include "config.h"

// WebServer instance
extern AsyncWebServer server;

// Function declarations
void webServerTask(void *pvParameters);
void handleRoot(AsyncWebServerRequest *request);
void handleData(AsyncWebServerRequest *request);

// WebServer initialization
void initWebServer();

#endif // WEBSERVER_H 