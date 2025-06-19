#ifndef CONFIG_H
#define CONFIG_H

// Pin Definitions
enum PINS {
    LED_SECURITY = 14, 
    LED_LIGHT    = 2,
    FAN          = 4,
    BUZZER       = 19,  
    PUMP         = 17,  
    FIRE_SENSOR  = 32,
    DHT_DATA     = 4,
    SDA_PIN      = 32,
    SCL_PIN      = 13,
    MQ2_DIGITAL  = 25,
    MQ2_ANALOG   = 34,
    MP2_ANALOG   = 25,
    MP2_DIGITAL  = 26,
    MP2_LED_POWER = 23,
};

// WiFi Configuration
#define WIFI_SSID "TP-LINK_25AC"
#define WIFI_PASSWORD "07990717"
#define SERVER_URL "http://192.168.0.104:8000/api/devices/1/sensor-data"

// Telegram Configuration
#define TELEGRAM_NOTIFICATION_INTERVAL 300000  // 5 minutes
#define TELEGRAM_HEARTBEAT_INTERVAL 3600000    // 1 hour

// Sensor Configuration
#define DHT_TYPE DHT11
#define DHT_PIN PINS::DHT_DATA
#define DHT_SAMPLE_INTERVAL 2000  // 2 seconds
#define DHT_SAMPLE_COUNT 5        // Number of samples for moving average

// Data Collection Settings
#define MAX_STORED_SAMPLES 10
#define MIN_COLLECTION_INTERVAL 5000

// Task Stack Sizes
#define DHT_TASK_STACK_SIZE 2048
#define MQ2_TASK_STACK_SIZE 2048
#define FLAME_TASK_STACK_SIZE 2048
#define MP02_TASK_STACK_SIZE 2048
#define ALERT_TASK_STACK_SIZE 4096
#define WEBSERVER_TASK_STACK_SIZE 8192

// Task Priorities
#define DHT_TASK_PRIORITY 1
#define MQ2_TASK_PRIORITY 1
#define FLAME_TASK_PRIORITY 1
#define MP02_TASK_PRIORITY 1
#define ALERT_TASK_PRIORITY 2
#define WEBSERVER_TASK_PRIORITY 1

#endif // CONFIG_H 