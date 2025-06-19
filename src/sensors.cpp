#include "sensors.h"
#include "model.h"
#include "data_collection.h"

// Global sensor readings
volatile float current_humidity = 0.0;
volatile float current_temperature = 0.0;
volatile float current_mq2_ppm = 0.0;
volatile int current_flame_value = 0;
volatile int current_mp02_value = 0;

// Moving average for DHT
float humidity_readings[DHT_SAMPLE_COUNT];
float temp_readings[DHT_SAMPLE_COUNT];
int dht_index = 0;

// Moving average for MQ2
float mq2_readings[5];  // Store last 5 readings
int mq2_index = 0;

// Debounce for flame sensor
static const int FLAME_DEBOUNCE_TIME = 50;  // ms
static unsigned long last_flame_change = 0;
static int last_flame_state = 0;

// Sensor objects
DHT dht(PINS::DHT_DATA, DHT_TYPE);

// Calibration values for MQ2
const float MQ2_RL = 10.0;  // Load resistance in kΩ
const float MQ2_R0 = 10.0;  // Sensor resistance in clean air

// Helper functions
static bool isValidDHTReading(float value) {
    return !isnan(value) && value > -40.0 && value < 125.0;  // DHT11 range
}

static float calculateMovingAverage(float* readings, int size) {
    float sum = 0;
    int valid_readings = 0;
    
    for(int i = 0; i < size; i++) {
        if(isValidDHTReading(readings[i])) {
            sum += readings[i];
            valid_readings++;
        }
    }
    
    return valid_readings > 0 ? sum / valid_readings : 0;
}

// Sensor reading functions
void readDHT11() {
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    
    if (isValidDHTReading(h) && isValidDHTReading(t)) {
        humidity_readings[dht_index] = h;
        temp_readings[dht_index] = t;
        
        dht_index = (dht_index + 1) % DHT_SAMPLE_COUNT;
        
        float new_humidity = calculateMovingAverage(humidity_readings, DHT_SAMPLE_COUNT);
        float new_temperature = calculateMovingAverage(temp_readings, DHT_SAMPLE_COUNT);
        
        // Only update if values are significantly different
        if (abs(new_humidity - current_humidity) > 1.0 || 
            abs(new_temperature - current_temperature) > 0.5) {
            current_humidity = new_humidity;
            current_temperature = new_temperature;
            
            Serial.print("DHT - Humidity: ");
            Serial.print(current_humidity);
            Serial.print(" %\t");
            Serial.print("Temperature: ");
            Serial.print(current_temperature);
            Serial.println(" *C");
        }
    } else {
        Serial.println("Failed to read from DHT11 sensor!");
    }
}

void readMQ2() {
    int mq2_raw = analogRead(PINS::MQ2_ANALOG);
    float voltage = mq2_raw * (3.3 / 4095.0);
    float mq2_ppm = voltage * 1000.0;  // Convert to ppm
    
    if (mq2_ppm >= 0 && mq2_ppm <= 1000) {
        mq2_readings[mq2_index] = mq2_ppm;
        mq2_index = (mq2_index + 1) % 5;
        
        float avg_ppm = 0;
        for(int i = 0; i < 5; i++) {
            avg_ppm += mq2_readings[i];
        }
        avg_ppm /= 5;
        
        if (abs(avg_ppm - current_mq2_ppm) > 1.0) {
            current_mq2_ppm = avg_ppm;
            Serial.print("MQ2 - Gas PPM: ");
            Serial.println(current_mq2_ppm);
        }
    } else {
        Serial.println("MQ2: Invalid reading!");
    }
}

void readMP2() {
    digitalWrite(PINS::MP2_LED_POWER, LOW);
    delayMicroseconds(280);
    float mp02_raw_analog = analogRead(PINS::MP2_ANALOG);
    delayMicroseconds(40);
    digitalWrite(PINS::MP2_LED_POWER, HIGH);

    float voltage = mp02_raw_analog * (3.3 / 4095.0);
    float dust_density_ug_m3 = 0.0;
    
    if (voltage >= 0.6) {
        dust_density_ug_m3 = (voltage - 0.6) * 1000.0;
    }

    if (dust_density_ug_m3 >= 0 && dust_density_ug_m3 <= 300) {
        int new_value = (int)dust_density_ug_m3;
        if (new_value != current_mp02_value) {
            current_mp02_value = new_value;
            Serial.print("MP02 - Dust Density: ");
            Serial.print(current_mp02_value);
            Serial.println(" ug/m^3");
        }
    } else {
        Serial.println("MP02: Invalid reading!");
    }
}

void readFlame() {
    int raw_flame_value = digitalRead(PINS::FIRE_SENSOR);
    unsigned long current_time = millis();
    
    // Invert the value: 1 means no fire, 0 means fire detected
    int flame_value = !raw_flame_value;  // Invert the value
    
    if (flame_value != last_flame_state) {
        if (current_time - last_flame_change > FLAME_DEBOUNCE_TIME) {
            current_flame_value = flame_value;
            last_flame_state = flame_value;
            last_flame_change = current_time;
            
            Serial.print("Flame - Value: ");
            Serial.print(current_flame_value);
            Serial.println(current_flame_value == 1 ? " (No Fire)" : " (Fire Detected)");
        }
    }
}

// New function to read all sensors at once
void readAllSensors() {
    // Read all sensors
    readDHT11();
    readMQ2();
    readMP2();
    readFlame();
    
    // Analyze with AI
    int fire_status = predictFireStatus(
        current_temperature,
        current_humidity,
        current_mq2_ppm,
        (float)current_mp02_value,
        current_flame_value
    );
    
    // Print analysis results
    Serial.println("\n=== Sensor Analysis ===");
    Serial.print("Temperature: "); Serial.print(current_temperature); Serial.println(" *C");
    Serial.print("Humidity: "); Serial.print(current_humidity); Serial.println(" %");
    Serial.print("Gas PPM: "); Serial.print(current_mq2_ppm); Serial.println(" ppm");
    Serial.print("Dust: "); Serial.print(current_mp02_value); Serial.println(" ug/m^3");
    Serial.print("Flame: "); Serial.println(current_flame_value);
    Serial.print("Fire Status: "); 
    switch (fire_status) {
        case 0:
            Serial.println("NO FIRE");
            break;
        case 1:
            Serial.println("COOKING FIRE");
            break;
        case 2:
            Serial.println("DANGEROUS FIRE");
            break;
        default:
            Serial.println("ERROR");
    }
    Serial.println("=====================\n");

    // Send sensor data and AI prediction to server
    addTrainingSample(
        current_temperature,
        current_humidity,
        current_mq2_ppm,
        (float)current_mp02_value,
        current_flame_value,
        fire_status
    );
}

// Task functions
void sensorTask(void *pvParameters) {
    for (;;) {
        readAllSensors();
        vTaskDelay(pdMS_TO_TICKS(2000));  // Read every 2 seconds
    }
}

void dhtTask(void *pvParameters) {
    for (;;) {
        readDHT11();
        vTaskDelay(pdMS_TO_TICKS(DHT_SAMPLE_INTERVAL));
    }
}

void mq2Task(void *pvParameters) {
    for (;;) {
        readMQ2();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void mp02Task(void *pvParameters) {
    for (;;) {
        readMP2();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void flameSensorTask(void *pvParameters) {
    for (;;) {
        readFlame();
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// Sensor initialization
void initSensors() {
    Serial.println("\n=== Initializing Sensors ===");
    
    // Initialize DHT
    Serial.println("\n1. Initializing DHT sensor...");
    Serial.print("DHT Pin: ");
    Serial.println(PINS::DHT_DATA);
    Serial.print("DHT Type: ");
    Serial.println(DHT_TYPE == DHT11 ? "DHT11" : "DHT22");
    
    dht.begin();
    delay(2000);  // Wait for DHT to stabilize
    
    // Initialize moving average arrays
    for(int i = 0; i < DHT_SAMPLE_COUNT; i++) {
        humidity_readings[i] = 0;
        temp_readings[i] = 0;
    }
    
    // Initialize MQ2 moving average
    for(int i = 0; i < 5; i++) {
        mq2_readings[i] = 0;
    }
    
    // Test DHT reading multiple times
    float h = 0, t = 0;
    int valid_readings = 0;
    
    for(int i = 0; i < 5; i++) {
        h = dht.readHumidity();
        t = dht.readTemperature();
        
        if(isValidDHTReading(h) && isValidDHTReading(t)) {
            valid_readings++;
            humidity_readings[0] = h;
            temp_readings[0] = t;
            break;
        }
        delay(1000);
    }
    
    Serial.print("Initial DHT reading - Humidity: ");
    Serial.print(h);
    Serial.print("%, Temperature: ");
    Serial.print(t);
    Serial.println("°C");
    
    if (valid_readings > 0) {
        Serial.println("DHT sensor initialized successfully");
        current_humidity = h;
        current_temperature = t;
    } else {
        Serial.println("Warning: DHT sensor may not be working properly");
        Serial.println("Please check:");
        Serial.println("1. DHT sensor is properly connected");
        Serial.println("2. Correct pin is used");
        Serial.println("3. Power supply is stable");
    }
    
    // Initialize MQ2
    Serial.println("\n2. Initializing MQ2 sensor...");
    Serial.print("MQ2 Analog Pin: ");
    Serial.println(PINS::MQ2_ANALOG);
    Serial.print("MQ2 Digital Pin: ");
    Serial.println(PINS::MQ2_DIGITAL);
    
    pinMode(PINS::MQ2_ANALOG, INPUT);
    pinMode(PINS::MQ2_DIGITAL, INPUT);
    
    // Test MQ2 reading multiple times
    int mq2_raw = 0;
    valid_readings = 0;
    
    for(int i = 0; i < 5; i++) {
        mq2_raw = analogRead(PINS::MQ2_ANALOG);
        if(mq2_raw > 0) {
            valid_readings++;
            float voltage = mq2_raw * (3.3 / 4095.0);
            current_mq2_ppm = voltage * 1000.0;  // Convert to ppm
            mq2_readings[0] = current_mq2_ppm;
            break;
        }
        delay(1000);
    }
    
    Serial.print("Initial MQ2 reading: ");
    Serial.print(current_mq2_ppm);
    Serial.println(" ppm");
    
    if (valid_readings > 0) {
        Serial.println("MQ2 sensor initialized successfully");
    } else {
        Serial.println("Warning: MQ2 sensor may not be working properly");
        Serial.println("Please check:");
        Serial.println("1. MQ2 sensor is properly connected");
        Serial.println("2. Correct pins are used");
        Serial.println("3. Power supply is stable");
    }
    
    // Initialize MP02
    Serial.println("\n3. Initializing MP02 sensor...");
    Serial.print("MP02 Analog Pin: ");
    Serial.println(PINS::MP2_ANALOG);
    Serial.print("MP02 LED Power Pin: ");
    Serial.println(PINS::MP2_LED_POWER);
    
    pinMode(PINS::MP2_LED_POWER, OUTPUT);
    digitalWrite(PINS::MP2_LED_POWER, HIGH);
    delay(100);  // Wait for LED to stabilize
    
    // Test MP02 reading multiple times
    int mp02_raw = 0;
    valid_readings = 0;
    
    for(int i = 0; i < 5; i++) {
        digitalWrite(PINS::MP2_LED_POWER, LOW);
        delayMicroseconds(280);
        mp02_raw = analogRead(PINS::MP2_ANALOG);
        delayMicroseconds(40);
        digitalWrite(PINS::MP2_LED_POWER, HIGH);
        
        if(mp02_raw > 0) {
            valid_readings++;
            float voltage = mp02_raw * (3.3 / 4095.0);
            if(voltage >= 0.6) {
                current_mp02_value = (int)((voltage - 0.6) * 1000.0);
            }
            break;
        }
        delay(1000);
    }
    
    Serial.print("Initial MP02 raw reading: ");
    Serial.println(mp02_raw);
    
    if (valid_readings > 0) {
        Serial.println("MP02 sensor initialized successfully");
    } else {
        Serial.println("Warning: MP02 sensor may not be working properly");
        Serial.println("Please check:");
        Serial.println("1. MP02 sensor is properly connected");
        Serial.println("2. Correct pins are used");
        Serial.println("3. Power supply is stable");
    }
    
    // Initialize Flame sensor
    Serial.println("\n4. Initializing Flame sensor...");
    Serial.print("Flame Sensor Pin: ");
    Serial.println(PINS::FIRE_SENSOR);
    
    pinMode(PINS::FIRE_SENSOR, INPUT_PULLUP);
    
    // Test Flame sensor reading multiple times
    int flame_value = 0;
    valid_readings = 0;
    
    for(int i = 0; i < 5; i++) {
        flame_value = digitalRead(PINS::FIRE_SENSOR);
        if(flame_value == 0 || flame_value == 1) {
            valid_readings++;
            current_flame_value = flame_value;
            break;
        }
        delay(100);
    }
    
    Serial.print("Initial Flame sensor reading: ");
    Serial.println(flame_value);
    
    if (valid_readings > 0) {
        Serial.println("Flame sensor initialized successfully");
    } else {
        Serial.println("Warning: Flame sensor may not be working properly");
        Serial.println("Please check:");
        Serial.println("1. Flame sensor is properly connected");
        Serial.println("2. Correct pin is used");
        Serial.println("3. Power supply is stable");
    }
    
    Serial.println("\n=== Sensor Initialization Complete ===\n");
}