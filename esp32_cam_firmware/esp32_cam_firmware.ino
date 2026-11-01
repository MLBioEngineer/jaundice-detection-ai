/*
 * =====================================================================
 * IcterusLens AI — ESP32-CAM Medical Eye Sclera Scanner Firmware
 * =====================================================================
 * Hardware:
 *   - AI-Thinker ESP32-CAM Module (OV2640 Sensor)
 *   - ESP32-CAM-MB USB Programming Shield
 *   - COB LED Ring Light for diffused shadowless eye illumination
 *   - 100uF - 470uF Capacitor across 5V and GND (Brownout Prevention)
 *
 * Board Settings in Arduino IDE:
 *   - Board: "AI Thinker ESP32-CAM"
 *   - CPU Frequency: 240MHz (WiFi/BT)
 *   - Flash Frequency: 80MHz
 *   - Flash Mode: QIO
 *   - Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"
 * =====================================================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// =====================================================================
// Wi-Fi Configuration
// By default, creates an Access Point (Hotspot).
// Connect your laptop to "IcterusLens_Scanner" with password "12345678".
// Then the camera is instantly available at http://192.168.4.1/capture
// =====================================================================
const char* AP_SSID = "IcterusLens_Scanner";
const char* AP_PASS = "12345678";

// (Optional) If you want ESP32 to connect to your home/mobile Wi-Fi instead:
// Set USE_STATION_MODE = true and provide your credentials:
const bool USE_STATION_MODE = false;
const char* STA_SSID = "YOUR_WIFI_SSID";
const char* STA_PASS = "YOUR_WIFI_PASSWORD";

// AI-Thinker ESP32-CAM Pin Definition
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_LED_GPIO     4   // Onboard flash LED (active HIGH)

WebServer server(80);

void handleRoot() {
  String html = "<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'><title>IcterusLens ESP32-CAM</title>";
  html += "<style>body{font-family:sans-serif;text-align:center;padding:20px;background:#0f172a;color:#f8fafc;}";
  html += "a{background:#3b82f6;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;display:inline-block;margin:15px;font-weight:bold;}";
  html += "</style></head><body>";
  html += "<h2>👁️ IcterusLens AI — ESP32-CAM Hardware</h2>";
  html += "<p>Device is ONLINE and ready for Sclera Screening.</p>";
  html += "<p><a href='/capture' target='_blank'>📸 Capture Single Frame (/capture)</a></p>";
  html += "<p>Enter this URL in your Streamlit Web App: <code>http://" + WiFi.softAPIP().toString() + "/capture</code></p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleCapture() {
  camera_fb_t * fb = NULL;
  
  // Discard older buffer to ensure fresh real-time capture
  fb = esp_camera_fb_get();
  if (fb) {
    esp_camera_fb_return(fb);
    fb = NULL;
  }
  
  // Capture fresh frame
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Camera capture failed");
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  Serial.printf("📸 Image captured: %u bytes\n", fb->len);
  
  // Send JPEG buffer to web client
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  
  esp_camera_fb_return(fb);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- IcterusLens AI ESP32-CAM Starting ---");

  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);

  // Configure Camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Frame resolution: SVGA (800x600) gives excellent eye sclera detail
  if(psramFound()){
    config.frame_size = FRAMESIZE_SVGA; // 800x600
    config.jpeg_quality = 10;           // 10-63 (lower = higher quality)
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;  // 640x480
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed with error 0x%x\n", err);
    return;
  }
  Serial.println("✅ Camera sensor initialized successfully.");

  // Sensor color adjustments for medical eye imaging
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 1);     // slight boost (-2 to 2)
    s->set_contrast(s, 1);       // crisp sclera-iris border (-2 to 2)
    s->set_saturation(s, 0);     // neutral color fidelity (-2 to 2)
    s->set_whitebal(s, 1);       // enable auto white balance
    s->set_awb_gain(s, 1);       // enable AWB gain
    s->set_wb_mode(s, 0);        // 0 = Auto
  }

  // Start Wi-Fi
  if (USE_STATION_MODE) {
    Serial.printf("Connecting to Wi-Fi SSID: %s\n", STA_SSID);
    WiFi.begin(STA_SSID, STA_PASS);
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    Serial.println("\n✅ Wi-Fi Connected!");
    Serial.print("🌐 Device IP Address: http://");
    Serial.println(WiFi.localIP());
  } else {
    Serial.printf("Creating Access Point: %s\n", AP_SSID);
    WiFi.softAP(AP_SSID, AP_PASS);
    Serial.println("✅ Access Point started successfully!");
    Serial.print("🌐 Endpoint URL: http://");
    Serial.print(WiFi.softAPIP());
    Serial.println("/capture");
  }

  // Web Server Routes
  server.on("/", HTTP_GET, handleRoot);
  server.on("/capture", HTTP_GET, handleCapture);
  server.begin();
  Serial.println("🚀 HTTP Server started. Ready to capture!");
}

void loop() {
  server.handleClient();
}
