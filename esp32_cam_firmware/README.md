# ESP32-CAM Hardware Setup Guide for IcterusLens AI

This directory contains the ready-to-flash Arduino firmware for your ESP32-CAM + COB LED Ring Light medical eye scanner.

## 1. Hardware Assembly
1. **Mount ESP32-CAM onto ESP32-CAM-MB Shield**:
   - Align the pins of the ESP32-CAM with the female headers on the ESP32-CAM-MB shield.
   - Gently press down until fully seated. The camera sensor faces upward; the USB port faces outward.
2. **Mount COB LED Ring Light**:
   - Position the ring light so that the ESP32-CAM lens peeks through the center opening of the ring.
   - Connect the Ring Light positive wire (White) to `5V` and negative wire (Blue) to `GND`.
3. **Add Capacitor (Brownout Prevention)**:
   - Insert the electrolytic capacitor (100uF - 470uF) across the `5V` and `GND` rails on your breadboard.
   - Match polarity: longer leg to `5V (+)`, shorter leg with stripe to `GND (-)`.

## 2. Flashing the Firmware with Arduino IDE
1. Open **Arduino IDE**.
2. Go to `File -> Preferences` and ensure ESP32 board URL is added:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Select Board: `Tools -> Board -> ESP32 Arduino -> AI Thinker ESP32-CAM`.
4. Connect the ESP32-CAM-MB to your laptop via USB cable.
5. Select Port: `Tools -> Port -> COM[X]`.
6. Click **Upload**.

## 3. Connecting to the Web App
1. After upload, open **Serial Monitor** at **115200 baud**.
2. By default, the device broadcasts a Wi-Fi hotspot:
   - **SSID**: `IcterusLens_Scanner`
   - **Password**: `12345678`
3. Connect your laptop (or demonstration laptop) to this Wi-Fi.
4. Open the website, navigate to the **📡 ESP32-CAM Hardware** tab.
5. URL is preset to `http://192.168.4.1/capture`.
6. Click **📸 Capture Eye Image** — the eye photo will instantly appear and get analyzed by the AI model!
