// activity_wearable — MCU side: read an MPU-6050-style IMU at 100 Hz and
// stream every sample to Python (Linux side) over the Bridge.
//
// What this sketch does:
//   1. Scans two of the UNO Q's three I2C buses for devices:
//        Wire2 = A4 (SDA) / A5 (SCL)  <- the IMU is wired here
//        Wire1 = Qwiic connector (3.3 V)
//      The third bus, Wire (SDA/SCL header pins next to AREF), is NOT
//      scanned: those pins are damaged on this board, and probing a damaged
//      bus can hang or report junk devices.
//      On the UNO Q these are SEPARATE buses (on a classic Uno R3, A4/A5 and
//      SDA/SCL were the same wires; here they are not).
//   2. Uses the first bus that has a device at 0x68 or 0x69 (MPU-6050
//      address with AD0 low / high). Reads WHO_AM_I, resets + wakes the
//      sensor, sets accel ±16 g and gyro ±2000 °/s, and reads the config
//      back so Python can verify it.
//   3. Every 10 ms (drift-free schedule, 100 Hz) burst-reads the 14 data
//      bytes and sends
//        Bridge.notify("imu", seq, t_us, ax, ay, az, gx, gy, gz)
//      seq  = tick counter (+1 every 10 ms, even if a read fails -> Python
//             sees a gap = dropped sample)
//      t_us = MCU micros() when the sample was taken (wraps every ~71 min;
//             Python unwraps it)
//      ax..gz = RAW int16 counts. Python converts to g and °/s (so the
//             conversion is unit-tested in pytest instead of buried here).
//   4. Python can ask for the scan/config result any time with
//        Bridge.call("imu_info")  -> "ok=1;bus=Wire2;addr=0x68;who=0x68;..."
//   5. Python sets the status LED (LED4) with Bridge.notify("set_status", n)
//      and the LED matrix with Bridge.call("draw", <104 bytes>).
//   6. With DEBUG_PRINT = true, prints a human-readable line (in g and °/s)
//      once per second to the serial monitor:  arduino-app-cli monitor
//      (off by default: it delays samples; "IMU NOT READY" always prints)
//
// If no IMU is found it rescans every 2 s, so you can fix wiring live.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>
#include <vector>

Arduino_LED_Matrix matrix;

// ---- MPU-6050 registers (InvenSense RM-MPU-6000A register map) ----------
static const uint8_t REG_SMPLRT_DIV   = 0x19;
static const uint8_t REG_CONFIG       = 0x1A;  // DLPF setting
static const uint8_t REG_GYRO_CONFIG  = 0x1B;
static const uint8_t REG_ACCEL_CONFIG = 0x1C;
static const uint8_t REG_ACCEL_XOUT_H = 0x3B;  // 14 bytes: accel xyz, temp, gyro xyz
static const uint8_t REG_PWR_MGMT_1   = 0x6B;
static const uint8_t REG_WHO_AM_I     = 0x75;

static const uint8_t GYRO_FS_2000DPS = 0x18;   // FS_SEL = 3
static const uint8_t ACCEL_FS_16G    = 0x18;   // AFS_SEL = 3
// DLPF_CFG = 3: accel 44 Hz / gyro 42 Hz bandwidth. Below the 50 Hz Nyquist
// limit of 100 Hz sampling, so fast impacts don't alias.
static const uint8_t DLPF_44HZ       = 0x03;

static const uint32_t PERIOD_US = 10000;       // 100 Hz
static const uint32_t RETRY_MS  = 2000;
// Once-per-second status line in `arduino-app-cli monitor`. Off by default:
// even one Serial.println blocks ~15-20 ms (it goes over the Bridge), which
// delays the next sample. Python's log already shows the same values. Set to
// true only to debug wiring.
static const bool DEBUG_PRINT = false;

// ---- The I2C buses we scan --------------------------------------------------
struct Bus {
  arduino::ZephyrI2C *wire;
  const char *name;
};
static Bus buses[] = {
  {&Wire2, "Wire2"},   // A4/A5 -- IMU wired here
  {&Wire1, "Wire1"},   // Qwiic
  // {&Wire, "Wire"},  // SDA/SCL header -- pins damaged, don't scan
};
static const int N_BUSES = sizeof(buses) / sizeof(buses[0]);
static const int MAX_FOUND = 8;

// Scan + config results. Plain ints written by loop(), read by the Bridge
// thread in imu_info(); each is a single 32-bit read so it can't tear.
static volatile int found_n[N_BUSES];
static volatile uint8_t found_addr[N_BUSES][MAX_FOUND];
static volatile int imu_bus = -1;       // index into buses[], -1 = none
static volatile int imu_addr = 0;
static volatile int who_am_i = -1;
static volatile int accel_cfg = -1;     // read back after configuring
static volatile int gyro_cfg = -1;
static volatile bool imu_ok = false;

static arduino::ZephyrI2C *imu = nullptr;

// ---- Streaming state -------------------------------------------------------
static uint32_t next_us = 0;
static uint32_t seq = 0;
static uint32_t last_retry_ms = 0;
static uint32_t next_print_ms = 0;
static uint32_t n_sent = 0, n_i2c_err = 0, n_late = 0;  // per-second debug counters
static int16_t last_raw[6];

// ---- Low-level I2C helpers -------------------------------------------------
static bool write_reg(uint8_t reg, uint8_t val) {
  imu->beginTransmission(imu_addr);
  imu->write(reg);
  imu->write(val);
  return imu->endTransmission() == 0;
}

static bool read_regs(uint8_t reg, uint8_t *buf, size_t n) {
  imu->beginTransmission(imu_addr);
  imu->write(reg);
  if (imu->endTransmission(false) != 0) return false;
  if (imu->requestFrom((uint8_t)imu_addr, n) != n) return false;
  for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)imu->read();
  return true;
}

static int read_reg(uint8_t reg) {
  uint8_t v;
  return read_regs(reg, &v, 1) ? v : -1;
}

// Probe with a 1-byte read rather than an empty write: on this Zephyr core a
// zero-length write isn't guaranteed to reach the bus.
static bool probe(arduino::ZephyrI2C *w, uint8_t addr) {
  if (w->requestFrom(addr, (size_t)1) != 1) return false;
  w->read();  // discard the byte
  return true;
}

// ---- Setup / scan ----------------------------------------------------------
static void scan_all_buses() {
  imu_bus = -1;
  for (int b = 0; b < N_BUSES; b++) {
    buses[b].wire->begin();
    int n = 0;
    for (uint8_t a = 0x08; a <= 0x77; a++) {
      if (probe(buses[b].wire, a)) {
        if (n < MAX_FOUND) found_addr[b][n] = a;
        n++;
        if (imu_bus < 0 && (a == 0x68 || a == 0x69)) {
          imu_bus = b;
          imu_addr = a;
        }
      }
    }
    found_n[b] = n;
  }
}

static bool setup_imu() {
  imu_ok = false;
  who_am_i = accel_cfg = gyro_cfg = -1;
  scan_all_buses();
  if (imu_bus < 0) return false;
  imu = buses[imu_bus].wire;

  who_am_i = read_reg(REG_WHO_AM_I);
  if (who_am_i < 0) return false;

  write_reg(REG_PWR_MGMT_1, 0x80);              // device reset
  delay(100);
  if (!write_reg(REG_PWR_MGMT_1, 0x01)) return false;  // wake, clock = gyro X PLL
  delay(10);
  write_reg(REG_CONFIG, DLPF_44HZ);
  write_reg(REG_SMPLRT_DIV, 0);                 // 1 kHz internal rate; we read the newest at 100 Hz
  write_reg(REG_GYRO_CONFIG, GYRO_FS_2000DPS);
  write_reg(REG_ACCEL_CONFIG, ACCEL_FS_16G);
  delay(50);                                    // let gyro start up + filter settle

  gyro_cfg = read_reg(REG_GYRO_CONFIG);
  accel_cfg = read_reg(REG_ACCEL_CONFIG);
  imu_ok = (gyro_cfg & 0x18) == GYRO_FS_2000DPS && (accel_cfg & 0x18) == ACCEL_FS_16G;
  return imu_ok;
}

// ---- Bridge: Python asks "what did you find?" ------------------------------
// Format: ok=1;bus=Wire2;addr=0x68;who=0x68;accel_cfg=0x18;gyro_cfg=0x18;
//         scan=Wire2:0x68|Wire1:
static String hex2(int v) {
  if (v < 0) return String("none");
  String s = String(v, HEX);
  if (s.length() < 2) s = "0" + s;
  return "0x" + s;
}

String imu_info() {
  String s = "ok=";
  s += imu_ok ? "1" : "0";
  s += ";bus=";
  s += imu_bus >= 0 ? buses[imu_bus].name : "none";
  s += ";addr=" + hex2(imu_bus >= 0 ? imu_addr : -1);
  s += ";who=" + hex2(who_am_i);
  s += ";accel_cfg=" + hex2(accel_cfg);
  s += ";gyro_cfg=" + hex2(gyro_cfg);
  s += ";scan=";
  for (int b = 0; b < N_BUSES; b++) {
    if (b) s += "|";
    s += buses[b].name;
    s += ":";
    int n = found_n[b] < MAX_FOUND ? found_n[b] : MAX_FOUND;
    for (int i = 0; i < n; i++) {
      if (i) s += ",";
      s += hex2(found_addr[b][i]);
    }
  }
  return s;
}

// ---- Bridge: Python sets the status LED ------------------------------------
// LED4 (RGB, on/off per colour, active-low: LOW = on). Shows the recording
// state when the board is on a power bank with no screen:
//   0 off (idle)  1 blue (countdown)  2 red (recording)  3 green (done)  4 purple (error)
// Must match LED_* in python/main.py.
void set_status(int state) {
  bool r = state == 2 || state == 4;
  bool g = state == 3;
  bool b = state == 1 || state == 4;
  digitalWrite(LED4_R, r ? LOW : HIGH);
  digitalWrite(LED4_G, g ? LOW : HIGH);
  digitalWrite(LED4_B, b ? LOW : HIGH);
}

// ---- Bridge: Python draws on the LED matrix --------------------------------
// One frame = 104 brightness values 0..7, row by row (8 rows x 13 columns).
// The pictures (countdown digits, progress bar, ...) are built in Python
// (wearable/display.py); this just shows them. Only called from the Bridge
// thread, and loop() never touches the matrix, so no locking is needed.
void draw(std::vector<uint8_t> frame) {
  if (frame.size() != 104) return;
  matrix.draw(frame.data());
}

// ---- Sampling --------------------------------------------------------------
static void take_sample(uint32_t t_us) {
  uint8_t buf[14];
  if (!read_regs(REG_ACCEL_XOUT_H, buf, sizeof(buf))) {
    n_i2c_err++;   // no notify -> Python sees a seq gap
    return;
  }
  // Big-endian int16s: ax ay az | temp | gx gy gz
  int16_t ax = (int16_t)((buf[0] << 8) | buf[1]);
  int16_t ay = (int16_t)((buf[2] << 8) | buf[3]);
  int16_t az = (int16_t)((buf[4] << 8) | buf[5]);
  int16_t gx = (int16_t)((buf[8] << 8) | buf[9]);
  int16_t gy = (int16_t)((buf[10] << 8) | buf[11]);
  int16_t gz = (int16_t)((buf[12] << 8) | buf[13]);
  Bridge.notify("imu", seq, t_us, ax, ay, az, gx, gy, gz);
  n_sent++;
  last_raw[0] = ax; last_raw[1] = ay; last_raw[2] = az;
  last_raw[3] = gx; last_raw[4] = gy; last_raw[5] = gz;
}

// On the UNO Q, Serial goes to the monitor over the Bridge, and every
// Serial.print() call is a separate round-trip. Build the whole line first and
// send it with ONE println, or the loop stalls (~200 ms) and misses samples.
static void print_status() {
  if (!imu_ok) {
    Serial.println("IMU NOT READY: " + imu_info());
    return;
  }
  // Human-readable values for the monitor only (Python does its own conversion).
  String s = "a[g]=";
  for (int i = 0; i < 3; i++) { s += ' '; s += String(last_raw[i] / 2048.0f, 3); }
  s += "  g[dps]=";
  for (int i = 3; i < 6; i++) { s += ' '; s += String(last_raw[i] / 16.4f, 1); }
  s += "  sent/s=";     s += n_sent;
  s += " i2c_err/s=";   s += n_i2c_err;
  s += " late/s=";      s += n_late;
  Serial.println(s);
  n_sent = n_i2c_err = n_late = 0;
}

void setup() {
  pinMode(LED4_R, OUTPUT);
  pinMode(LED4_G, OUTPUT);
  pinMode(LED4_B, OUTPUT);
  set_status(0);
  matrix.begin();
  matrix.setGrayscaleBits(3);   // frames use brightness 0..7
  matrix.clear();
  Bridge.begin();
  Bridge.provide("imu_info", imu_info);
  Bridge.provide("set_status", set_status);
  Bridge.provide("draw", draw);
  setup_imu();
  next_us = micros();
  next_print_ms = millis() + 1000;
}

void loop() {
  uint32_t now_ms = millis();

  if (!imu_ok) {
    if (now_ms - last_retry_ms >= RETRY_MS) {
      last_retry_ms = now_ms;
      if (setup_imu()) next_us = micros();
    }
  } else {
    uint32_t now = micros();
    if ((int32_t)(now - next_us) >= 0) {
      next_us += PERIOD_US;
      // Fell more than 5 periods behind (shouldn't happen): resync instead of
      // bursting. Advance seq past the skipped ticks so Python counts them as
      // dropped samples (seq = one tick per 10 ms, always).
      if ((int32_t)(now - next_us) > (int32_t)(5 * PERIOD_US)) {
        seq += (now - next_us) / PERIOD_US + 1;
        next_us = now + PERIOD_US;
        n_late++;
      }
      take_sample(now);
      seq++;
    }
  }

  // "IMU NOT READY" is always printed: nothing is being sampled then anyway.
  if ((int32_t)(now_ms - next_print_ms) >= 0) {
    next_print_ms += 1000;
    if (DEBUG_PRINT || !imu_ok) print_status();
  }
}
