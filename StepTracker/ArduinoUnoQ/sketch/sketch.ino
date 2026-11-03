// shin_imu — MCU side of the shin IMU streamer (Arduino UNO Q).
//
// Reads an MPU-6050 (driver reused from activity_wearable) at a fixed rate
// and streams batches of samples to the Linux side over the Router Bridge.
// No processing here beyond reading and timestamping.
//
// Timing:
//   * A Zephyr k_timer fires every period (10 ms at 100 Hz). Its ISR bumps
//     tick_count and gives a semaphore. Nothing is paced with delay().
//   * A high-priority sampler thread wakes on the semaphore, stamps the
//     sample with the 64-bit cycle counter (160 MHz -> ns, never wraps), and
//     burst-reads the 14 data bytes over I2C.
//   * seq = index of the timer tick the sample belongs to. If a tick is ever
//     missed (I2C error, thread late, queue full) seq skips, so the Linux side
//     sees a gap instead of a silent loss.
//   * Samples go through a k_msgq to loop() (lowest priority), which packs
//     BATCH samples into one binary blob and sends
//       Bridge.notify("imu_batch", <bytes>)
//     Sending (~250 B over the 115200 baud Bridge UART, ~22 ms) never delays
//     a read, because the sampler thread preempts loop().
//
// Batch format: BATCH records of 24 bytes, little-endian:
//   uint32 seq | uint64 t_mcu_ns | int16 ax ay az gx gy gz (raw counts)
//
// Bridge methods provided (called from Python):
//   now_ns()          -> uint64 MCU clock, for the MCU->Linux clock mapping
//   mcu_stats()       -> "fw=shin_imu/1;k=v;..." scan/config/counters
//   stream(bool)      -> start/stop the timer
//   set_rate(int)     -> bool; 25/50/100/200 Hz (DLPF follows the rate)
//   set_synthetic(bool) test pattern when no IMU is found (off at boot)
//   set_led(int)      LED4: 0 off, 1 green, 2 blue, 3 red

#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include <vector>

// Reported first in mcu_stats(); the Linux streamer checks it to know the
// right sketch is flashed. Bump when the batch format changes.
#define FW_ID "shin_imu/1"

// ---- MPU-6050 registers (InvenSense RM-MPU-6000A register map) ----------
static const uint8_t REG_SMPLRT_DIV   = 0x19;
static const uint8_t REG_CONFIG       = 0x1A;  // DLPF setting
static const uint8_t REG_GYRO_CONFIG  = 0x1B;
static const uint8_t REG_ACCEL_CONFIG = 0x1C;
static const uint8_t REG_ACCEL_XOUT_H = 0x3B;  // 14 bytes: accel xyz, temp, gyro xyz
static const uint8_t REG_PWR_MGMT_1   = 0x6B;
static const uint8_t REG_WHO_AM_I     = 0x75;

static const uint8_t GYRO_FS_2000DPS = 0x18;   // FS_SEL = 3 (largest range)
static const uint8_t ACCEL_FS_16G    = 0x18;   // AFS_SEL = 3 (largest range)

static const uint32_t RETRY_MS = 2000;
static const int MAX_I2C_FAILS = 50;           // consecutive failures -> rescan

// ---- Rates ----------------------------------------------------------------
// Period must be a whole number of kernel ticks (100 us at 10 kHz).
// DLPF bandwidth is kept below Nyquist (rate/2) so impacts don't alias.
struct RateCfg { int hz; uint8_t dlpf; };
static const RateCfg RATES[] = {
  {25, 0x05},    // 10 Hz bandwidth
  {50, 0x04},    // 21 Hz
  {100, 0x03},   // 44 Hz
  {200, 0x02},   // 94 Hz
};
static const int DEFAULT_RATE_HZ = 100;

// ---- Streaming ------------------------------------------------------------
static const int BATCH = 10;
static const int QUEUE_DEPTH = 256;            // 2.5 s at 100 Hz

struct __attribute__((packed)) Sample {
  uint32_t seq;
  uint64_t t_ns;
  int16_t raw[6];
};
static_assert(sizeof(Sample) == 24, "Sample must pack to 24 bytes");

static char queue_buf[QUEUE_DEPTH * sizeof(Sample)] __aligned(4);
static struct k_msgq sample_q;
static struct k_timer tick_timer;
static struct k_sem tick_sem;
static volatile uint32_t tick_count = 0;

// Requested state, written by Bridge handlers, applied by the sampler thread
// (it owns the I2C bus and the timer).
static volatile int req_rate_hz = DEFAULT_RATE_HZ;
static volatile bool req_streaming = true;
static volatile bool synthetic = false;

// Applied state + counters, read by mcu_stats() on the Bridge thread.
static volatile int rate_hz = 0;
static volatile bool streaming = false;
static volatile uint32_t n_late = 0, n_i2c_err = 0, n_overflow = 0, n_sent = 0;

// ---- I2C buses ------------------------------------------------------------
// On the UNO Q these are three SEPARATE buses (A4/A5 are not the SDA/SCL
// header). The first one with a device at 0x68/0x69 is used.
struct Bus {
  arduino::ZephyrI2C *wire;
  const char *name;
};
static Bus buses[] = {
  {&Wire, "Wire"},     // SDA/SCL header next to AREF -- IMU wired here
  {&Wire2, "Wire2"},   // A4/A5 (where activity_wearable had it)
  {&Wire1, "Wire1"},   // Qwiic
};
static const int N_BUSES = sizeof(buses) / sizeof(buses[0]);

static volatile int imu_bus = -1;
static volatile int imu_addr = 0;
static volatile int who_am_i = -1;
static volatile int accel_cfg = -1;
static volatile int gyro_cfg = -1;
static volatile int dlpf_cfg = -1;
static volatile bool imu_ok = false;
static arduino::ZephyrI2C *imu = nullptr;

// ---- Clock ----------------------------------------------------------------
// 64-bit cycle counter -> ns, split so the multiply can't overflow.
static uint64_t mcu_now_ns() {
  const uint64_t hz = CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC;
  uint64_t cyc = k_cycle_get_64();
  return (cyc / hz) * 1000000000ULL + (cyc % hz) * 1000000000ULL / hz;
}

// ---- Low-level I2C helpers (from activity_wearable) -----------------------
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
  w->read();
  return true;
}

static void find_imu() {
  imu_bus = -1;
  for (int b = 0; b < N_BUSES && imu_bus < 0; b++) {
    buses[b].wire->begin();
    for (uint8_t a : {0x68, 0x69}) {
      if (probe(buses[b].wire, a)) {
        imu_bus = b;
        imu_addr = a;
        break;
      }
    }
  }
}

static const RateCfg *rate_cfg(int hz) {
  for (const RateCfg &r : RATES)
    if (r.hz == hz) return &r;
  return nullptr;
}

static bool setup_imu() {
  imu_ok = false;
  who_am_i = accel_cfg = gyro_cfg = dlpf_cfg = -1;
  find_imu();
  if (imu_bus < 0) return false;
  imu = buses[imu_bus].wire;

  who_am_i = read_reg(REG_WHO_AM_I);
  if (who_am_i < 0) return false;

  write_reg(REG_PWR_MGMT_1, 0x80);                     // device reset
  k_msleep(100);
  if (!write_reg(REG_PWR_MGMT_1, 0x01)) return false;  // wake, clock = gyro X PLL
  k_msleep(10);
  write_reg(REG_SMPLRT_DIV, 0);                 // 1 kHz internal; we read the newest each tick
  write_reg(REG_GYRO_CONFIG, GYRO_FS_2000DPS);
  write_reg(REG_ACCEL_CONFIG, ACCEL_FS_16G);
  k_msleep(50);                                 // gyro start-up + filter settle

  gyro_cfg = read_reg(REG_GYRO_CONFIG);
  accel_cfg = read_reg(REG_ACCEL_CONFIG);
  imu_ok = (gyro_cfg & 0x18) == GYRO_FS_2000DPS && (accel_cfg & 0x18) == ACCEL_FS_16G;
  return imu_ok;
}

// ---- Timer ----------------------------------------------------------------
static void on_tick(struct k_timer *) {
  tick_count++;
  k_sem_give(&tick_sem);
}

// Apply the requested rate/streaming state. Sampler thread only.
static void apply_requested() {
  int want_rate = req_rate_hz;
  bool want_stream = req_streaming;
  if (want_rate == rate_hz && want_stream == streaming) return;

  k_timer_stop(&tick_timer);
  const RateCfg *r = rate_cfg(want_rate);
  if (imu_ok && write_reg(REG_CONFIG, r->dlpf)) dlpf_cfg = read_reg(REG_CONFIG);
  rate_hz = want_rate;
  streaming = want_stream;
  if (streaming) {
    k_sem_reset(&tick_sem);
    k_timeout_t period = K_USEC(1000000 / rate_hz);
    k_timer_start(&tick_timer, period, period);
  }
}

// ---- Sampler thread -------------------------------------------------------
static bool read_sample(Sample &s) {
  if (synthetic && !imu_ok) {
    // Test pattern: 1 g on z plus a slow sine on x / gyro z. Only enabled on
    // request (set_synthetic) and reported in mcu_stats as src=synthetic.
    float ph = 2.0f * 3.14159265f * (s.seq % 200) / 200.0f;
    s.raw[0] = (int16_t)(1000 * sinf(ph));
    s.raw[1] = 0;
    s.raw[2] = 2048;
    s.raw[3] = s.raw[4] = 0;
    s.raw[5] = (int16_t)(800 * cosf(ph));
    return true;
  }
  if (!imu_ok) return false;
  uint8_t buf[14];
  if (!read_regs(REG_ACCEL_XOUT_H, buf, sizeof(buf))) return false;
  // Big-endian int16s: ax ay az | temp | gx gy gz
  static const int OFF[6] = {0, 2, 4, 8, 10, 12};
  for (int i = 0; i < 6; i++) s.raw[i] = (int16_t)((buf[OFF[i]] << 8) | buf[OFF[i] + 1]);
  return true;
}

static void sampler_main(void *, void *, void *) {
  uint32_t last_seq = 0;
  bool have_last = false;
  int fails = 0;
  uint32_t next_retry_ms = 0;

  for (;;) {
    if (!imu_ok && (int32_t)(millis() - next_retry_ms) >= 0) {
      next_retry_ms = millis() + RETRY_MS;
      if (setup_imu()) rate_hz = 0;       // force DLPF re-apply below
    }
    apply_requested();
    if (!streaming) {
      k_msleep(20);
      have_last = false;
      continue;
    }
    if (k_sem_take(&tick_sem, K_MSEC(100)) != 0) continue;

    Sample s;
    s.t_ns = mcu_now_ns();
    s.seq = tick_count - 1;               // the tick this sample belongs to
    k_sem_reset(&tick_sem);               // if we were late, the gap shows in seq
    if (have_last && s.seq - last_seq > 1) n_late += s.seq - last_seq - 1;
    last_seq = s.seq;
    have_last = true;

    if (!read_sample(s)) {
      if (imu_ok) {
        n_i2c_err++;
        if (++fails >= MAX_I2C_FAILS) imu_ok = false;   // rescan
      }
      continue;
    }
    fails = 0;
    if (k_msgq_put(&sample_q, &s, K_NO_WAIT) != 0) n_overflow++;
  }
}

static const int SAMPLER_STACK = 2048;
static const int SAMPLER_PRIORITY = 2;   // above the Bridge thread (5) and loop() (14)
static struct k_thread sampler_thread;

// ---- Bridge handlers ------------------------------------------------------
uint64_t now_ns() { return mcu_now_ns(); }

static String hex2(int v) {
  if (v < 0) return String("none");
  String s = String(v, HEX);
  if (s.length() < 2) s = "0" + s;
  return "0x" + s;
}

String mcu_stats() {
  String s = "fw=" FW_ID ";ok=";
  s += imu_ok ? "1" : "0";
  s += ";src=";
  s += imu_ok ? "imu" : (synthetic ? "synthetic" : "none");
  s += ";bus=";
  s += imu_bus >= 0 ? buses[imu_bus].name : "none";
  s += ";addr=" + hex2(imu_bus >= 0 ? imu_addr : -1);
  s += ";who=" + hex2(who_am_i);
  s += ";accel_cfg=" + hex2(accel_cfg);
  s += ";gyro_cfg=" + hex2(gyro_cfg);
  s += ";dlpf=" + hex2(dlpf_cfg);
  s += ";rate=" + String(rate_hz);
  s += ";streaming=" + String(streaming ? 1 : 0);
  s += ";tick=" + String((uint32_t)tick_count);
  s += ";late=" + String((uint32_t)n_late);
  s += ";i2c_err=" + String((uint32_t)n_i2c_err);
  s += ";overflow=" + String((uint32_t)n_overflow);
  s += ";sent=" + String((uint32_t)n_sent);
  s += ";queued=" + String(k_msgq_num_used_get(&sample_q));
  return s;
}

void stream(bool on) { req_streaming = on; }

bool set_rate(int hz) {
  if (!rate_cfg(hz)) return false;
  req_rate_hz = hz;
  return true;
}

void set_synthetic(bool on) { synthetic = on; }

// LED4 (RGB, active-low: LOW = on).
void set_led(int state) {
  digitalWrite(LED4_R, state == 3 ? LOW : HIGH);
  digitalWrite(LED4_G, state == 1 ? LOW : HIGH);
  digitalWrite(LED4_B, state == 2 ? LOW : HIGH);
}

// ---- Setup / loop ---------------------------------------------------------
void setup() {
  pinMode(LED4_R, OUTPUT);
  pinMode(LED4_G, OUTPUT);
  pinMode(LED4_B, OUTPUT);
  set_led(0);

  k_msgq_init(&sample_q, queue_buf, sizeof(Sample), QUEUE_DEPTH);
  k_sem_init(&tick_sem, 0, K_SEM_MAX_LIMIT);
  k_timer_init(&tick_timer, on_tick, nullptr);

  Bridge.begin();
  Bridge.provide("now_ns", now_ns);
  Bridge.provide("mcu_stats", mcu_stats);
  Bridge.provide("stream", stream);
  Bridge.provide("set_rate", set_rate);
  Bridge.provide("set_synthetic", set_synthetic);
  Bridge.provide("set_led", set_led);

  k_thread_stack_t *stack = k_thread_stack_alloc(SAMPLER_STACK, 0);
  k_tid_t tid = k_thread_create(&sampler_thread, stack, SAMPLER_STACK, sampler_main,
                                nullptr, nullptr, nullptr, SAMPLER_PRIORITY, 0, K_NO_WAIT);
  k_thread_name_set(tid, "imu_sampler");
}

void loop() {
  static std::vector<uint8_t> batch;
  static int n = 0;
  if (batch.capacity() < BATCH * sizeof(Sample)) batch.reserve(BATCH * sizeof(Sample));

  Sample s;
  bool got = k_msgq_get(&sample_q, &s, K_MSEC(50)) == 0;
  if (got) {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(&s);
    batch.insert(batch.end(), p, p + sizeof(Sample));
    n++;
  }
  // Full batch, or a partial one left over after streaming stopped.
  if (n == BATCH || (n > 0 && !got && !streaming)) {
    Bridge.notify("imu_batch", batch);
    n_sent += n;
    batch.clear();
    n = 0;
  }
}
