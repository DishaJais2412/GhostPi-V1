
# -*- coding: utf-8 -*-
import spidev
import lgpio
import time
import sys
import csv
import statistics
import os

CHIP = 0
GPIO_PIN = 25
CSV_FILE = "havells_baseline.csv"

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 50000
spi.mode = 0

def w(addr, val): spi.xfer([addr & 0x3F, val])
def strobe(cmd):  spi.xfer([cmd]); time.sleep(0.005)

def init_cc1101():
    print("[*] Initializing CC1101 in Async OOK Mode...")
    strobe(0x30)
    time.sleep(0.01)
    w(0x02, 0x0D)
    w(0x08, 0x32)
    w(0x0B, 0x06)
    w(0x0D, 0x10); w(0x0E, 0xA7); w(0x0F, 0x62)
    w(0x10, 0x47); w(0x11, 0x83)
    w(0x12, 0x30)
    w(0x1B, 0x43); w(0x1C, 0x40); w(0x1D, 0x91)
    w(0x22, 0x11)
    strobe(0x33)
    time.sleep(0.01)
    strobe(0x34)

def setup_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Pulse_Count", "Mean_Pulse_Width_us", "Pulse_Variance", "Total_Duration_ms"])
        print(f"[*] Created new baseline file: {CSV_FILE}")
    else:
        print(f"[*] Appending to existing file: {CSV_FILE}")

last_tick = 0
pulses = []

def cbf(chip, gpio, level, timestamp_ns):
    global last_tick, pulses
    if last_tick != 0:
        duration_us = (timestamp_ns - last_tick) / 1000.0
        if duration_us > 4000:
            if len(pulses) > 20:
                pulse_count = len(pulses)
                abs_pulses = [abs(p) for p in pulses]
                mean_pw = statistics.mean(abs_pulses)
                variance_pw = statistics.variance(abs_pulses) if pulse_count > 1 else 0
                total_duration_ms = sum(abs_pulses) / 1000.0
                with open(CSV_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([pulse_count, mean_pw, variance_pw, total_duration_ms])
                print(f"[+] CAPTURED -> Count: {pulse_count:03d} | Mean: {mean_pw:6.2f}us | Var: {variance_pw:8.2f} | Total: {total_duration_ms:5.2f}ms")
            pulses = []
        elif duration_us > 100:
            pulses.append(int(duration_us) if level == 0 else -int(duration_us))
    last_tick = timestamp_ns

def main():
    print("-------------------------------------------------")
    print(" Ghost Pi - Phase 1: Hardware Fingerprint Logger")
    print("-------------------------------------------------")
    setup_csv()
    init_cc1101()
    h = lgpio.gpiochip_open(CHIP)
    lgpio.gpio_claim_alert(h, GPIO_PIN, lgpio.BOTH_EDGES)
    cb = lgpio.callback(h, GPIO_PIN, lgpio.BOTH_EDGES, cbf)  # cb stored to prevent GC
    print(f"[*] Radio Locked. Listening on GPIO {GPIO_PIN}.")
    print("[!] Press and hold your Havells doorbell ~100 times to build the baseline.")
    print("[!] Press Ctrl+C to exit when finished.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Shutting down. Data saved to havells_baseline.csv")
        cb.cancel()
        lgpio.gpiochip_close(h)
        spi.close()
        sys.exit(0)

if __name__ == "__main__":
    main()

