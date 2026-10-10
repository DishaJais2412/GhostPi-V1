"""
Ghost Pi RF Sniffer — Complete Edition
=======================================
Raspberry Pi 5 + CC1101 module
- Configures CC1101 via SPI (433.92 MHz, OOK, async serial on GDO0)
- Captures and decodes EV1527 doorbell packets via lgpio edge interrupts

Wiring:
  CC1101 VCC  → Pi Pin 17 (3.3V)
  CC1101 GND  → Pi Pin 20 (GND)
  CC1101 MOSI → Pi Pin 19 (GPIO 10)
  CC1101 MISO → Pi Pin 21 (GPIO 9)
  CC1101 SCK  → Pi Pin 23 (GPIO 11)
  CC1101 CSN  → Pi Pin 24 (GPIO 8 / CE0)
  CC1101 GDO0 → Pi Pin 22 (GPIO 25)

Install deps:
  sudo apt install python3-spidev -y
  uv add lgpio spidev
  (or: sudo pip install lgpio spidev --break-system-packages)

Run:
  sudo python3 ghost_pi_sniffer.py
"""

import time
import signal
import sys
import spidev
import lgpio
from datetime import datetime
from collections import deque

# ── Pin config ────────────────────────────────────────────────────────────────

GDO0_PIN  = 25    # BCM GPIO 25 = Physical Pin 22
SPI_BUS   = 0
SPI_DEV   = 0     # CE0 = Physical Pin 24

# ── CC1101 Register addresses ─────────────────────────────────────────────────

CC1101_IOCFG0   = 0x02
CC1101_FIFOTHR  = 0x03
CC1101_PKTCTRL0 = 0x08
CC1101_FSCTRL1  = 0x0B
CC1101_FREQ2    = 0x0D
CC1101_FREQ1    = 0x0E
CC1101_FREQ0    = 0x0F
CC1101_MDMCFG4  = 0x10
CC1101_MDMCFG3  = 0x11
CC1101_MDMCFG2  = 0x12
CC1101_MDMCFG1  = 0x13
CC1101_MDMCFG0  = 0x14
CC1101_MCSM0    = 0x18
CC1101_FOCCFG   = 0x19
CC1101_BSCFG    = 0x1A
CC1101_AGCCTRL2 = 0x1B
CC1101_AGCCTRL1 = 0x1C
CC1101_AGCCTRL0 = 0x1D
CC1101_FREND1   = 0x21
CC1101_FREND0   = 0x22
CC1101_FSCAL3   = 0x23
CC1101_FSCAL2   = 0x24
CC1101_FSCAL1   = 0x25
CC1101_FSCAL0   = 0x26

# Strobe commands
CC1101_SRES  = 0x30   # Reset
CC1101_SRX   = 0x34   # Enter RX mode
CC1101_SIDLE = 0x36   # Idle

# ── EV1527 timing tolerances (microseconds) ───────────────────────────────────

SHORT_MIN,   SHORT_MAX   =  150,   650
LONG_MIN,    LONG_MAX    =  650,  1500
SYNC_LO_MIN, SYNC_LO_MAX = 6000, 16000
MIN_BITS = 20

# ── Shared state ──────────────────────────────────────────────────────────────

_edges      = deque()
_running    = True
_seen_codes = {}
_chip       = None
_cb_handle  = None
_spi        = None

# ── CC1101 SPI helpers ────────────────────────────────────────────────────────

def _spi_write_reg(reg, value):
    _spi.xfer2([reg & 0x3F, value])

def _spi_read_reg(reg):
    result = _spi.xfer2([(reg | 0x80) & 0xFF, 0x00])
    return result[1]

def _spi_strobe(cmd):
    _spi.xfer2([cmd])
    time.sleep(0.001)

# ── CC1101 initialisation ─────────────────────────────────────────────────────

def init_cc1101():
    global _spi
    _spi = spidev.SpiDev()
    _spi.open(SPI_BUS, SPI_DEV)
    _spi.max_speed_hz = 50000    # 50 kHz — safe and slow for reliable init
    _spi.mode = 0

    print("[CC1101] Resetting...")
    _spi_strobe(CC1101_SRES)
    time.sleep(0.1)

    # Verify chip is responding — PARTNUM should be 0x00, VERSION 0x14
    partnum = _spi_read_reg(0x30 | 0x80)   # PARTNUM (status register)
    version = _spi_read_reg(0x31 | 0x80)   # VERSION (status register)
    print(f"[CC1101] PARTNUM=0x{partnum:02X}  VERSION=0x{version:02X}")
    if partnum != 0x00 or version not in (0x04, 0x14):
        print("[CC1101] WARNING: Unexpected chip ID — check SPI wiring!")
        print("         Expected PARTNUM=0x00, VERSION=0x04 or 0x14")
        print("         Continuing anyway...\n")

    print("[CC1101] Configuring for 433.92 MHz OOK / EV1527...")

    # GDO0 = async serial data output
    _spi_write_reg(CC1101_IOCFG0,   0x0D)

    # Async serial mode, infinite packet length
    _spi_write_reg(CC1101_PKTCTRL0, 0x32)

    # Frequency = 433.92 MHz
    # Formula: f = (FREQ * 26MHz) / 2^16
    # FREQ = 433.92e6 * 65536 / 26e6 = 0x10B071
    _spi_write_reg(CC1101_FREQ2,    0x10)
    _spi_write_reg(CC1101_FREQ1,    0xB0)
    _spi_write_reg(CC1101_FREQ0,    0x71)

    # Channel bandwidth ~100 kHz, data rate ~4.8 kbps (ignored in async OOK)
    _spi_write_reg(CC1101_MDMCFG4,  0xF5)
    _spi_write_reg(CC1101_MDMCFG3,  0x83)

    # OOK modulation, no sync word, no preamble detection
    _spi_write_reg(CC1101_MDMCFG2,  0x30)
    _spi_write_reg(CC1101_MDMCFG1,  0x22)
    _spi_write_reg(CC1101_MDMCFG0,  0xF8)

    # IF frequency
    _spi_write_reg(CC1101_FSCTRL1,  0x06)

    # Calibration / AGC
    _spi_write_reg(CC1101_MCSM0,    0x18)
    _spi_write_reg(CC1101_FOCCFG,   0x16)
    _spi_write_reg(CC1101_BSCFG,    0x6C)
    _spi_write_reg(CC1101_AGCCTRL2, 0x43)  # max gain
    _spi_write_reg(CC1101_AGCCTRL1, 0x40)
    _spi_write_reg(CC1101_AGCCTRL0, 0x91)

    # PA table pointer for OOK (index 1 = transmit power level 1)
    _spi_write_reg(CC1101_FREND0,   0x11)
    _spi_write_reg(CC1101_FREND1,   0x56)

    # Frequency synthesizer calibration values (from datasheet)
    _spi_write_reg(CC1101_FSCAL3,   0xE9)
    _spi_write_reg(CC1101_FSCAL2,   0x2A)
    _spi_write_reg(CC1101_FSCAL1,   0x00)
    _spi_write_reg(CC1101_FSCAL0,   0x1F)

    # Enter RX mode
    _spi_strobe(CC1101_SRX)
    time.sleep(0.05)

    print("[CC1101] Now in RX mode. GDO0 is live.\n")

# ── lgpio edge callback ───────────────────────────────────────────────────────

def _edge_callback(chip, gpio, level, timestamp):
    """Runs in lgpio C thread. Timestamp is nanoseconds."""
    if level != 2:   # 2 = watchdog, ignore
        _edges.append((timestamp, level))

# ── EV1527 decode ─────────────────────────────────────────────────────────────

def _classify(duration_us):
    if SHORT_MIN <= duration_us <= SHORT_MAX:
        return "short"
    if LONG_MIN  <= duration_us <= LONG_MAX:
        return "long"
    if SYNC_LO_MIN <= duration_us <= SYNC_LO_MAX:
        return "sync"
    return None


def _decode_ev1527(pulses):
    bits = []
    i = 0
    while i < len(pulses) - 1:
        hi_us, hi_lvl = pulses[i]
        lo_us, lo_lvl = pulses[i + 1]

        if hi_lvl != 1 or lo_lvl != 0:
            i += 1
            continue

        hi = _classify(hi_us)
        lo = _classify(lo_us)

        if   hi == "long"  and lo == "short": bits.append(1); i += 2
        elif hi == "short" and lo == "long":  bits.append(0); i += 2
        elif lo == "sync":                    i += 2          # boundary
        else:                                 i += 1          # garbage

    if len(bits) < MIN_BITS:
        return None

    addr_int = int("".join(str(b) for b in bits[:20]), 2)
    btn_bits = bits[20:24] if len(bits) >= 24 else bits[20:]
    btn_int  = int("".join(str(b) for b in btn_bits), 2) if btn_bits else 0
    raw      = "".join(str(b) for b in bits[:24])

    return {
        "raw"     : raw,
        "address" : addr_int,
        "button"  : btn_int,
        "num_bits": len(bits),
    }

# ── Packet assembler ──────────────────────────────────────────────────────────

def _process_edges():
    packet    = []
    last_ns   = None
    last_lvl  = None
    capturing = False

    snap = []
    while _edges:
        snap.append(_edges.popleft())

    for ts_ns, level in snap:
        if last_ns is None:
            last_ns  = ts_ns
            last_lvl = level
            continue

        dur_us = (ts_ns - last_ns) / 1_000

        # Long LOW after signal = sync gap = packet boundary
        if last_lvl == 0 and SYNC_LO_MIN <= dur_us <= SYNC_LO_MAX:
            if capturing and len(packet) > 4:
                result = _decode_ev1527(packet)
                if result:
                    _report(result)
            packet    = []
            capturing = True

        elif capturing:
            if dur_us < SYNC_LO_MIN:
                packet.append((dur_us, last_lvl))
            else:
                capturing = False
                packet    = []

        last_ns  = ts_ns
        last_lvl = level

# ── Reporter ──────────────────────────────────────────────────────────────────

def _report(r):
    code = r["raw"]
    _seen_codes[code] = _seen_codes.get(code, 0) + 1
    count = _seen_codes[code]
    ts    = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    print(f"\n{'='*52}")
    print(f"  DOORBELL DECODED  —  {ts}")
    print(f"{'='*52}")
    print(f"  Address   : {r['address']:,}  ({hex(r['address'])})")
    print(f"  Button    : {r['button']}  (binary {r['button']:04b})")
    print(f"  Bits      : {r['num_bits']}")
    print(f"  Raw code  : {code}")
    print(f"  Count     : {count}x")
    print(f"{'='*52}")
    if count == 1:
        print(f"  >> Save these for replay:")
        print(f"     ADDRESS = 0x{r['address']:05X}")
        print(f"     BUTTON  = 0x{r['button']:01X}")

# ── GPIO setup ────────────────────────────────────────────────────────────────

def setup_gpio():
    global _chip, _cb_handle
    _chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(_chip, GDO0_PIN, lgpio.SET_PULL_DOWN)
    lgpio.gpio_set_debounce_micros(_chip, GDO0_PIN, 0)
    _cb_handle = lgpio.callback(_chip, GDO0_PIN, lgpio.BOTH_EDGES, _edge_callback)

# ── Teardown ──────────────────────────────────────────────────────────────────

def teardown(sig=None, frame=None):
    global _running
    _running = False
    print("\n\n[!] Shutting down...")
    try:
        if _cb_handle: _cb_handle.cancel()
        if _chip is not None: lgpio.gpiochip_close(_chip)
        if _spi  is not None: _spi.close()
    except Exception:
        pass

    if _seen_codes:
        print(f"\n[+] {len(_seen_codes)} unique code(s) captured this session:")
        for code, count in _seen_codes.items():
            addr = int(code[:20], 2) if len(code) >= 20 else 0
            btn  = int(code[20:24], 2) if len(code) >= 24 else 0
            print(f"    0x{addr:05X} / btn={btn}  x{count}  raw={code}")

    sys.exit(0)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    signal.signal(signal.SIGINT,  teardown)
    signal.signal(signal.SIGTERM, teardown)

    print("╔══════════════════════════════════════════════════╗")
    print("║     Ghost Pi RF Sniffer — Complete Edition       ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Pi 5 + CC1101 / 433.92 MHz OOK / EV1527        ║")
    print("╚══════════════════════════════════════════════════╝\n")

    init_cc1101()   # Configure CC1101 over SPI first
    setup_gpio()    # Then start listening on GDO0

    print("[*] Press your HAVELLS doorbell button now...\n")

    while _running:
        if _edges:
            _process_edges()
        else:
            time.sleep(0.001)

    teardown()
