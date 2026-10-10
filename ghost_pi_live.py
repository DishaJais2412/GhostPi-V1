# -*- coding: utf-8 -*-
"""
ghost_pi.py - Live RF Signal Classifier
CC1101 + ONNX (GhostNet) on Raspberry Pi 5

Usage:
    sudo python ghost_pi.py [--model path] [--freq 433.92] [--threshold 0.55]
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import onnxruntime as ort
import spidev

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ghost_pi")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CC1101 SPI strobe commands
STROBE_SRES  = 0x30   # Chip reset
STROBE_SRX   = 0x34   # Enter RX mode
STROBE_SIDLE = 0x36   # Exit TX/RX -> IDLE

# CC1101 register addresses
REG_RX_FIFO = 0x3F    # RX FIFO
BURST_READ  = 0xC0    # Burst-read flag (OR'd with address)

# SPI bus defaults
DEFAULT_SPI_BUS     = 0
DEFAULT_SPI_DEVICE  = 0
DEFAULT_SPI_SPEED   = 50_000   # 50 kHz - safe for CC1101 config writes

# Inference defaults
DEFAULT_MODEL_PATH  = "ghostnet_v3_edge.onnx"
DEFAULT_CONFIDENCE  = 0.55   # Discard detections below this probability
DEFAULT_FREQ_MHZ    = 433.92
NUM_IQ_PAIRS        = 1024   # Samples per inference frame
POLL_INTERVAL_S     = 0.05   # 50 ms -> ~20 inferences / second

# Energy gate: skip inference when signal is pure noise
MIN_SIGNAL_ENERGY   = 0.01   # RMS threshold (after normalisation)

# Human-readable class labels - extend to match your model's output head
CLASS_LABELS: dict[int, str] = {
    0: "Noise / Unknown",
    1: "ASK/OOK Remote",
    2: "FSK Sensor",
    3: "Rolling Code Fob",
    4: "Fixed Code Fob",
    5: "PWM Gate Remote",
}

# ---------------------------------------------------------------------------
# CC1101 register map (433.92 MHz, ASK/OOK, 250 kBaud)
# ---------------------------------------------------------------------------
CC1101_CONFIG: dict[int, int] = {
    0x0B: 0x06,   # FSCTRL1  - IF frequency ~152 kHz
    0x0D: 0x10,   # FREQ2    - } Frequency word for
    0x0E: 0xB1,   # FREQ1    - }   433.92 MHz with
    0x0F: 0x3B,   # FREQ0    - }   26 MHz XTAL
    0x10: 0x2D,   # MDMCFG4  - Channel BW 101.6 kHz, DR exponent
    0x11: 0x3B,   # MDMCFG3  - Data rate mantissa -> 249.94 kBaud
    0x12: 0x13,   # MDMCFG2  - ASK/OOK, no manchester, 16/16 sync
    0x15: 0x40,   # DEVIATN  - Not used in OOK but set for FSK fallback
    0x18: 0x18,   # MCSM0    - Auto-calibrate on IDLE->RX/TX
    0x19: 0x16,   # FOCCFG   - FOC enabled, K=4 pre/post
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    timestamp: str
    class_id:  int
    label:     str
    confidence: float

    def __str__(self) -> str:
        return (
            f"{self.timestamp} | "
            f"{self.label:<22} | "
            f"{self.confidence * 100:5.1f}%"
        )


@dataclass
class SessionStats:
    total_frames:   int = 0
    gated_frames:   int = 0   # Skipped due to low energy
    detections:     int = 0
    class_counts:   dict = field(default_factory=dict)

    def record(self, det: Detection) -> None:
        self.detections += 1
        self.class_counts[det.label] = self.class_counts.get(det.label, 0) + 1

    def summary(self) -> str:
        lines = [
            "\n+========== SESSION SUMMARY ==========+",
            f"  Frames processed : {self.total_frames}",
            f"  Frames gated     : {self.gated_frames}  (low energy)",
            f"  Total detections : {self.detections}",
        ]
        if self.class_counts:
            lines.append("  Breakdown:")
            for label, count in sorted(self.class_counts.items(),
                                       key=lambda x: -x[1]):
                lines.append(f"    {label:<22} {count}")
        lines.append("+=====================================+")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def open_spi(bus: int = DEFAULT_SPI_BUS,
             device: int = DEFAULT_SPI_DEVICE,
             speed: int = DEFAULT_SPI_SPEED) -> spidev.SpiDev:
    """Open and return a configured SpiDev handle."""
    spi = spidev.SpiDev()
    try:
        spi.open(bus, device)
    except FileNotFoundError:
        log.error("SPI device /dev/spidev%d.%d not found.", bus, device)
        log.error("Enable SPI with 'sudo raspi-config' and run as root.")
        sys.exit(1)
    spi.max_speed_hz = speed
    return spi


def setup_cc1101(spi: spidev.SpiDev,
                 config: dict[int, int] = CC1101_CONFIG) -> None:
    """Reset CC1101 and write full register configuration."""
    log.info("Initialising CC1101 radio...")
    spi.xfer2([STROBE_SRES])
    time.sleep(0.1)

    for reg, val in config.items():
        spi.xfer2([reg | 0x00, val])   # Single-byte write (write bit = 0)

    spi.xfer2([STROBE_SRX])
    log.info("CC1101 locked onto %.2f MHz in RX mode.", DEFAULT_FREQ_MHZ)


def shutdown_cc1101(spi: spidev.SpiDev) -> None:
    """Safely idle the CC1101 and close the SPI bus."""
    try:
        spi.xfer2([STROBE_SIDLE])
        spi.close()
        log.info("CC1101 idled. SPI bus closed.")
    except Exception:
        pass   # Best-effort on shutdown


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------

def read_iq_tensor(spi: spidev.SpiDev,
                   num_pairs: int = NUM_IQ_PAIRS) -> np.ndarray:
    """
    Burst-read the CC1101 RX FIFO and return a (1, 2, N) float32 tensor.

    Layout expected by GhostNet:
        axis-0 channel 0 = I (in-phase)
        axis-0 channel 1 = Q (quadrature)
    """
    raw = spi.xfer2([REG_RX_FIFO | BURST_READ] + [0x00] * (num_pairs * 2))
    data = np.frombuffer(bytes(raw[1:]), dtype=np.uint8).astype(np.float32)

    # Map [0, 255] -> [-1.0, 1.0]
    normalised = (data - 128.0) / 128.0

    i_chan = normalised[0::2][:num_pairs]
    q_chan = normalised[1::2][:num_pairs]

    return np.stack([i_chan, q_chan], axis=0).reshape(1, 2, num_pairs)


def signal_energy(tensor: np.ndarray) -> float:
    """Return RMS amplitude across both IQ channels."""
    return float(np.sqrt(np.mean(tensor ** 2)))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max()
    exp     = np.exp(shifted)
    return exp / exp.sum()


# ---------------------------------------------------------------------------
# ONNX session
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> ort.InferenceSession:
    """Load and return an ONNX Runtime inference session."""
    if not os.path.exists(model_path):
        log.error("Model file not found: %s", model_path)
        log.error("Ensure the .onnx (and .onnx.data) files are present.")
        sys.exit(1)

    log.info("Loading ONNX model: %s", model_path)
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    log.info("GhostNet inference engine online. Input: %s",
             session.get_inputs()[0].name)
    return session


def run_inference(session: ort.InferenceSession,
                  tensor: np.ndarray) -> tuple[int, float, np.ndarray]:
    """
    Run a single inference pass.

    Returns:
        (predicted_class_index, max_confidence, full_probability_vector)
    """
    input_name = session.get_inputs()[0].name
    logits     = session.run(None, {input_name: tensor})[0][0]
    probs      = softmax(logits)
    pred       = int(np.argmax(probs))
    return pred, float(probs[pred]), probs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ghost Pi - RF Signal Classifier")
    p.add_argument("--model",     default=DEFAULT_MODEL_PATH,
                   help="Path to GhostNet .onnx model file")
    p.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE,
                   help="Minimum confidence to report a detection (0-1)")
    p.add_argument("--energy",    type=float, default=MIN_SIGNAL_ENERGY,
                   help="Minimum RMS energy to run inference (noise gate)")
    p.add_argument("--verbose",   action="store_true",
                   help="Print every frame, not just detections")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BANNER = r"""
  ___ _               _     ___ _
 / __| |_  ___  ___| |_   | _ (_)
| (_ | ' \/ _ \(_-<|  _|  |  _/ |
 \___|_||_\___//__/ \__|  |_| |_|
  Live RF Signal Classifier  v3.1
"""


def main() -> None:
    args  = parse_args()
    stats = SessionStats()

    print(BANNER)

    # -- Hardware ---------------------------------------------------------
    spi = open_spi()
    setup_cc1101(spi)

    # -- Model ------------------------------------------------------------
    session: Optional[ort.InferenceSession] = None
    try:
        session = load_model(args.model)
    except SystemExit:
        shutdown_cc1101(spi)
        raise

    # -- Header -----------------------------------------------------------
    print("\n[SNIFFER ACTIVE]  Threshold: "
          f"{args.threshold * 100:.0f}%  |  Energy gate: {args.energy:.3f}\n")
    print(f"{'TIME':<8}  {'CLASS':<22}  {'CONF':>5}")
    print("-" * 42)

    # -- Inference loop ---------------------------------------------------
    try:
        while True:
            tensor = read_iq_tensor(spi)
            stats.total_frames += 1

            # Energy gate: skip noisy frames
            rms = signal_energy(tensor)
            if rms < args.energy:
                stats.gated_frames += 1
                if args.verbose:
                    log.debug("Frame gated (RMS=%.4f)", rms)
                time.sleep(POLL_INTERVAL_S)
                continue

            pred_class, confidence, _ = run_inference(session, tensor)

            if args.verbose or confidence >= args.threshold:
                label = CLASS_LABELS.get(pred_class,
                                         f"Class {pred_class}")
                det = Detection(
                    timestamp  = time.strftime("%H:%M:%S"),
                    class_id   = pred_class,
                    label      = label,
                    confidence = confidence,
                )

                if confidence >= args.threshold:
                    stats.record(det)
                    print(det)
                elif args.verbose:
                    print(f"{det}  [below threshold]")

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\n[!] Interrupt received - shutting down...")

    finally:
        shutdown_cc1101(spi)
        print(stats.summary())


if __name__ == "__main__":
    main()
