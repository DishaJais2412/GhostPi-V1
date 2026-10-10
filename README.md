# GhostPi (V1: 24-Class Deep Learning ONNX Classifier)

This repository contains the first generation of the **Ghost Pi** project. It serves as a proof-of-concept for classifying raw RF signals on edge devices using deep learning.

*Note: The project was later evolved into a Physical-Layer Intrusion Detection System (IDS) using a PyTorch Autoencoder to detect hardware jitter and spoofing attacks.*

## Project Architecture (V1)
In this version, Ghost Pi acts as a Live RF Signal Classifier built for the Raspberry Pi 5. It uses a custom deep learning architecture (`GhostNet Ultra V3`) to process raw radio signals and classify them into 24 distinct categories.

### Key Components

1. **Custom PyTorch Architecture (`convert_to_onnx.py`)**
   - The neural network uses a combination of **Residual Blocks** (ResBlocks) and **Squeeze-and-Excitation Blocks** (SEBlocks) to extract features from 1D IQ radio signals.
   - It was trained using mixed precision to classify 24 distinct types of signals.
   - This script demonstrates the conversion of the trained PyTorch weights (`.pt`) into an **ONNX** format for high-performance edge deployment.

2. **Live Edge Inference (`ghost_pi_live.py`)**
   - Runs directly on the Raspberry Pi 5.
   - Interfaces directly with the CC1101 radio transceiver over SPI.
   - Uses `onnxruntime` to execute the exported `ghostnet_v3_edge.onnx` model in real-time.
   - Energy gates the RF signal to avoid running inference on background noise.

3. **Data Collection (`sniff_pulses.py` & `rf_sniffer.py`)**
   - Scripts that interface with the `lgpio` edge interrupt system to precisely measure microsecond pulse widths from ASK/OOK transmissions.
   - Used to generate the custom datasets required to train the GhostNet Ultra model.

## Tech Stack
- **Hardware:** Raspberry Pi 5, CC1101 Transceiver
- **Machine Learning:** PyTorch, ONNX, ONNX Runtime
- **Signal Processing:** NumPy, lgpio, SPI

## Deployment
The pipeline was built to train on an AMD GPU setup, then exported via ONNX. The resulting model is loaded on the Pi 5 to classify live incoming RF signals asynchronously.
