import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import os
import sys
from types import ModuleType

# --- THE "PRIVATEUSEONE" FIX ---
if 'torch.privateuseone' not in sys.modules:
    dummy_module = ModuleType('torch.privateuseone')
    sys.modules['torch.privateuseone'] = dummy_module

# --- GHOSTNET ULTRA V3 ARCHITECTURE ---
class ResBlock(nn.Module):
    def __init__(self, ch, bottleneck=4, dilation=1):
        super().__init__()
        mid = ch // bottleneck
        self.net = nn.Sequential(
            nn.Conv1d(ch, mid, 1, bias=False),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid, mid, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid, ch, 1, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class SEBlock(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, ch // ratio), nn.GELU(),
            nn.Linear(ch // ratio, ch), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.fc(self.gap(x)).unsqueeze(-1)

class GhostNetUltraV3(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 128, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 512, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(512), nn.GELU(),
            nn.Conv1d(512, 768, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(768), nn.GELU(),
        )
        self.body = nn.Sequential(
            ResBlock(768, dilation=1), SEBlock(768),
            ResBlock(768, dilation=2), SEBlock(768),
            ResBlock(768, dilation=4), SEBlock(768),
            ResBlock(768, dilation=2), SEBlock(768),
            ResBlock(768, dilation=1), SEBlock(768),
            ResBlock(768, dilation=1), SEBlock(768),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768 * 2, 512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        feat = self.body(self.stem(x))
        pooled = torch.cat([self.avg_pool(feat), self.max_pool(feat)], dim=1)
        return self.head(pooled)

# --- CONVERSION LOGIC ---
torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])

model_folder = "ghostnet_ultra_best"
print(f"[*] Initializing model architecture...")
model = GhostNetUltraV3(num_classes=24)

print(f"[*] Loading weights from {model_folder}...")
checkpoint_path = os.path.join(model_folder, "data.pkl") if os.path.isdir(model_folder) else model_folder

# Force AMD/DirectML devices directly to CPU
device_mapping = {
    'privateuseone:0': 'cpu',
    'privateuseone': 'cpu',
    'cuda:0': 'cpu',
    'cuda': 'cpu'
}

try:
    # Use the device_mapping dictionary to intercept the hardware string
    checkpoint = torch.load(checkpoint_path, map_location=device_mapping, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("[+] Model weights loaded successfully.")

    print("[*] Exporting to ONNX...")
    dummy_input = torch.randn(1, 2, 1024)
    torch.onnx.export(
        model,
        dummy_input,
        "ghostnet_v3_edge.onnx",
        export_params=True,
        opset_version=14,
        input_names=['iq_signal'],
        output_names=['class_logits']
    )

    print("\n[SUCCESS] 'ghostnet_v3_edge.onnx' created!")

except Exception as e:
    print(f"\n[-] Critical Error: {e}")
