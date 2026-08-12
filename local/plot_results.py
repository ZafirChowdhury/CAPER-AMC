import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
from sklearn.metrics import confusion_matrix, f1_score
from pathlib import Path
import pickle

# Import architecture and dataset loader functions from your main script
from caper_amc_local import (
    CAPERAMC, DEVICE, load_rml2016_10a_to_vram,
    MIN_SNR, MAX_SNR, BATCH_SIZE, batch_normalize_gpu, amp_context
)

# 1. Locate dataset
pkl_files = list(Path(".").glob("*.pkl")) + list(Path(".").glob("*.pickle"))
dataset_path = pkl_files[0]

# 2. Load test data into VRAM
vram_data = load_rml2016_10a_to_vram(dataset_path, DEVICE, min_snr=MIN_SNR, max_snr=MAX_SNR)
CLASS_NAMES = vram_data["class_names"]
test_data = vram_data["test"]

# 3. Load the saved model checkpoint
checkpoint_path = "./caper_amc_results/caper_amc_best_vram.pth"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

model = CAPERAMC(CLASS_NAMES, MIN_SNR, MAX_SNR).to(DEVICE)
model.load_state_dict(checkpoint["ema_state_dict"])
model.eval()

print("\nGenerating evaluation metrics and plots...")

# 4. Run inference on the test set
X, y, snr = test_data["X"], test_data["y"], test_data["snr"]
all_targets, all_preds = [], []
snr_stats = {int(s): {"correct": 0, "total": 0} for s in torch.unique(snr).cpu().numpy()}

eval_batch_size = BATCH_SIZE * 2
with torch.no_grad():
    for i in range(0, X.size(0), eval_batch_size):
        inputs = batch_normalize_gpu(X[i:i + eval_batch_size])
        targets = y[i:i + eval_batch_size]
        snr_db = snr[i:i + eval_batch_size]

        with amp_context():
            outputs = model(inputs, use_gumbel=False)

        preds = outputs["logits"].argmax(dim=1)
        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        for p, t, s in zip(preds, targets, snr_db):
            s_val = int(s.item())
            snr_stats[s_val]["total"] += 1
            if p == t:
                snr_stats[s_val]["correct"] += 1

targets_np = np.concatenate(all_targets)
preds_np = np.concatenate(all_preds)

# 5. Plot and Save Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(targets_np, preds_np, normalize='true')
plt_sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Normalized Confusion Matrix - CAPER-AMC")
plt.xlabel("Predicted Modulation")
plt.ylabel("True Modulation")
plt.tight_layout()
cm_path = "./caper_amc_results/confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved Confusion Matrix to: {cm_path}")

# 6. Plot and Save Accuracy vs SNR Graph
snrs = sorted(list(snr_stats.keys()))
accuracies = [(snr_stats[s]["correct"] / snr_stats[s]["total"]) * 100 if snr_stats[s]["total"] > 0 else 0 for s in snrs]

plt.figure(figsize=(10, 5))
plt.plot(snrs, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
plt.title("Classification Accuracy Across SNR Levels")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(snrs)
plt.ylim(0, 100)
plt.tight_layout()
acc_path = "./caper_amc_results/snr_accuracy_curve.png"
plt.savefig(acc_path, dpi=300)
plt.close()
print(f"Saved Accuracy Graph to: {acc_path}")
