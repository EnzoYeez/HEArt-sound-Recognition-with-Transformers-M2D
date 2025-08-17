import os
import glob
import numpy as np
import torch
import librosa
from tqdm import tqdm

# ===================== Configuration ===================== #
DATA_FOLDER = "C:/Users/Yeez/Desktop/音频信号/Final/cshCode/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data"
PRETRAINED_WEIGHTS = "C:/Users/Yeez/Desktop/音频信号/Final/cshCode/m2d_vit_base-80x1001p16x16p16k/weights_ep69it3124-0.47929.pth"
SAVE_TAG = "murmur3cls"

TARGET_SR = 16000
MAX_LEN = 10  # seconds
AUSC_SITES = ["AV", "MV", "PV", "TV"]
EMBED_SIZE = 3840

# ===================== Label Mapping ===================== #
label_dict = {"absent": 0, "present": 1, "unknown": 2}

# ===================== Load Model ===================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m2d_encoder = PortableM2D(weight_file=PRETRAINED_WEIGHTS).to(device)
m2d_encoder.eval()

# ===================== Initialization ===================== #
all_feats, all_labels, all_masks, all_ids = [], [], [], []
count_total, count_skip = 0, 0

# ===================== Main Processing Loop ===================== #
for file in tqdm(os.listdir(DATA_FOLDER)):
    if not file.endswith(".txt"):
        continue

    sid = file.replace(".txt", "")
    meta_path = os.path.join(DATA_FOLDER, file)

    # --- Parse metadata --- #
    with open(meta_path, "r") as f:
        raw_lines = f.readlines()
        meta_info = {
            line.split(":")[0].strip("# ").lower(): line.split(":")[1].strip()
            for line in raw_lines if ":" in line
        }

    murmur_status = meta_info.get("murmur", "").lower()
    if murmur_status not in label_dict:
        count_skip += 1
        continue

    label = label_dict[murmur_status]
    feature_set = []
    mask_flags = []

    # --- Process each auscultation site --- #
    for site in AUSC_SITES:
        match_path = os.path.join(DATA_FOLDER, f"{sid}_{site}*.wav")
        audio_files = glob.glob(match_path)

        if not audio_files:
            feature_set.append(np.zeros(EMBED_SIZE))
            mask_flags.append(0)
            continue

        pooled = []
        for wav in audio_files:
            try:
                sig, _ = librosa.load(wav, sr=TARGET_SR)
                if len(sig) < TARGET_SR * MAX_LEN:
                    sig = np.pad(sig, (0, TARGET_SR * MAX_LEN - len(sig)))
                else:
                    sig = sig[:TARGET_SR * MAX_LEN]

                signal_tensor = torch.tensor(sig).unsqueeze(0).to(device)
                with torch.no_grad():
                    frame_repr = m2d_encoder(signal_tensor)
                mean_repr = frame_repr.mean(dim=1).squeeze(0).cpu().numpy()
                pooled.append(mean_repr)
            except:
                continue

        if pooled:
            feature_set.append(np.stack(pooled).mean(axis=0))
            mask_flags.append(1)
        else:
            feature_set.append(np.zeros(EMBED_SIZE))
            mask_flags.append(0)

    # --- Collect features and labels --- #
    all_feats.append(np.concatenate(feature_set))
    all_labels.append(label)
    all_masks.append(mask_flags)
    all_ids.append(sid)
    count_total += 1

# ===================== Save Output ===================== #
np.save(f"{SAVE_TAG}_features.npy", np.stack(all_feats))
np.save(f"{SAVE_TAG}_labels.npy", np.array(all_labels))
np.save(f"{SAVE_TAG}_valid_masks.npy", np.array(all_masks))
np.save(f"{SAVE_TAG}_subject_ids.npy", np.array(all_ids))

# ===================== Completion Notice ===================== #
print(f"\n✅ Extraction complete: {count_total} samples processed, {count_skip} skipped.")
print(f"📁 Files saved with prefix: {SAVE_TAG}_*.npy")