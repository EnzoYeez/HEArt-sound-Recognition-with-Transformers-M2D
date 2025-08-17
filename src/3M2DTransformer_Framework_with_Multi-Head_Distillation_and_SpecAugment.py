import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
import random
import os

# ========== Dataset ========== #
class MurmurTokenDataset(Dataset):
    def __init__(self, features, masks, labels):
        self.features = torch.tensor(features.reshape(-1, 4, 3840), dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx], self.labels[idx]

# ========== SpecAugment ========== #
def apply_spec_augment(x, freq_mask_ratio=0.2, time_mask_ratio=0.25):
    B, T, F = x.size()
    for i in range(B):
        f_len = int(F * freq_mask_ratio)
        t_len = int(T * time_mask_ratio)
        f_start = random.randint(0, F - f_len)
        t_start = random.randint(0, T - t_len)
        x[i, :, f_start:f_start + f_len] = 0
        x[i, t_start:t_start + t_len, :] = 0
    return x

# ========== M2D-style Transformer with Multi-Head Classifier ========== #
class M2DTransformer(nn.Module):
    def __init__(self, token_dim=3840, hidden=512, nhead=4, num_layers=4, num_classes=3, heads=3):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, hidden)
        self.mask_embedding = nn.Embedding(2, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            dropout=0.2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = nn.Sequential(nn.Linear(hidden, 1), nn.Softmax(dim=1))
        self.heads = nn.ModuleList([nn.Linear(hidden, num_classes) for _ in range(heads)])
        self.heads_num = heads

    def forward(self, x, mask):
        x = self.token_proj(x)
        m = self.mask_embedding(mask.long())
        x = x + m
        x_encoded = self.encoder(x)
        w = self.attn_pool(x_encoded)
        pooled = torch.sum(x_encoded * w, dim=1)
        outputs = [head(pooled) for head in self.heads]
        logits = torch.stack(outputs, dim=1)  # [B, H, C]
        mean_logits = torch.mean(logits, dim=1)  # Ensemble输出
        return mean_logits, logits  # 第二个用于计算 Distillation Loss

# ========== Focal Loss + Distillation Loss ========== #
class FocalLossWithDistill(nn.Module):
    def __init__(self, gamma=1.5, smoothing=0.05, weight=None, alpha=0.7):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.weight = weight
        self.alpha = alpha  # main loss 和 distillation loss 的平衡

    def forward(self, logits_mean, logits_all, targets):
        # Focal loss on mean logits
        log_probs = F.log_softmax(logits_mean, dim=-1)
        targets_onehot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_onehot = (1 - self.smoothing) * targets_onehot + self.smoothing / log_probs.size(1)
        probs = torch.exp(log_probs)
        focal = ((1 - probs) ** self.gamma) * log_probs
        loss_main = -torch.sum(targets_onehot * focal, dim=1)
        loss_main = (loss_main * self.weight[targets]).mean() if self.weight is not None else loss_main.mean()

        # Distillation loss: variance between heads
        logits_soft = F.log_softmax(logits_all, dim=-1)
        mean_soft = F.log_softmax(logits_mean.detach(), dim=-1).unsqueeze(1)
        loss_distill = F.kl_div(logits_soft, mean_soft.expand_as(logits_soft), reduction='batchmean', log_target=True)

        return self.alpha * loss_main + (1 - self.alpha) * loss_distill

# ========== M2D Metrics ========== #
def compute_m2d_metrics(true_labels, pred_labels):
    recall_per_class = recall_score(true_labels, pred_labels, average=None, labels=[0, 1, 2])
    uar = np.mean(recall_per_class)
    counts = np.bincount(true_labels, minlength=3)
    correct = recall_per_class * counts
    wacc = (5 * correct[1] + 3 * correct[2] + correct[0]) / (5 * counts[1] + 3 * counts[2] + counts[0])
    return recall_per_class, uar, wacc

# ========== Training Loop ========== #
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=100):
    best_wacc = 0.0
    best_preds, best_targets = None, None

    for epoch in range(epochs):
        model.train()
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            x = apply_spec_augment(x)
            optimizer.zero_grad()
            mean_logits, logits_all = model(x, m)
            loss = criterion(mean_logits, logits_all, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, m, y in test_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                mean_logits, _ = model(x, m)
                pred = mean_logits.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_targets))
        macro_f1 = f1_score(all_targets, all_preds, average="macro")
        recall_per_class, uar, wacc = compute_m2d_metrics(np.array(all_targets), np.array(all_preds))

        if wacc > best_wacc:
            best_wacc = wacc
            best_preds = all_preds
            best_targets = all_targets
            torch.save(model.state_dict(), "best_m2d_model.pth")

        print(f"Epoch {epoch+1}/{epochs} - Acc: {acc:.4f} - F1: {macro_f1:.4f} - UAR: {uar:.4f} - W.acc: {wacc:.4f}")

    return best_preds, best_targets

# ========== Main Entrypoint ========== #
def main():
    features = np.load("murmur3cls_features.npy")
    masks = np.load("murmur3cls_valid_masks.npy")
    labels = np.load("murmur3cls_labels.npy")

    X_train, X_test, m_train, m_test, y_train, y_test = train_test_split(
        features, masks, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = MurmurTokenDataset(X_train, m_train, y_train)
    test_dataset = MurmurTokenDataset(X_test, m_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M2DTransformer().to(device)

    class_weights = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float32).to(device)
    criterion = FocalLossWithDistill(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # ✅ 记得接收 train() 的返回值
    best_preds, best_targets = train(model, train_loader, test_loader, criterion, optimizer, scheduler, device)

    # ✅ 保存预测结果
    np.save("best_preds.npy", np.array(best_preds))
    np.save("best_targets.npy", np.array(best_targets))

    # ✅ 保存 pooled 特征和 attention 权重
    pooled_list = []
    attention_weights = []
    model.eval()
    with torch.no_grad():
        for x, m, y in test_loader:
            x, m = x.to(device), m.to(device)
            x_proj = model.token_proj(x)
            x_proj = x_proj + model.mask_embedding(m.long())
            x_encoded = model.encoder(x_proj)
            w = model.attn_pool(x_encoded)  # [B, 4, 1]
            pooled = torch.sum(x_encoded * w, dim=1)
            pooled_list.append(pooled.cpu())
            attention_weights.append(w.squeeze(-1).cpu())

    pooled_array = torch.cat(pooled_list).numpy()
    attn_array = torch.cat(attention_weights).numpy()

    np.save("pooled_features.npy", pooled_array)
    np.save("attention_weights.npy", attn_array)

    # ✅ 写入文本报告
    report = classification_report(best_targets, best_preds, target_names=["Absent", "Present", "Unknown"])
    with open("final_m2d_report.txt", "w") as f:
        f.write(report)
        recall_per_class, uar, wacc = compute_m2d_metrics(np.array(best_targets), np.array(best_preds))
        f.write("\n=== M2D-style Metrics ===\n")
        f.write(f"Recall - Present : {recall_per_class[1]:.3f}\n")
        f.write(f"Recall - Unknown : {recall_per_class[2]:.3f}\n")
        f.write(f"Recall - Absent  : {recall_per_class[0]:.3f}\n")
        f.write(f"UAR : {uar:.3f}\nW.acc : {wacc:.3f}\n")


if __name__ == "__main__":
    main()
