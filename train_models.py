# # ============================================================================
# # STEP 3: FIXED - Train CNN Models on RGB Composite CWT Representations
# # ============================================================================

# import os
# import json
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import fbeta_score, roc_auc_score, f1_score, roc_curve, confusion_matrix
# from tqdm import tqdm
# import timm
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ============================================================================
# # FIXED CONFIGURATION
# # ============================================================================

# PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
# BATCH_SIZE = 16  # ✅ FIXED: Reduced for Swin
# ACCUMULATION_STEPS = 2  # ✅ FIXED: Effective batch = 32
# EPOCHS = 100  # ✅ FIXED: More epochs for transformers
# LR = 0.001
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# NUM_WORKERS = 1

# print("="*80)
# print("STEP 3: TRAIN FIXED MODELS ON RGB COMPOSITE CWT")
# print("="*80)
# print(f"Device: {DEVICE}")
# print(f"Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUMULATION_STEPS})")

# # ============================================================================
# # THRESHOLD FUNCTIONS
# # ============================================================================

# def find_optimal_thresholds(y_true, y_scores):
#     """Find optimal thresholds per class using ROC curve"""
#     thresholds = []
#     for i in range(y_true.shape[1]):
#         fpr, tpr, threshold = roc_curve(y_true[:, i], y_scores[:, i])
#         optimal_idx = np.argmax(tpr - fpr)
#         thresholds.append(threshold[optimal_idx])
#     return np.array(thresholds)

# def apply_thresholds(y_scores, thresholds):
#     """Apply class-wise thresholds"""
#     y_pred = (y_scores > thresholds).astype(int)
#     for i, pred in enumerate(y_pred):
#         if pred.sum() == 0:
#             y_pred[i, np.argmax(y_scores[i])] = 1
#     return y_pred

# # ============================================================================
# # LOSS FUNCTIONS
# # ============================================================================

# class FocalLoss(nn.Module):
#     """Focal Loss for class imbalance"""
    
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
    
#     def forward(self, inputs, targets):
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

# # ============================================================================
# # FIXED DATASET CLASS
# # ============================================================================

# class CWTDataset(Dataset):
#     """Dataset for RGB composite CWT representations"""
    
#     def __init__(self, scalo_path, phaso_path, labels, mode='scalogram', augment=False):
#         self.scalograms = np.load(scalo_path, mmap_mode='r', allow_pickle=True)
#         self.phasograms = np.load(phaso_path, mmap_mode='r', allow_pickle=True)
#         self.labels = torch.FloatTensor(labels)
#         self.mode = mode
#         self.augment = augment
        
#         print(f"  Dataset: {len(self.labels)} samples, mode={mode}, shape={self.scalograms.shape}")
    
#     def __len__(self):
#         return len(self.labels)
    
#     def _augment_image(self, img):
#         """Light augmentation for CWT images"""
#         if torch.rand(1).item() > 0.5:
#             img = torch.flip(img, dims=[2])  # Horizontal flip
        
#         if torch.rand(1).item() > 0.7:
#             img = torch.flip(img, dims=[1])  # Vertical flip
        
#         if torch.rand(1).item() > 0.5:
#             brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
#             img = torch.clamp(img * brightness, 0, 1)
        
#         return img
    
#     def __getitem__(self, idx):
#         scalo = torch.FloatTensor(np.array(self.scalograms[idx], copy=True))
#         phaso = torch.FloatTensor(np.array(self.phasograms[idx], copy=True))
#         label = self.labels[idx]
        
#         if self.augment:
#             scalo = self._augment_image(scalo)
#             phaso = self._augment_image(phaso)
        
#         if self.mode == 'scalogram':
#             return scalo, label
#         elif self.mode == 'phasogram':
#             return phaso, label
#         elif self.mode == 'both':
#             return (scalo, phaso), label
#         elif self.mode == 'fusion':
#             fused = torch.cat([scalo, phaso], dim=0)  # (6, H, W)
#             return fused, label
#         else:
#             raise ValueError(f"Unknown mode: {self.mode}")

# # ============================================================================
# # FIXED MODELS - NO CHANNEL ADAPTERS!
# # ============================================================================

# class SwinTransformerECG(nn.Module):
#     """
#     ✅ FIXED Swin Transformer - Direct 3-channel RGB input
#     NO channel adapter needed!
#     """
    
#     def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
#                  model_name='swin_base_patch4_window7_224'):
#         super().__init__()
        
#         self.backbone = timm.create_model(
#             model_name,
#             pretrained=pretrained,
#             num_classes=0,
#             in_chans=3  # ✅ Direct 3-channel RGB input
#         )
        
#         num_features = self.backbone.num_features
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(num_features, 512),
#             nn.GELU(),
#             nn.LayerNorm(512),
#             nn.Dropout(dropout / 2),
#             nn.Linear(512, num_classes)
#         )
        
#         n_params = sum(p.numel() for p in self.parameters())
#         print(f"  SwinTransformerECG: {n_params/1e6:.1f}M parameters")
    
#     def forward(self, x):
#         # x: (B, 3, 224, 224) RGB composite
#         # ✅ FIXED: ImageNet normalization
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
#         x = (x - mean) / std
        
#         features = self.backbone(x)
#         return self.classifier(features)


# class SwinTransformerEarlyFusion(nn.Module):
#     """
#     ✅ FIXED Early Fusion: 6 channels (3 scalo + 3 phaso) → 3
#     """
    
#     def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
#                  model_name='swin_base_patch4_window7_224'):
#         super().__init__()
        
#         self.adapter = nn.Conv2d(6, 3, kernel_size=1, bias=False)  # ✅ FIXED: 6→3
        
#         self.backbone = timm.create_model(
#             model_name,
#             pretrained=pretrained,
#             num_classes=0,
#             in_chans=3
#         )
        
#         num_features = self.backbone.num_features
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(num_features, 512),
#             nn.GELU(),
#             nn.LayerNorm(512),
#             nn.Dropout(dropout / 2),
#             nn.Linear(512, num_classes)
#         )
        
#         n_params = sum(p.numel() for p in self.parameters())
#         print(f"  SwinTransformerEarlyFusion: {n_params/1e6:.1f}M parameters")
    
#     def forward(self, x):
#         # x: (B, 6, 224, 224)
#         x = self.adapter(x)  # → (B, 3, 224, 224)
        
#         # ImageNet normalization
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
#         x = (x - mean) / std
        
#         features = self.backbone(x)
#         return self.classifier(features)


# class SwinTransformerLateFusion(nn.Module):
#     """
#     ✅ FIXED Late Fusion: Dual stream, no adapters needed
#     """
    
#     def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
#                  model_name='swin_base_patch4_window7_224'):
#         super().__init__()
        
#         self.backbone_scalogram = timm.create_model(
#             model_name, pretrained=pretrained, num_classes=0, in_chans=3
#         )
        
#         self.backbone_phasogram = timm.create_model(
#             model_name, pretrained=pretrained, num_classes=0, in_chans=3
#         )
        
#         num_features = self.backbone_scalogram.num_features
        
#         self.fusion = nn.Sequential(
#             nn.Linear(num_features * 2, 1024),
#             nn.GELU(),
#             nn.LayerNorm(1024),
#             nn.Dropout(dropout)
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(1024, 512),
#             nn.GELU(),
#             nn.LayerNorm(512),
#             nn.Dropout(dropout / 2),
#             nn.Linear(512, num_classes)
#         )
        
#         n_params = sum(p.numel() for p in self.parameters())
#         print(f"  SwinTransformerLateFusion: {n_params/1e6:.1f}M parameters")
    
#     def forward(self, scalogram, phasogram):
#         # Both: (B, 3, 224, 224) RGB composites
        
#         # ImageNet normalization
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(scalogram.device)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(scalogram.device)
        
#         scalo_norm = (scalogram - mean) / std
#         phaso_norm = (phasogram - mean) / std
        
#         feat_scalo = self.backbone_scalogram(scalo_norm)
#         feat_phaso = self.backbone_phasogram(phaso_norm)
        
#         combined = torch.cat([feat_scalo, feat_phaso], dim=1)
#         fused = self.fusion(combined)
#         return self.classifier(fused)


# class ResNet2DCNN(nn.Module):
#     """Simple 2D CNN baseline for comparison"""
    
#     def __init__(self, num_classes=5, num_channels=3):
#         super().__init__()
        
#         from torchvision.models import resnet50, ResNet50_Weights
        
#         self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
#         # Modify first conv if needed
#         if num_channels != 3:
#             self.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
#         # Replace final FC
#         num_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )
        
#         n_params = sum(p.numel() for p in self.parameters())
#         print(f"  ResNet2DCNN: {n_params/1e6:.1f}M parameters")
    
#     def forward(self, x):
#         # ImageNet normalization for pretrained model
#         if x.shape[1] == 3:
#             mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
#             std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
#             x = (x - mean) / std
        
#         return self.backbone(x)
    
# # ============================================================================
# # CNN BASELINE MODELS
# # ============================================================================

# class ResidualBlock2D(nn.Module):
#     """Residual block for 2D CNN"""
    
#     def __init__(self, in_ch, out_ch, stride=1, downsample=None):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_ch)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#         self.downsample = downsample
    
#     def forward(self, x):
#         identity = x
        
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
        
#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         out += identity
#         out = F.relu(out)
#         return out


# class CWT2DCNN(nn.Module):
#     """2D CNN for CWT representations"""
    
#     def __init__(self, num_classes=5, num_channels=12):
#         super().__init__()
        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(3, stride=2, padding=1)
#         )
        
#         self.layer1 = self._make_layer(64, 64, 2)
#         self.layer2 = self._make_layer(64, 128, 2, stride=2)
#         self.layer3 = self._make_layer(128, 256, 2, stride=2)
#         self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
#         self.fc = nn.Sequential(
#             nn.Linear(512 * 2, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )
        
#         n_params = sum(p.numel() for p in self.parameters())
#         print(f"  CWT2DCNN: {n_params/1e6:.1f}M parameters")
    
#     def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
#         layers = []
#         layers.append(self._make_block(in_ch, out_ch, stride))
#         for _ in range(1, num_blocks):
#             layers.append(self._make_block(out_ch, out_ch))
#         return nn.Sequential(*layers)
    
#     def _make_block(self, in_ch, out_ch, stride=1):
#         downsample = None
#         if stride != 1 or in_ch != out_ch:
#             downsample = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 1, stride=stride),
#                 nn.BatchNorm2d(out_ch)
#             )
#         return ResidualBlock2D(in_ch, out_ch, stride, downsample)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x_avg = self.avgpool(x)
#         x_max = self.maxpool(x)
#         x = torch.cat([x_avg, x_max], dim=1).flatten(1)
        
#         return self.fc(x)


# class DualStreamCNN(nn.Module):
#     """Dual-stream CNN for scalogram + phasogram fusion"""
    
#     def __init__(self, num_classes=5, num_channels=12):
#         super().__init__()
        
#         self.scalogram_branch = CWT2DCNN(num_classes, num_channels)
#         self.phasogram_branch = CWT2DCNN(num_classes, num_channels)
        
#         self.scalogram_branch.fc = nn.Identity()
#         self.phasogram_branch.fc = nn.Identity()
        
#         self.fusion_fc = nn.Sequential(
#             nn.Linear(512 * 2 * 2, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )
        
#         n_params = sum(p.numel() for p in self.parameters())
#         print(f"  DualStreamCNN: {n_params/1e6:.1f}M parameters")
    
#     def forward(self, scalogram, phasogram):
#         feat_scalo = self.scalogram_branch(scalogram)
#         feat_phaso = self.phasogram_branch(phasogram)
        
#         combined = torch.cat([feat_scalo, feat_phaso], dim=1)
#         return self.fusion_fc(combined)


# # ============================================================================
# #TRAINING FUNCTIONS WITH GRADIENT ACCUMULATION
# # ============================================================================

# def train_epoch(model, dataloader, criterion, optimizer, device, is_dual=False, accumulation_steps=1):
#     """✅ FIXED: Train with gradient accumulation"""
#     model.train()
#     running_loss = 0.0
#     optimizer.zero_grad()
    
#     pbar = tqdm(dataloader, desc="Training", leave=False)
#     for batch_idx, batch in enumerate(pbar):
#         if is_dual:
#             (x1, x2), y = batch
#             x1, x2, y = x1.to(device), x2.to(device), y.to(device)
#             outputs = model(x1, x2)
#         else:
#             x, y = batch
#             x, y = x.to(device), y.to(device)
#             outputs = model(x)
        
#         loss = criterion(outputs, y) / accumulation_steps  # ✅ Scale loss
#         loss.backward()
        
#         if (batch_idx + 1) % accumulation_steps == 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             optimizer.zero_grad()
        
#         running_loss += loss.item() * accumulation_steps * y.size(0)
#         pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
#     return running_loss / len(dataloader.dataset)


# @torch.no_grad()
# def validate(model, dataloader, criterion, device, is_dual=False):
#     """Validate model"""
#     model.eval()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []
    
#     pbar = tqdm(dataloader, desc="Validating", leave=False)
#     for batch in pbar:
#         if is_dual:
#             (x1, x2), y = batch
#             x1, x2 = x1.to(device), x2.to(device)
#             out = model(x1, x2)
#         else:
#             x, y = batch
#             x = x.to(device)
#             out = model(x)
        
#         loss = criterion(out, y.to(device))
#         running_loss += loss.item() * y.size(0)
        
#         probs = torch.sigmoid(out).cpu().numpy()
#         all_preds.append(probs)
#         all_labels.append(y.numpy())
    
#     return running_loss / len(dataloader.dataset), np.vstack(all_preds), np.vstack(all_labels)


# def compute_metrics(y_true, y_pred, y_scores):
#     """Compute evaluation metrics"""
#     try:
#         macro_auc = roc_auc_score(y_true, y_scores, average='macro')
#     except:
#         macro_auc = 0.0
    
#     f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
#     f_beta = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
    
#     return {
#         'macro_auc': macro_auc,
#         'f1_macro': f1_macro,
#         'f_beta_macro': f_beta
#     }


# def plot_confusion_matrix_all_classes(y_true, y_pred, class_names, save_path=None):
#     """Plot confusion matrix"""
#     y_true_single = np.argmax(y_true, axis=1)
#     y_pred_single = np.argmax(y_pred, axis=1)
    
#     cm = confusion_matrix(y_true_single, y_pred_single, labels=range(len(class_names)))
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()

# # ============================================================================
# # FIXED MAIN TRAINING PIPELINE
# # ============================================================================

# def train_model(config, metadata, device):
#     """Train a single model configuration"""
    
#     print(f"\n{'='*80}")
#     print(f"Training: {config['name']}")
#     print(f"{'='*80}")
    
#     # Load labels
#     y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
#     y_val = np.load(os.path.join(PROCESSED_PATH, 'y_val.npy'))
#     y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    
#     # Create datasets - USE FIXED FILES!
#     mode = config['mode']
#     is_dual = (config['model'] == 'SwinTransformerLateFusion')
    
#     print(f"\nCreating datasets (mode={mode})...")
#     train_dataset = CWTDataset(
#         os.path.join(PROCESSED_PATH, 'train_scalograms.npy'),  # ✅ FIXED files
#         os.path.join(PROCESSED_PATH, 'train_phasograms.npy'),
#         y_train, mode=mode, augment=True
#     )
#     val_dataset = CWTDataset(
#         os.path.join(PROCESSED_PATH, 'val_scalograms.npy'),
#         os.path.join(PROCESSED_PATH, 'val_phasograms.npy'),
#         y_val, mode=mode, augment=False
#     )
#     test_dataset = CWTDataset(
#         os.path.join(PROCESSED_PATH, 'test_scalograms.npy'),
#         os.path.join(PROCESSED_PATH, 'test_phasograms.npy'),
#         y_test, mode=mode, augment=False
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
#         num_workers=NUM_WORKERS, pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=BATCH_SIZE, shuffle=False,
#         num_workers=NUM_WORKERS, pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=BATCH_SIZE, shuffle=False,
#         num_workers=NUM_WORKERS, pin_memory=True
#     )
    
#     # Create model
#     print(f"\nCreating fixed model...")
#     num_classes = metadata['num_classes']
    
#     if config['model'] == 'SwinTransformerECG':
#         model = SwinTransformerECG(num_classes=num_classes, pretrained=True)
#     elif config['model'] == 'SwinTransformerEarlyFusion':
#         model = SwinTransformerEarlyFusion(num_classes=num_classes, pretrained=True)
#     elif config['model'] == 'SwinTransformerLateFusion':
#         model = SwinTransformerLateFusion(num_classes=num_classes, pretrained=True)
#     elif config['model'] == 'ResNet2DCNN':
#         num_ch = 6 if mode == 'fusion' else 3
#         model = ResNet2DCNN(num_classes=num_classes, num_channels=num_ch)
#     else:
#         raise ValueError(f"Unknown model: {config['model']}")
    
#     model = model.to(device)
    
#     # Loss function
#     loss_type = config.get('loss', 'bce')
#     if loss_type == 'focal':
#         criterion = FocalLoss(alpha=0.25, gamma=2.0)
#         print(f"Using Focal Loss")
#     else:
#         criterion = nn.BCEWithLogitsLoss()
#         print(f"Using BCE Loss")
    
#     # ✅ FIXED: Proper learning rates
#     if 'Swin' in config['model']:
#         lr = 3e-5  # ✅ Lower for Swin
#         print(f"Using LR={lr} (Swin Transformer)")
#     else:
#         lr = 1e-4
#         print(f"Using LR={lr}")
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#         optimizer, T_0=10, T_mult=2, eta_min=1e-7
#     )
    
#     # Training loop
#     print(f"\nTraining for {EPOCHS} epochs...")
#     best_val_auc = 0.0
#     best_thresholds = None
#     history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}
    
#     for epoch in range(EPOCHS):
#         print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
#         # Train with gradient accumulation
#         train_loss = train_epoch(
#             model, train_loader, criterion, optimizer, device, 
#             is_dual=is_dual, accumulation_steps=ACCUMULATION_STEPS
#         )
        
#         # Validate
#         val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device, is_dual)
        
#         # Compute metrics
#         thresholds = find_optimal_thresholds(val_labels, val_preds)
#         val_pred_binary = apply_thresholds(val_preds, thresholds)
#         val_metrics = compute_metrics(val_labels, val_pred_binary, val_preds)
        
#         # Update history
#         history['train_loss'].append(train_loss)
#         history['val_loss'].append(val_loss)
#         history['val_auc'].append(val_metrics['macro_auc'])
#         history['val_f1'].append(val_metrics['f1_macro'])
        
#         print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#         print(f"Val AUC: {val_metrics['macro_auc']:.4f} | Val F1: {val_metrics['f1_macro']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
#         # Save best model
#         if val_metrics['macro_auc'] > best_val_auc:
#             best_val_auc = val_metrics['macro_auc']
#             best_thresholds = thresholds.copy()
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_auc': best_val_auc,
#                 'thresholds': best_thresholds,
#                 'config': config
#             }, os.path.join(PROCESSED_PATH, f"best_{config['name']}.pth"))
#             print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        
#         scheduler.step()
    
#     # Test with best model
#     print(f"\nTesting {config['name']}...")
#     checkpoint = torch.load(os.path.join(PROCESSED_PATH, f"best_{config['name']}.pth"))
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device, is_dual)
#     test_pred_binary = apply_thresholds(test_preds, best_thresholds)
#     test_metrics = compute_metrics(test_labels, test_pred_binary, test_preds)
    
#     print(f"\nTest Results - {config['name']}:")
#     print(f"  AUC:    {test_metrics['macro_auc']:.4f}")
#     print(f"  F1:     {test_metrics['f1_macro']:.4f}")
#     print(f"  F-beta: {test_metrics['f_beta_macro']:.4f}")
    
#     # Save confusion matrix
#     try:
#         plot_confusion_matrix_all_classes(
#             test_labels, test_pred_binary, metadata['classes'],
#             save_path=os.path.join(PROCESSED_PATH, f"confusion_{config['name']}.png")
#         )
#         print(f"✓ Confusion matrix saved")
#     except Exception as e:
#         print(f"❌ Error: {e}")
    
#     # Save results
#     results = {
#         'config': config,
#         'best_val_auc': best_val_auc,
#         'test_metrics': test_metrics,
#         'optimal_thresholds': best_thresholds.tolist(),
#         'history': history
#     }
    
#     with open(os.path.join(PROCESSED_PATH, f"results_{config['name']}.json"), 'w') as f:
#         json.dump(results, f, indent=2)
    
#     return results


# def main():
#     # Load metadata
#     print("\n[1/2] Loading metadata...")
#     with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
#         metadata = pickle.load(f)
    
#     print(f"Dataset info:")
#     print(f"  Classes: {metadata['classes']}")
#     print(f"  Train: {metadata['train_size']}, Val: {metadata['val_size']}, Test: {metadata['test_size']}")
    
#     # ✅ FIXED: Use fixed models only
#     configs = [
        
#         ## Add the Baseline CNN models with both losses with BCE 
        
#         {'mode': 'scalogram', 'model': 'SwinTransformerECG', 'name': 'Swin-Scalo', 'loss': 'bce'},
#         {'mode': 'phasogram', 'model': 'SwinTransformerECG', 'name': 'Swin-Phaso', 'loss': 'bce'},
#         {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'Swin-EarlyFusion', 'loss': 'bce'},
#         {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'Swin-LateFusion', 'loss': 'bce'},
#         {'mode': 'scalogram', 'model': 'ResNet2DCNN', 'name': 'ResNet50-Baseline', 'loss': 'bce'},
#     ]
    
#     # Train all models
#     print("\n[2/2] Training fixed models...")
#     all_results = {}
    
#     for config in configs:
#         try:
#             results = train_model(config, metadata, DEVICE)
#             all_results[config['name']] = results['test_metrics']
#         except Exception as e:
#             print(f"\n❌ Error training {config['name']}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     # Final comparison
#     print("\n" + "="*80)
#     print("FINAL RESULTS COMPARISON")
#     print("="*80)
#     print(f"{'Model':<40} | {'AUC':<8} | {'F1':<8} | {'F-beta':<8}")
#     print("-" * 70)
    
#     for name, metrics in sorted(all_results.items(), key=lambda x: x[1]['macro_auc'], reverse=True):
#         print(f"{name:<40} | {metrics['macro_auc']:.4f}   | "
#               f"{metrics['f1_macro']:.4f}   | {metrics['f_beta_macro']:.4f}")
    
#     # Save final results
#     with open(os.path.join(PROCESSED_PATH, 'final_results.json'), 'w') as f:
#         json.dump(all_results, f, indent=2)
    
#     # Find best model
#     if all_results:
#         best_model = max(all_results.items(), key=lambda x: x[1]['macro_auc'])
#         print(f"\n🏆 Best Model: {best_model[0]}")
#         print(f"   AUC: {best_model[1]['macro_auc']:.4f}")
#         print(f"   F1:  {best_model[1]['f1_macro']:.4f}")
    
#     print("\n" + "="*80)
#     print("✅ TRAINING COMPLETE WITH FIXED MODELS!")
#     print("="*80)


# if __name__ == '__main__':
#     main()

# ============================================================================
# STEP 3: FIXED - Train CNN Models on RGB Composite CWT Representations
# ============================================================================

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import fbeta_score, roc_auc_score, f1_score, roc_curve, confusion_matrix
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# FIXED CONFIGURATION
# ============================================================================

PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
OUTPUT_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/outputs/'
BATCH_SIZE = 16  # ✅ FIXED: Reduced for Swin
ACCUMULATION_STEPS = 2  # ✅ FIXED: Effective batch = 32
EPOCHS = 65  # ✅ FIXED: More epochs for transformers
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1
os.makedirs(OUTPUT_PATH, exist_ok=True)
print("="*80)
print("STEP 3: TRAIN FIXED MODELS ON RGB COMPOSITE CWT")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUMULATION_STEPS})")

# ============================================================================
# THRESHOLD FUNCTIONS
# ============================================================================

def find_optimal_thresholds(y_true, y_scores):
    """Find optimal thresholds per class using ROC curve"""
    thresholds = []
    for i in range(y_true.shape[1]):
        fpr, tpr, threshold = roc_curve(y_true[:, i], y_scores[:, i])
        optimal_idx = np.argmax(tpr - fpr)
        thresholds.append(threshold[optimal_idx])
    return np.array(thresholds)

def apply_thresholds(y_scores, thresholds):
    """Apply class-wise thresholds"""
    y_pred = (y_scores > thresholds).astype(int)
    for i, pred in enumerate(y_pred):
        if pred.sum() == 0:
            y_pred[i, np.argmax(y_scores[i])] = 1
    return y_pred

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# FIXED DATASET CLASS
# ============================================================================

class CWTDataset(Dataset):
    """Dataset for RGB composite CWT representations"""
    
    def __init__(self, scalo_path, phaso_path, labels, mode='scalogram', augment=False):
        self.scalograms = np.load(scalo_path, mmap_mode='r', allow_pickle=True)
        self.phasograms = np.load(phaso_path, mmap_mode='r', allow_pickle=True)
        self.labels = torch.FloatTensor(labels)
        self.mode = mode
        self.augment = augment
        
        print(f"  Dataset: {len(self.labels)} samples, mode={mode}, shape={self.scalograms.shape}")
    
    def __len__(self):
        return len(self.labels)
    
    def _augment_image(self, img):
        """Light augmentation for CWT images"""
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[2])  # Horizontal flip
        
        if torch.rand(1).item() > 0.7:
            img = torch.flip(img, dims=[1])  # Vertical flip
        
        if torch.rand(1).item() > 0.5:
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            img = torch.clamp(img * brightness, 0, 1)
        
        return img
    
    def __getitem__(self, idx):
        scalo = torch.FloatTensor(np.array(self.scalograms[idx], copy=True))
        phaso = torch.FloatTensor(np.array(self.phasograms[idx], copy=True))
        label = self.labels[idx]
        
        if self.augment:
            scalo = self._augment_image(scalo)
            phaso = self._augment_image(phaso)
        
        if self.mode == 'scalogram':
            return scalo, label
        elif self.mode == 'phasogram':
            return phaso, label
        elif self.mode == 'both':
            return (scalo, phaso), label
        elif self.mode == 'fusion':
            fused = torch.cat([scalo, phaso], dim=0)  # (6, H, W)
            return fused, label
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

# ============================================================================
# FIXED MODELS - NO CHANNEL ADAPTERS!
# ============================================================================

class SwinTransformerECG(nn.Module):
    """
    ✅ FIXED Swin Transformer - Direct 3-channel RGB input
    NO channel adapter needed!
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_base_patch4_window7_224'):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3  # ✅ Direct 3-channel RGB input
        )
        
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  SwinTransformerECG: {n_params/1e6:.1f}M parameters")
    
    def forward(self, x):
        # x: (B, 3, 224, 224) RGB composite
        # ✅ FIXED: ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        features = self.backbone(x)
        return self.classifier(features)


class SwinTransformerEarlyFusion(nn.Module):
    """
    ✅ FIXED Early Fusion: 6 channels (3 scalo + 3 phaso) → 3
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_base_patch4_window7_224'):
        super().__init__()
        
        self.adapter = nn.Conv2d(6, 3, kernel_size=1, bias=False)  # ✅ FIXED: 6→3
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  SwinTransformerEarlyFusion: {n_params/1e6:.1f}M parameters")
    
    def forward(self, x):
        # x: (B, 6, 224, 224)
        x = self.adapter(x)  # → (B, 3, 224, 224)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        features = self.backbone(x)
        return self.classifier(features)


class SwinTransformerLateFusion(nn.Module):
    """
    ✅ FIXED Late Fusion: Dual stream, no adapters needed
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_base_patch4_window7_224'):
        super().__init__()
        
        self.backbone_scalogram = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, in_chans=3
        )
        
        self.backbone_phasogram = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, in_chans=3
        )
        
        num_features = self.backbone_scalogram.num_features
        
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  SwinTransformerLateFusion: {n_params/1e6:.1f}M parameters")
    
    def forward(self, scalogram, phasogram):
        # Both: (B, 3, 224, 224) RGB composites
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(scalogram.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(scalogram.device)
        
        scalo_norm = (scalogram - mean) / std
        phaso_norm = (phasogram - mean) / std
        
        feat_scalo = self.backbone_scalogram(scalo_norm)
        feat_phaso = self.backbone_phasogram(phaso_norm)
        
        combined = torch.cat([feat_scalo, feat_phaso], dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)


class ResNet2DCNN(nn.Module):
    """Simple 2D CNN baseline for comparison"""
    
    def __init__(self, num_classes=5, num_channels=3):
        super().__init__()
        
        from torchvision.models import resnet50, ResNet50_Weights
        
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify first conv if needed
        if num_channels != 3:
            self.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final FC
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ResNet2DCNN: {n_params/1e6:.1f}M parameters")
    
    def forward(self, x):
        # ImageNet normalization for pretrained model
        if x.shape[1] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std
        
        return self.backbone(x)

# ============================================================================
# MODIFIED CNN BASELINE MODELS - UPDATED FOR 3-CHANNEL INPUT
# ============================================================================

class ResidualBlock2D(nn.Module):
    """Residual block for 2D CNN"""
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out


class CWT2DCNN(nn.Module):
    """2D CNN for CWT representations - MODIFIED for 3-channel input"""
    
    def __init__(self, num_classes=5, num_channels=3):  # ✅ CHANGED: default to 3 channels
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  CWT2DCNN: {n_params/1e6:.1f}M parameters")
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_ch, out_ch, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
        return ResidualBlock2D(in_ch, out_ch, stride, downsample)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)
        
        return self.fc(x)


class DualStreamCNN(nn.Module):
    """Dual-stream CNN for scalogram + phasogram fusion - MODIFIED for 3-channel input"""
    
    def __init__(self, num_classes=5, num_channels=3):  # ✅ CHANGED: default to 3 channels
        super().__init__()
        
        self.scalogram_branch = CWT2DCNN(num_classes, num_channels)
        self.phasogram_branch = CWT2DCNN(num_classes, num_channels)
        
        self.scalogram_branch.fc = nn.Identity()
        self.phasogram_branch.fc = nn.Identity()
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),  # 512*2 from each branch (avg+max pooling)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  DualStreamCNN: {n_params/1e6:.1f}M parameters")
    
    def forward(self, scalogram, phasogram):
        feat_scalo = self.scalogram_branch(scalogram)
        feat_phaso = self.phasogram_branch(phasogram)
        
        combined = torch.cat([feat_scalo, feat_phaso], dim=1)
        return self.fusion_fc(combined)


class CWT2DCNN6Channel(nn.Module):
    """2D CNN for 6-channel fusion input"""
    
    def __init__(self, num_classes=5, num_channels=6):  # For fusion mode
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  CWT2DCNN6Channel: {n_params/1e6:.1f}M parameters")
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_ch, out_ch, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
        return ResidualBlock2D(in_ch, out_ch, stride, downsample)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)
        
        return self.fc(x)

# ============================================================================
# TRAINING FUNCTIONS WITH GRADIENT ACCUMULATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, is_dual=False, accumulation_steps=1):
    """✅ FIXED: Train with gradient accumulation"""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if is_dual:
            (x1, x2), y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            outputs = model(x1, x2)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)
        
        loss = criterion(outputs, y) / accumulation_steps  # ✅ Scale loss
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps * y.size(0)
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, criterion, device, is_dual=False):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        if is_dual:
            (x1, x2), y = batch
            x1, x2 = x1.to(device), x2.to(device)
            out = model(x1, x2)
        else:
            x, y = batch
            x = x.to(device)
            out = model(x)
        
        loss = criterion(out, y.to(device))
        running_loss += loss.item() * y.size(0)
        
        probs = torch.sigmoid(out).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(y.numpy())
    
    return running_loss / len(dataloader.dataset), np.vstack(all_preds), np.vstack(all_labels)


def compute_metrics(y_true, y_pred, y_scores):
    """Compute evaluation metrics"""
    try:
        macro_auc = roc_auc_score(y_true, y_scores, average='macro')
    except:
        macro_auc = 0.0
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f_beta = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
    
    return {
        'macro_auc': macro_auc,
        'f1_macro': f1_macro,
        'f_beta_macro': f_beta
    }


def plot_confusion_matrix_all_classes(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    y_true_single = np.argmax(y_true, axis=1)
    y_pred_single = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true_single, y_pred_single, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MODIFIED MAIN TRAINING PIPELINE
# ============================================================================

def train_model(config, metadata, device):
    """Train a single model configuration"""
    
    print(f"\n{'='*80}")
    print(f"Training: {config['name']}")
    print(f"{'='*80}")
    
    # Load labels
    y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
    y_val = np.load(os.path.join(PROCESSED_PATH, 'y_val.npy'))
    y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    
    # Create datasets - USE FIXED FILES!
    mode = config['mode']
    is_dual = (config['model'] == 'SwinTransformerLateFusion') or (config['model'] == 'DualStreamCNN')  # ✅ ADDED DualStreamCNN
    
    print(f"\nCreating datasets (mode={mode})...")
    train_dataset = CWTDataset(
        os.path.join(PROCESSED_PATH, 'train_scalograms.npy'),  # ✅ FIXED files
        os.path.join(PROCESSED_PATH, 'train_phasograms.npy'),
        y_train, mode=mode, augment=True
    )
    val_dataset = CWTDataset(
        os.path.join(PROCESSED_PATH, 'val_scalograms.npy'),
        os.path.join(PROCESSED_PATH, 'val_phasograms.npy'),
        y_val, mode=mode, augment=False
    )
    test_dataset = CWTDataset(
        os.path.join(PROCESSED_PATH, 'test_scalograms.npy'),
        os.path.join(PROCESSED_PATH, 'test_phasograms.npy'),
        y_test, mode=mode, augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Create model
    print(f"\nCreating fixed model...")
    num_classes = metadata['num_classes']
    
    # ✅ MODIFIED: Added baseline CNN models
    if config['model'] == 'SwinTransformerECG':
        model = SwinTransformerECG(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'SwinTransformerEarlyFusion':
        model = SwinTransformerEarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'SwinTransformerLateFusion':
        model = SwinTransformerLateFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'ResNet2DCNN':
        num_ch = 6 if mode == 'fusion' else 3
        model = ResNet2DCNN(num_classes=num_classes, num_channels=num_ch)
    elif config['model'] == 'CWT2DCNN':
        num_ch = 6 if mode == 'fusion' else 3
        model = CWT2DCNN(num_classes=num_classes, num_channels=num_ch)
    elif config['model'] == 'CWT2DCNN6Channel':
        model = CWT2DCNN6Channel(num_classes=num_classes, num_channels=6)
    elif config['model'] == 'DualStreamCNN':
        model = DualStreamCNN(num_classes=num_classes, num_channels=3)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    
    model = model.to(device)
    
    # Loss function
    loss_type = config.get('loss', 'bce')
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print(f"Using Focal Loss")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"Using BCE Loss")
    
    # ✅ FIXED: Proper learning rates
    if 'Swin' in config['model']:
        lr = 3e-5  # ✅ Lower for Swin
        print(f"Using LR={lr} (Swin Transformer)")
    else:
        lr = 1e-4
        print(f"Using LR={lr}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    best_val_auc = 0.0
    best_thresholds = None
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train with gradient accumulation
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            is_dual=is_dual, accumulation_steps=ACCUMULATION_STEPS
        )
        
        # Validate
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device, is_dual)
        
        # Compute metrics
        thresholds = find_optimal_thresholds(val_labels, val_preds)
        val_pred_binary = apply_thresholds(val_preds, thresholds)
        val_metrics = compute_metrics(val_labels, val_pred_binary, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['macro_auc'])
        history['val_f1'].append(val_metrics['f1_macro'])
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {val_metrics['macro_auc']:.4f} | Val F1: {val_metrics['f1_macro']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_metrics['macro_auc'] > best_val_auc:
            best_val_auc = val_metrics['macro_auc']
            best_thresholds = thresholds.copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'thresholds': best_thresholds,
                'config': config
            }, os.path.join(OUTPUT_PATH, f"best_{config['name']}.pth"))
            print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        
        scheduler.step()
    
    # Test with best model
    print(f"\nTesting {config['name']}...")
    checkpoint = torch.load(os.path.join(OUTPUT_PATH, f"best_{config['name']}.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device, is_dual)
    test_pred_binary = apply_thresholds(test_preds, best_thresholds)
    test_metrics = compute_metrics(test_labels, test_pred_binary, test_preds)
    
    print(f"\nTest Results - {config['name']}:")
    print(f"  AUC:    {test_metrics['macro_auc']:.4f}")
    print(f"  F1:     {test_metrics['f1_macro']:.4f}")
    print(f"  F-beta: {test_metrics['f_beta_macro']:.4f}")
    
    # Save confusion matrix
    try:
        plot_confusion_matrix_all_classes(
            test_labels, test_pred_binary, metadata['classes'],
            save_path=os.path.join(OUTPUT_PATH, f"confusion_{config['name']}.png")
        )
        print(f"✓ Confusion matrix saved")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Save results
    results = {
        'config': config,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'optimal_thresholds': best_thresholds.tolist(),
        'history': history
    }
    
    with open(os.path.join(OUTPUT_PATH, f"results_{config['name']}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    # Load metadata
    print("\n[1/2] Loading metadata...")
    with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Dataset info:")
    print(f"  Classes: {metadata['classes']}")
    print(f"  Train: {metadata['train_size']}, Val: {metadata['val_size']}, Test: {metadata['test_size']}")
    
    # ✅ MODIFIED: Added baseline CNN models with both losses
    configs = [
        # Swin Transformer models
        {'mode': 'scalogram', 'model': 'SwinTransformerECG', 'name': 'Swin-Scalo', 'loss': 'bce'},
        # {'mode': 'phasogram', 'model': 'SwinTransformerECG', 'name': 'Swin-Phaso', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'Swin-EarlyFusion-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'Swin-EarlyFusion-Focal', 'loss': 'focal'},
        {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'Swin-LateFusion', 'loss': 'bce'},
        
        # ResNet baseline
        {'mode': 'scalogram', 'model': 'ResNet2DCNN', 'name': 'ResNet50-Baseline', 'loss': 'bce'},
        
        # Baseline CNN models with BCE loss
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'CWT2DCNN-Scalo-BCE', 'loss': 'bce'},
        # {'mode': 'phasogram', 'model': 'CWT2DCNN', 'name': 'CWT2DCNN-Phaso-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'CWT2DCNN6Channel', 'name': 'CWT2DCNN-Fusion-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'DualStreamCNN', 'name': 'DualStreamCNN-BCE', 'loss': 'bce'},
        
        # Baseline CNN models with Focal loss
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'CWT2DCNN-Scalo-Focal', 'loss': 'focal'},
        # {'mode': 'phasogram', 'model': 'CWT2DCNN', 'name': 'CWT2DCNN-Phaso-Focal', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'CWT2DCNN6Channel', 'name': 'CWT2DCNN-Fusion-Focal', 'loss': 'focal'},
        {'mode': 'both', 'model': 'DualStreamCNN', 'name': 'DualStreamCNN-Focal', 'loss': 'focal'},
    ]
    
    # Train all models
    print("\n[2/2] Training fixed models...")
    all_results = {}
    
    for config in configs:
        try:
            results = train_model(config, metadata, DEVICE)
            all_results[config['name']] = results['test_metrics']
        except Exception as e:
            print(f"\n❌ Error training {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(f"{'Model':<40} | {'AUC':<8} | {'F1':<8} | {'F-beta':<8}")
    print("-" * 70)
    
    for name, metrics in sorted(all_results.items(), key=lambda x: x[1]['macro_auc'], reverse=True):
        print(f"{name:<40} | {metrics['macro_auc']:.4f}   | "
              f"{metrics['f1_macro']:.4f}   | {metrics['f_beta_macro']:.4f}")
    
    # Save final results
    with open(os.path.join(OUTPUT_PATH, 'final_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Find best model
    if all_results:
        best_model = max(all_results.items(), key=lambda x: x[1]['macro_auc'])
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   AUC: {best_model[1]['macro_auc']:.4f}")
        print(f"   F1:  {best_model[1]['f1_macro']:.4f}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE WITH FIXED MODELS!")
    print("="*80)


if __name__ == '__main__':
    main()