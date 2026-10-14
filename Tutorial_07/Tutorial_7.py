# ===== Block 1: Import libraries =====
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'  # mixed precision makes training faster on the GPU
print('Device:', device)


# ===== Block 2: Load and preprocess the dataset =====
# ToTensor() scales pixels to [0, 1]. PyTorch uses integer labels with
# CrossEntropyLoss, so one-hot encoding is not needed.
plain_transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=plain_transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=plain_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

class_names = train_data.classes
print('Classes:', class_names)
print('Training images:', len(train_data))
print('Test images:', len(test_data))


# ===== Block 3: Helper functions =====
class TransferModel(nn.Module):
    # Pretrained base + new classification layers (Flatten -> Dense 512 -> Dense 10)
    def __init__(self, base, n_features, input_size=None, dropout=0.0):
        super().__init__()
        if input_size:
            self.resize = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False)
        else:
            self.resize = nn.Identity()
        self.base = base
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 512), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.head(self.base(self.resize(x)))


def vgg16_base():
    # Convolutional part only (same as include_top=False in Keras)
    return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features


def resnet50_base():
    # Everything except the final ImageNet classification layer
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    return nn.Sequential(*list(resnet.children())[:-1])


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model, optimizer, train_loader, test_loader, epochs=10, patience=None):
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_loss, best_weights, wait = float('inf'), None, 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        # Frozen BatchNorm layers stay in inference mode (same as Keras)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
                m.eval()

        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        val_loss, val_acc = evaluate(model, test_loader)
        history['loss'].append(total_loss / total)
        history['accuracy'].append(correct / total)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / total:.4f} - accuracy: {correct / total:.4f}"
              f" - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        # Early stopping
        if patience is not None:
            if val_loss < best_loss:
                best_loss, best_weights, wait = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    history['time'] = time.time() - start
    return history


def plot_history(history, title=''):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Accuracy - {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss - {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


results = []
all_histories = {}


def record(name, model, history, loader):
    _, acc = evaluate(model, loader)
    results.append((name, count_trainable(model), acc, len(history['loss']), history['time']))
    all_histories[name] = history
    return acc


# ===== Block 4: Feature extraction using pretrained VGG16 =====
vgg_base = vgg16_base()
freeze(vgg_base)

model_feature_extraction = TransferModel(vgg_base, 512).to(device)
print('Trainable parameters:', count_trainable(model_feature_extraction))

optimizer = optim.Adam(trainable_params(model_feature_extraction), lr=0.001)
history_feature_extraction = train_model(model_feature_extraction, optimizer, train_loader, test_loader, epochs=10)


# ===== Block 5: Fine-tuning using pretrained ResNet50 =====
resnet_base = resnet50_base()
freeze(resnet_base)

model_finetune = TransferModel(resnet_base, 2048).to(device)

# Unfreeze the last 5 layers. In Keras these are the last conv + batch norm
# (plus add/relu layers with no weights) of the final ResNet50 block.
last_block = resnet_base[-2][-1]
unfreeze(last_block.conv3)
unfreeze(last_block.bn3)
print('Trainable parameters:', count_trainable(model_finetune))

optimizer = optim.Adam(trainable_params(model_finetune), lr=1e-5)
history_finetune = train_model(model_finetune, optimizer, train_loader, test_loader, epochs=10)


# ===== Block 6: Evaluate and compare both models =====
loss_feature_extraction, acc_feature_extraction = evaluate(model_feature_extraction, test_loader)
loss_finetune, acc_finetune = evaluate(model_finetune, test_loader)

print(f"Feature Extraction Accuracy: {acc_feature_extraction * 100:.2f}%")
print(f"Fine-Tuning Accuracy: {acc_finetune * 100:.2f}%")

record('VGG16 feature extraction', model_feature_extraction, history_feature_extraction, test_loader)
record('ResNet50 fine-tuning (tutorial)', model_finetune, history_finetune, test_loader)

plot_history(history_feature_extraction, 'VGG16 Feature Extraction')
plot_history(history_finetune, 'ResNet50 Fine-Tuning (Tutorial)')


# ===== Block 7: Task - Improved fine-tuning of ResNet50 =====
# Improvements:
# 1. Preprocessing: ImageNet normalization and resizing images from 32x32 to 128x128
# 2. Data augmentation (random crop + flip) to prevent overfitting
# 3. Unfreeze more layers (the whole of layer3 and layer4)
# 4. Higher learning rate for the new layers (1e-3) than for the pretrained ones (1e-4)
# 5. Dropout in the classifier and weight decay
# 6. Early stopping
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

train_data_aug = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform_aug)
test_data_norm = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform_norm)
train_loader_aug = DataLoader(train_data_aug, batch_size=64, shuffle=True, num_workers=2)
test_loader_norm = DataLoader(test_data_norm, batch_size=256, shuffle=False, num_workers=2)

improved_base = resnet50_base()
freeze(improved_base)
unfreeze(improved_base[-3])  # layer3
unfreeze(improved_base[-2])  # layer4

model_improved = TransferModel(improved_base, 2048, input_size=128, dropout=0.5).to(device)
print('Trainable parameters:', count_trainable(model_improved))

optimizer = optim.Adam([
    {'params': trainable_params(improved_base), 'lr': 1e-4},
    {'params': model_improved.head.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

history_improved = train_model(model_improved, optimizer, train_loader_aug, test_loader_norm,
                               epochs=12, patience=3)

acc_improved = record('ResNet50 fine-tuning (improved)', model_improved, history_improved, test_loader_norm)
print(f"Improved Fine-Tuning Accuracy: {acc_improved * 100:.2f}%")
plot_history(history_improved, 'ResNet50 Fine-Tuning (Improved)')


# ===== Block 8: Task - Feature extraction with ResNet50 vs custom model vs VGG16 =====
# Feature extraction with ResNet50
print("Training: ResNet50 feature extraction")
fe_resnet_base = resnet50_base()
freeze(fe_resnet_base)
model_fe_resnet = TransferModel(fe_resnet_base, 2048).to(device)
optimizer = optim.Adam(trainable_params(model_fe_resnet), lr=0.001)
history_fe_resnet = train_model(model_fe_resnet, optimizer, train_loader, test_loader, epochs=10)
record('ResNet50 feature extraction', model_fe_resnet, history_fe_resnet, test_loader)

# Custom CNN trained from scratch (the CNN from Tutorial 5)
print("\nTraining: Custom CNN")
custom_model = nn.Sequential(
    nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
    nn.Linear(64, 10)
).to(device)
optimizer = optim.Adam(custom_model.parameters(), lr=0.001)
history_custom = train_model(custom_model, optimizer, train_loader, test_loader, epochs=10)
record('Custom CNN (from scratch)', custom_model, history_custom, test_loader)

compare = ['ResNet50 feature extraction', 'Custom CNN (from scratch)', 'VGG16 feature extraction']
plt.figure(figsize=(8, 5))
for name in compare:
    plt.plot(all_histories[name]['val_accuracy'], label=name)
plt.title('Feature Extraction: ResNet50 vs Custom CNN vs VGG16')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid()
plt.show()


# ===== Block 9: Task - Fine-tuning with VGG16 =====
ft_vgg_base = vgg16_base()
freeze(ft_vgg_base)
unfreeze(ft_vgg_base[24:])  # last convolution block of VGG16

model_ft_vgg = TransferModel(ft_vgg_base, 512).to(device)
print('Trainable parameters:', count_trainable(model_ft_vgg))

optimizer = optim.Adam(trainable_params(model_ft_vgg), lr=1e-4)
history_ft_vgg = train_model(model_ft_vgg, optimizer, train_loader, test_loader, epochs=10)
record('VGG16 fine-tuning', model_ft_vgg, history_ft_vgg, test_loader)

plot_history(history_ft_vgg, 'VGG16 Fine-Tuning')

plt.figure(figsize=(8, 5))
for name in ['VGG16 feature extraction', 'VGG16 fine-tuning']:
    plt.plot(all_histories[name]['val_accuracy'], label=name)
plt.title('VGG16: Feature Extraction vs Fine-Tuning')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid()
plt.show()


# ===== Block 10: Summary of all models =====
print(f"{'Model':34} {'Trainable params':>17} {'Test acc':>9} {'Epochs':>7} {'Time (s)':>9}")
for name, params, acc, epochs, t in results:
    print(f"{name:34} {params:17,} {acc:9.4f} {epochs:7} {t:9.1f}")

names = [r[0] for r in results]
accs = [r[2] * 100 for r in results]
plt.figure(figsize=(10, 5))
plt.barh(names[::-1], accs[::-1])
for i, acc in enumerate(accs[::-1]):
    plt.text(acc + 0.5, i, f'{acc:.1f}%', va='center')
plt.xlim(0, 100)
plt.xlabel('Test Accuracy (%)')
plt.title('Test Accuracy of All Models')
plt.tight_layout()
plt.show()
