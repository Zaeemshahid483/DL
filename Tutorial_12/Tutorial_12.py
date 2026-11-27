# ===== Block 1: Import libraries =====
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
print('Device:', device)


# ===== Block 2: Load data set =====
# Load the dataset (MNIST)
train_set = datasets.MNIST(root='./data', train=True, download=True)
test_set = datasets.MNIST(root='./data', train=False, download=True)
x_train = train_set.data.float() / 255.
x_test = test_set.data.float() / 255.

# Add an extra dimension for the channel (since MNIST is grayscale).
# PyTorch puts the channel first: (N, 1, 28, 28) instead of Keras' (N, 28, 28, 1)
x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)

print('x_train:', tuple(x_train.shape), '| x_test:', tuple(x_test.shape))
print('Pixel values from', x_train.min().item(), 'to', x_train.max().item())


# ===== Block 3: Define the autoencoder model =====
class Autoencoder(nn.Module):
    # The tutorial's convolutional autoencoder
    def __init__(self):
        super().__init__()
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                            # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)                             # 14x14 -> 7x7: encoded = 64 x 7 x 7
        )
        # Decoder part
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),                # 7x7 -> 14x14 (Keras UpSampling2D)
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),                # 14x14 -> 28x28
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def summary(model):
    # Output shape and parameters of every layer, like Keras' model.summary()
    rows, hooks = [], []

    def hook(module, inputs, output):
        rows.append((module.__class__.__name__, tuple(output.shape), sum(p.numel() for p in module.parameters())))

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Upsample, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    print(f"{'Layer':16} {'Output shape':>18} {'Params':>10}")
    for name, shape, params in rows:
        print(f"{name:16} {str(shape):>18} {params:10,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


def latent_size(model):
    with torch.no_grad():
        model.eval()
        return model.encoder(torch.zeros(1, 1, 28, 28, device=next(model.parameters()).device)).numel()


autoencoder = Autoencoder().to(device)
summary(autoencoder)
print(f'Latent representation: {latent_size(autoencoder):,} numbers for a 28x28 = 784 pixel image')


# ===== Block 4: Train the autoencoder =====
def evaluate(model, x, batch_size=1000):
    # Binary cross-entropy (the training loss), mean squared error and PSNR of the reconstructions
    model.eval()
    bce, mse, psnr = 0.0, 0.0, 0.0
    with torch.no_grad():
        for k in range(0, len(x), batch_size):
            xb = x[k:k + batch_size].to(device)
            out = model(xb)
            bce += F.binary_cross_entropy(out, xb, reduction='sum').item()
            per_image = ((out - xb) ** 2).flatten(1).mean(1)
            mse += per_image.sum().item()
            psnr += (10 * torch.log10(1 / per_image.clamp(min=1e-10))).sum().item()
    n = len(x)
    return {'bce': bce / (n * 784), 'mse': mse / n, 'psnr': psnr / n}


def train(model, epochs=50, batch_size=128, lr=1e-3, print_every=1):
    # Adam with the Keras default learning rate (0.001) and binary cross-entropy, as in the tutorial
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    xtr = x_train.to(device)
    history = {'loss': [], 'val_loss': []}
    start = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xtr), device=device)  # shuffle=True
        total = 0.0
        for k in range(0, len(xtr), batch_size):
            xb = xtr[order[k:k + batch_size]]
            loss = F.binary_cross_entropy(model(xb), xb)  # the input is also the target
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        val = evaluate(model, x_test)
        history['loss'].append(total / len(xtr))
        history['val_loss'].append(val['bce'])
        if epoch == 0 or (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {history['loss'][-1]:.4f} - val_loss: {val['bce']:.4f}")
    history['time'] = time.time() - start
    return history


# Train the autoencoder
history = train(autoencoder, epochs=50, batch_size=128)
print(f"Training time: {history['time']:.1f} s")


# ===== Block 5: Visualize the results =====
def predict(model, x, batch_size=1000):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[k:k + batch_size].to(device)).cpu() for k in range(0, len(x), batch_size)])


scores = evaluate(autoencoder, x_test)
print(f"Test loss (binary cross-entropy): {scores['bce']:.4f} | MSE: {scores['mse']:.5f} | PSNR: {scores['psnr']:.2f} dB")

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(history['loss']) + 1), history['loss'], label='Training loss')
plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Validation loss')
plt.title('Tutorial autoencoder: loss')
plt.xlabel('Epoch')
plt.ylabel('Binary cross-entropy')
plt.legend()
plt.grid()
plt.show()

decoded_imgs = predict(autoencoder, x_test[:10])

# Visualize the original and reconstructed images side by side
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    ax.axis("off")
    plt.title("Original")

    # Display reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    ax.axis("off")
    plt.title("Reconstructed")

plt.show()


# ===== Block 6: Task - Define different autoencoder architectures =====
class DenseAE(nn.Module):
    # Basic fully connected autoencoder: 784 -> 256 -> 128 -> latent -> 128 -> 256 -> 784
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ConvDenseAE(nn.Module):
    # The design described in the tutorial's text: conv layers, a dense latent layer, transposed conv layers
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 14 -> 28
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def conv_bn(in_ch, out_ch):
    return [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU()]


def up_bn(in_ch, out_ch):
    # Transposed convolution that doubles the size of the feature map
    return [nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU()]


class DeepConvAE(nn.Module):
    # More layers (two convolutions per level), batch normalization and transposed convolutions.
    # latent_dim=None keeps the 64 x 7 x 7 feature map as the latent representation (like the tutorial);
    # a number adds a dense bottleneck of that size.
    def __init__(self, latent_dim=None):
        super().__init__()
        encoder = [*conv_bn(1, 32), *conv_bn(32, 32), nn.MaxPool2d(2),
                   *conv_bn(32, 64), *conv_bn(64, 64), nn.MaxPool2d(2)]
        decoder = []
        if latent_dim:
            encoder += [nn.Flatten(), nn.Linear(64 * 7 * 7, latent_dim)]
            decoder += [nn.Linear(latent_dim, 64 * 7 * 7), nn.ReLU(), nn.Unflatten(1, (64, 7, 7))]
        decoder += [*up_bn(64, 64), *conv_bn(64, 64),   # 7 -> 14
                    *up_bn(64, 32), *conv_bn(32, 32),   # 14 -> 28
                    nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))


architectures = {
    'Tutorial conv AE': lambda: Autoencoder(),
    'Deeper conv AE + BN': lambda: DeepConvAE(),
    'Dense AE (latent 32)': lambda: DenseAE(32),
    'Conv AE + dense latent 32': lambda: ConvDenseAE(32),
    'Deeper conv AE + BN, latent 32': lambda: DeepConvAE(32),
}
for name, build in architectures.items():
    m = build().to(device)
    out_shape = tuple(m(torch.zeros(2, 1, 28, 28, device=device)).shape)
    print(f"{name:32} latent: {latent_size(m):5,} numbers | parameters: {sum(p.numel() for p in m.parameters()):9,} | output: {out_shape}")


# ===== Block 7: Task - Train and compare the architectures =====
TASK_EPOCHS = 20
task_results, task_histories, task_models = {}, {}, {}


def run(name, build, epochs=TASK_EPOCHS):
    set_seed(42)  # same starting point and batches for every model
    model = build().to(device)
    print(f"\nTraining: {name}")
    h = train(model, epochs=epochs, print_every=5)
    s = evaluate(model, x_test)
    task_results[name] = {'params': sum(p.numel() for p in model.parameters()), 'latent': latent_size(model),
                          'time': h['time'], **s}
    task_histories[name] = h
    task_models[name] = model
    print(f"Test BCE: {s['bce']:.4f} | MSE: {s['mse']:.5f} | PSNR: {s['psnr']:.2f} dB | time: {h['time']:.1f} s")


for name, build in architectures.items():
    run(name, build)

print(f"\n{'Model':32} {'Latent':>7} {'Compression':>12} {'Parameters':>11} {'BCE':>7} {'MSE':>8} {'PSNR (dB)':>10} {'Time (s)':>9}")
for name, r in task_results.items():
    print(f"{name:32} {r['latent']:7,} {784 / r['latent']:11.1f}x {r['params']:11,} {r['bce']:7.4f} {r['mse']:8.5f} {r['psnr']:10.2f} {r['time']:9.1f}")
print(f'(test set, {TASK_EPOCHS} epochs each; compression = 784 pixels / latent size, below 1 means no compression)')


# ===== Block 8: Task - Visualize the reconstructions =====
plt.figure(figsize=(9, 5))
for name, h in task_histories.items():
    plt.plot(range(1, TASK_EPOCHS + 1), h['val_loss'], label=name)
plt.yscale('log')
plt.title('Validation loss of the different architectures')
plt.xlabel('Epoch')
plt.ylabel('Binary cross-entropy (log scale)')
plt.legend()
plt.grid()
plt.show()

idx = list(range(10))
names = list(task_models)
fig, axes = plt.subplots(len(names) + 1, len(idx), figsize=(16, 1.8 * (len(names) + 1)))
recon = {name: predict(task_models[name], x_test[idx]) for name in names}
for c, i in enumerate(idx):
    axes[0, c].imshow(x_test[i, 0], cmap='gray')
    for r, name in enumerate(names):
        axes[r + 1, c].imshow(recon[name][c, 0], cmap='gray', vmin=0, vmax=1)
for r, label in enumerate(['Original'] + [f"{n} ({task_results[n]['psnr']:.1f} dB)" for n in names]):
    for c in range(len(idx)):
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    axes[r, 0].set_ylabel(label, rotation=0, ha='right', va='center', fontsize=9)
plt.suptitle('Test digits reconstructed by each architecture')
plt.tight_layout()
plt.show()


# ===== Block 9: Task - Does the model really compress? Reconstructing random noise =====
# A model with a real bottleneck can only rebuild what it has learned (digits).
# A model whose latent representation is larger than the input can copy almost anything.
set_seed(0)
noise = torch.rand(8, 1, 28, 28)
check = ['Tutorial conv AE', 'Deeper conv AE + BN', 'Deeper conv AE + BN, latent 32']
fig, axes = plt.subplots(len(check) + 1, 8, figsize=(14, 1.9 * (len(check) + 1)))
for c in range(8):
    axes[0, c].imshow(noise[c, 0], cmap='gray', vmin=0, vmax=1)
print(f"{'Model':32} {'MSE on digits':>14} {'MSE on noise':>13}")
for r, name in enumerate(check):
    out = predict(task_models[name], noise)
    for c in range(8):
        axes[r + 1, c].imshow(out[c, 0], cmap='gray', vmin=0, vmax=1)
    print(f"{name:32} {task_results[name]['mse']:14.5f} {((out - noise) ** 2).mean().item():13.5f}")
for r, label in enumerate(['Random noise'] + check):
    for c in range(8):
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    axes[r, 0].set_ylabel(label, rotation=0, ha='right', va='center', fontsize=9)
plt.suptitle('Random noise images and their reconstructions')
plt.tight_layout()
plt.show()


# ===== Block 10: Task - Effect of the latent size =====
latent_sizes = [2, 8, 32, 128]
for d in latent_sizes:
    name = f'Deeper conv AE + BN, latent {d}'
    if name not in task_results:  # latent 32 was already trained above
        run(name, lambda d=d: DeepConvAE(d))

psnr = [task_results[f'Deeper conv AE + BN, latent {d}']['psnr'] for d in latent_sizes]
print(f"\n{'Latent size':>11} {'Compression':>12} {'MSE':>8} {'PSNR (dB)':>10}")
for d in latent_sizes:
    r = task_results[f'Deeper conv AE + BN, latent {d}']
    print(f"{d:11} {784 / d:11.0f}x {r['mse']:8.5f} {r['psnr']:10.2f}")

plt.figure(figsize=(14, 4.5))
plt.subplot(1, 2, 1)
plt.plot(latent_sizes, psnr, marker='o')
plt.xscale('log', base=2)
plt.xticks(latent_sizes, [str(d) for d in latent_sizes])
plt.title('Reconstruction quality vs latent size')
plt.xlabel('Latent size (numbers)')
plt.ylabel('PSNR (dB)')
plt.grid()

plt.subplot(1, 2, 2)
names = list(task_results)
order = np.argsort([task_results[n]['psnr'] for n in names])
plt.barh([names[i] for i in order], [task_results[names[i]]['psnr'] for i in order])
for k, i in enumerate(order):
    plt.text(task_results[names[i]]['psnr'] + 0.2, k, f"{task_results[names[i]]['psnr']:.1f}", va='center', fontsize=9)
plt.title('PSNR of all autoencoders (higher is better)')
plt.xlabel('PSNR (dB)')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(len(latent_sizes) + 1, 10, figsize=(16, 1.8 * (len(latent_sizes) + 1)))
for c in range(10):
    axes[0, c].imshow(x_test[c, 0], cmap='gray')
for r, d in enumerate(latent_sizes):
    out = predict(task_models[f'Deeper conv AE + BN, latent {d}'], x_test[:10])
    for c in range(10):
        axes[r + 1, c].imshow(out[c, 0], cmap='gray', vmin=0, vmax=1)
for r, label in enumerate(['Original'] + [f'Latent {d}' for d in latent_sizes]):
    for c in range(10):
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    axes[r, 0].set_ylabel(label, rotation=0, ha='right', va='center', fontsize=10)
plt.suptitle('Deeper conv AE + BN: reconstructions with different latent sizes')
plt.tight_layout()
plt.show()
