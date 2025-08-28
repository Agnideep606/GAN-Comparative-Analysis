import os
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
from math import ceil

from models.vanilla_gan import VanillaGenerator

# Settings
GAN_NAME = "vanilla_gan"
DATASET = "mnist"   # used to determine image shape if needed
CHECKPOINT = f"checkpoints/{GAN_NAME}/generator.pth"
OUT_ROOT = f"results/sample_images/{GAN_NAME}"
LATENT_DIM = 100
NUM_CLASSES = 10
TOTAL_PER_CLASS = 4000
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(t):
    return (t + 1.0) / 2.0

def main():
    # instantiate generator (choose image shape consistent with your training)
    img_shape = (1, 28, 28) if DATASET in ["mnist", "fmnist"] else (3, 32, 32)
    gen = VanillaGenerator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, img_shape=img_shape).to(device)
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"{CHECKPOINT} not found")
    gen.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    gen.eval()

    os.makedirs(OUT_ROOT, exist_ok=True)
    for label in range(NUM_CLASSES):
        out_dir = os.path.join(OUT_ROOT, str(label))
        os.makedirs(out_dir, exist_ok=True)
        imgs_saved = 0
        steps = ceil(TOTAL_PER_CLASS / BATCH_SIZE)
        for step in range(steps):
            cur_batch = min(BATCH_SIZE, TOTAL_PER_CLASS - imgs_saved)
            z = torch.randn(cur_batch, LATENT_DIM, device=device)
            labels = torch.full((cur_batch,), label, dtype=torch.long, device=device)
            labels_onehot = F.one_hot(labels, NUM_CLASSES).float()
            with torch.no_grad():
                out = gen(z, labels_onehot.to(device))
                out = denormalize(out.clamp(-1, 1))
                for i in range(out.size(0)):
                    idx = imgs_saved + i + 1
                    save_image(out[i], os.path.join(out_dir, f"img_{idx:05d}.png"))
            imgs_saved += cur_batch
            if imgs_saved >= TOTAL_PER_CLASS:
                break
        print(f"Saved {imgs_saved} images for class {label} at {out_dir}")

if __name__ == "__main__":
    main()
