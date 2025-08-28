import os
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
from math import ceil

from models.infogan import InfoGANGenerator

GAN_NAME = "infogan"
CHECKPOINT = f"checkpoints/{GAN_NAME}/generator.pth"
OUT_ROOT = f"results/sample_images/{GAN_NAME}"
NOISE_DIM = 62
CATEGORICAL_DIM = 10
CONTINUOUS_DIM = 2
TOTAL_PER_CLASS = 4000
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(t):
    return (t + 1.0) / 2.0

def main():
    gen = InfoGANGenerator(latent_dim=NOISE_DIM, code_dim=CONTINUOUS_DIM, img_shape=(1,28,28)).to(device)
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"{CHECKPOINT} not found")
    gen.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    gen.eval()
    os.makedirs(OUT_ROOT, exist_ok=True)

    for label in range(CATEGORICAL_DIM):
        out_dir = os.path.join(OUT_ROOT, str(label))
        os.makedirs(out_dir, exist_ok=True)
        imgs_saved = 0
        steps = ceil(TOTAL_PER_CLASS / BATCH_SIZE)
        for step in range(steps):
            cur_batch = min(BATCH_SIZE, TOTAL_PER_CLASS - imgs_saved)
            z = torch.randn(cur_batch, NOISE_DIM, device=device)
            cat = torch.full((cur_batch,), label, dtype=torch.long, device=device)
            cat_onehot = F.one_hot(cat, CATEGORICAL_DIM).float()
            cont = torch.rand(cur_batch, CONTINUOUS_DIM, device=device) * 2 - 1
            # many InfoGAN gens expect either gen(z, c) or gen(concat)
            try:
                with torch.no_grad():
                    imgs = gen(z, torch.cat([cat_onehot, cont], dim=1))
            except TypeError:
                latent = torch.cat([z, cat_onehot, cont], dim=1)
                with torch.no_grad():
                    imgs = gen(latent)
            imgs = denormalize(imgs.clamp(-1,1))
            for i in range(imgs.size(0)):
                idx = imgs_saved + i + 1
                save_image(imgs[i], os.path.join(out_dir, f"img_{idx:05d}.png"))
            imgs_saved += cur_batch
            if imgs_saved >= TOTAL_PER_CLASS:
                break
        print(f"Saved {imgs_saved} images for class {label} at {out_dir}")

if __name__ == "__main__":
    main()
