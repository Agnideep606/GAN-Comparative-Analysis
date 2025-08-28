import os
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
from math import ceil

from models.dcgan import DCGANGenerator

GAN_NAME = "dcgan"
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
    # channel selection: for mnist we'll expand to 1 channel generator will produce 1-ch images; train accordingly
    gen = DCGANGenerator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, img_channels=3).to(device)
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
            z = torch.randn(cur_batch, LATENT_DIM, 1, 1, device=device)
            # DCGAN generator expects z shaped (N, latent_dim,1,1); labels_onehot we'll pass as vector by flattening
            # We adapt by flattening label one-hot and repeating to match conv input via generator's FC expectation.
            labels = torch.full((cur_batch,), label, dtype=torch.long, device=device)
            labels_onehot = F.one_hot(labels, NUM_CLASSES).float()  # (N, num_classes)
            # many DCGAN implementations expect gen(z, labels_onehot) to accept (N,latent,1,1) and (N,num_classes)
            # our generator concatenates label to z before FC, so we need to reshape z to (N, latent_dim) first
            z_flat = z.view(cur_batch, LATENT_DIM)
            with torch.no_grad():
                out = gen(z_flat, labels_onehot.to(device))
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
