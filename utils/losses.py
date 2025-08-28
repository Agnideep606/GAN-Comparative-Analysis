import torch
import torch.nn.functional as F

# Standard GAN loss
def gan_discriminator_loss(real_preds, fake_preds):
    real_loss = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
    fake_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss

def gan_generator_loss(fake_preds):
    return F.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))

# Wasserstein GAN loss
def wgan_discriminator_loss(real_preds, fake_preds):
    return -(torch.mean(real_preds) - torch.mean(fake_preds))

def wgan_generator_loss(fake_preds):
    return -torch.mean(fake_preds)

# Gradient penalty for WGAN-GP
def gradient_penalty(critic, real_data, fake_data, device="cuda"):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated = interpolated.to(device)
    interpolated_scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# InfoGAN loss (simplified: GAN + mutual information)
def infogan_generator_loss(fake_preds, q_loss):
    return F.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds)) + q_loss

def infogan_discriminator_loss(real_preds, fake_preds):
    real_loss = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
    fake_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss
