import torch as t
import torch.nn as nn

class AdversarialLoss:
    def __init__(self):
        self.bce_loss = nn.BCELoss()

    def generator_loss(self, pred_fake):
        # generator tries to fool the discriminator (labels = 1)
        return self.bce_loss(pred_fake, t.ones_like(pred_fake))

    def discriminator_loss(self, pred_fake, pred_real):
        # Discriminator tries to correctly classify real (1) and fake (0)
        loss_real = self.bce_loss(pred_real, t.ones_like(pred_real))
        loss_fake = self.bce_loss(pred_fake, t.zeros_like(pred_fake))
        return (loss_real + loss_fake) / 2