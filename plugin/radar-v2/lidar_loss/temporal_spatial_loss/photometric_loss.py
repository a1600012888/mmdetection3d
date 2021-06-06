import torch
import torch.nn as nn


class PhotometricLoss(nn.Module):
    def __init__(self, alpha=0.1, clip_loss=0.5, C1=1e-4, C2=9e-4):
        """
        Calculate photomertric loss for two list of image

        Parameters
        ----------
        alpha: float
            Coefficient of SSIM, loss = alpha * (1 - SSIM)/2 + (1 - alpha) * L1
        clip_loss: float
            clip photometric loss below (mean + clip_loss*std)
        C1, C2: float
            used in SSIM calculation
        """
        super(PhotometricLoss, self).__init__()
        self.ssim_loss_weight = alpha
        self.clip_loss = clip_loss
        self.C1 = C1
        self.C2 = C2

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)

        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : list of torch.Tensor([6,1,H,W])
            Photometric loss
        """
        assert len(images) == len(t_est)

        list_len = len(images)
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(list_len)]
        # SSIM loss
        ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3) for i in range(list_len)]
        # Weighted Sum: alpha * ssim + (1 - alpha) * l1
        photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                            (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True) for i in range(list_len)]
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(list_len):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def forward(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)

        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : list of torch.Tensor([6,1,H,W])
            Photometric loss
        """
        return self.calc_photometric_loss(t_est, images)


def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim
