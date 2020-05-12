import math
import numbers
import torch
import torch.nn as nn
from torch.nn import functional as F


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently
                               supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with
        BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.Tensor([target_real_label]))
        self.register_buffer('fake_label', torch.Tensor([target_fake_label]))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a
                                    discriminator
            target_is_real (bool) - - if the ground truth label is for real
                                      images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of
            the input
        """
        if isinstance(target_is_real, bool):
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            return target_tensor.expand_as(prediction)
        elif isinstance(target_is_real, torch.Tensor):
            res = []
            for p, t in zip(prediction, target_is_real):
                res.append(t.expand_as(p))
            return torch.stack(res)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a
                                    discriminator
            target_is_real (bool) - - if the ground truth label is for real
                                      images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors.
            Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (1 / (std * math.sqrt(2 * math.pi)) *
                       torch.exp(-((mgrid - mean) / std) ** 2 / 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only <= 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class PatchLoss(nn.Module):
    def __init__(self, channels=3, loss_type="L1", scale=[0.05, 0.75, 0.2]):
        super(PatchLoss, self).__init__()
        self.gaussian_w_3 = GaussianSmoothing(channels, 3, 1)
        self.gaussian_w_5 = GaussianSmoothing(channels, 5, 1)
        if loss_type == "L1":
            self.loss = nn.L1Loss()
        elif loss_type == "L2":
            self.loss = nn.MSELoss()
        else:
            raise RuntimeError(
                'Only L1 and L2 are supported. Received {}.'.format(loss_type)
            )
        if len(scale) != 3:
            raise RuntimeError(
                'Needs 3 scales for patch loss. Received {}.'.format(loss_type)
            )
        self.scale = scale

    def __call__(self, y_hat, y):
        loss = self.loss(y_hat, y) * self.scale[0]
        y_hat_gau_3 = self.gaussian_w_3(y_hat)
        y_gau_3 = self.gaussian_w_3(y)
        loss += self.loss(y_hat_gau_3, y_gau_3) * self.scale[1]
        y_hat_gau_5 = self.gaussian_w_5(y_hat)
        y_gau_5 = self.gaussian_w_5(y)
        loss += self.loss(y_hat_gau_5, y_gau_5) * self.scale[2]
        return loss


if __name__ == "__main__":
    patch_loss = PatchLoss()
    y_hat = torch.randn((3, 128, 128), requires_grad=True)
    y = torch.randn((3, 128, 128), requires_grad=True)
    loss = patch_loss(y_hat.unsqueeze(0), y.unsqueeze(0))
    loss.backward()  # check if can do backpropagation without mistake
    print(loss.item())
