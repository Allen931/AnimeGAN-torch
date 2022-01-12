import torch
import torch.nn.functional as F
import torch.nn as nn
from src.vgg19 import Vgg
from utils.image_processing import gram, rgb_to_yuv


class GeneratorLoss(nn.Module):
    def __init__(self, args, content_loss, grayscale_loss, color_loss, tv_loss, loss_logger):
        super(GeneratorLoss, self).__init__()
        self.adv_type = args.gan_type
        self.bce_loss = nn.BCELoss()
        if torch.cuda.is_available():
            self.bce_loss.cuda()
        self.content_loss = content_loss
        self.grayscale_loss = grayscale_loss
        self.color_loss = color_loss
        self.tv_loss = tv_loss

        self.wadvg = args.wadvg
        self.wcon = args.wcon
        self.wgray = args.wgray
        self.wcol = args.wcol
        self.wtv = args.wtv

        self.loss_logger = loss_logger

        self.vgg19 = Vgg()

    def __call__(self, fake_img, img, fake_logit, anime_gray):
        fake_feat = self.vgg19(fake_img)
        anime_feat = self.vgg19(anime_gray)
        img_feat = self.vgg19(img).detach()

        loss_adv = self.wadvg * self.adv_loss_g(fake_logit)
        loss_con = self.wcon * self.content_loss(img_feat, fake_feat)
        loss_gray = self.wgray * self.grayscale_loss(anime_feat, fake_feat)
        loss_color = self.wcol * self.color_loss(img, fake_img)
        loss_tv = self.wtv * self.tv_loss(fake_img)
        self.loss_logger.update_loss_G(loss_adv, loss_con, loss_gray, loss_color, loss_tv)
        return loss_adv + loss_con + loss_gray + loss_color + loss_tv

    def adv_loss_g(self, pred):
        if self.adv_type == 'hinge':
            return -torch.mean(pred)

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')


class DiscriminatorLoss(nn.Module):
    def __init__(self, args, loss_logger):
        super(DiscriminatorLoss, self).__init__()
        self.adv_type = args.gan_type
        self.bce_loss = nn.BCELoss()

        if torch.cuda.is_available():
            self.bce_loss.cuda()

        self.wadvd = args.wadvd

        self.wadvd_real = args.wadvd_real
        self.wadvd_gray = args.wadvd_gray
        self.wadvd_fake = args.wadvd_fake
        self.wadvd_smooth = args.wadvd_smooth

        self.loss_logger = loss_logger

    def __call__(self, real, gray, fake, real_smooth):
        loss = self.wadvd * (self.wadvd_real * self.adv_loss_d_real(real) + self.wadvd_gray * self.adv_loss_d_fake(
            gray) + self.wadvd_fake * self.adv_loss_d_fake(fake) + self.wadvd_smooth * self.adv_loss_d_fake(
            real_smooth))
        self.loss_logger.update_loss_D(loss)
        return loss

    def adv_loss_d_real(self, real):
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 - real))

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(real - 1.0))

        elif self.adv_type == 'normal':
            return self.bce_loss(real, torch.ones_like(real))

        raise ValueError(f'Do not support loss type {self.adv_type}')

    def adv_loss_d_fake(self, fake):
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 + fake))

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(fake))

        elif self.adv_type == 'normal':
            return self.bce_loss(fake, torch.zeros_like(fake))

        raise ValueError(f'Do not support loss type {self.adv_type}')


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

        self.l1 = nn.L1Loss()
        if torch.cuda.is_available():
            self.l1.cuda()

    def __call__(self, feat, re_feat):
        return self.l1(feat, re_feat)


class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.l1 = nn.L1Loss()
        if torch.cuda.is_available():
            self.l1.cuda()

    def __call__(self, real, fake):
        return self.l1(gram(real), gram(fake))


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.HuberLoss()
        if torch.cuda.is_available():
            self.l1.cuda()
            self.huber.cuda()

    def __call__(self, image, image_generated):
        image = rgb_to_yuv(image)
        image_generated = rgb_to_yuv(image_generated)

        # After convert to yuv, both images have channel last

        return (self.l1(image[:, 0, :, :], image_generated[:, 0, :, :]) +
                self.huber(image[:, 1, :, :], image_generated[:, 1, :, :]) +
                self.huber(image[:, 2, :, :], image_generated[:, 2, :, :]))


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.l2 = nn.MSELoss()
        if torch.cuda.is_available():
            self.l2.cuda()

    def __call__(self, inputs):
        """
        A smooth loss in fact. Like the smooth prior in MRF.
        V(y) = || y_{n+1} - y_n ||_2
        """
        return self.l2(inputs[:, :, :-1, ...], inputs[:, :, 1:, ...]) + self.l2(inputs[:, :, :, :-1], inputs[:, :, :, 1:])


class LossLogger:
    def __init__(self):
        self.loss_g_adv = []
        self.loss_d_adv = []
        self.loss_content = []
        self.loss_init = []
        self.loss_color = []
        self.loss_gram = []
        self.loss_tv = []

    def reset(self):
        self.loss_g_adv = []
        self.loss_d_adv = []
        self.loss_content = []
        self.loss_init = []
        self.loss_color = []
        self.loss_gram = []
        self.loss_tv = []

    def update_loss_G(self, adv, gram, color, content, tv):
        self.loss_g_adv.append(adv.cpu().detach().numpy())
        self.loss_gram.append(gram.cpu().detach().numpy())
        self.loss_color.append(color.cpu().detach().numpy())
        self.loss_content.append(content.cpu().detach().numpy())
        self.loss_tv.append(tv.cpu().detach().numpy())

    def update_loss_D(self, loss):
        self.loss_d_adv.append(loss.cpu().detach().numpy())

    def update_loss_content(self, loss):
        self.loss_init.append(loss.cpu().detach().numpy())

    def avg_loss_G(self):
        return (
            self._avg(self.loss_g_adv),
            self._avg(self.loss_gram),
            self._avg(self.loss_color),
            self._avg(self.loss_content),
            self._avg(self.loss_tv)
        )

    def avg_loss_D(self):
        return self._avg(self.loss_d_adv)

    def avg_loss_init(self):
        return self._avg(self.loss_init)

    @staticmethod
    def _avg(losses):
        return sum(losses) / len(losses)
