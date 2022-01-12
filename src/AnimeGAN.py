import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import time

from src.networks import Generator, Discriminator
from src.losses import GeneratorLoss, DiscriminatorLoss, ContentLoss, GrayscaleLoss, ColorLoss, LossLogger, TotalVariationLoss
from utils import denormalize_input

gaussian_mean = torch.tensor(0.0)
gaussian_std = torch.tensor(0.1)


class AnimeGAN(nn.Module):
    def __init__(self, args, data_loader, val_loader):
        super(AnimeGAN, self).__init__()

        print()
        print("##### Information #####")
        print("# gan type : ", args.gan_type)
        print("# dataset : ", args.dataset)
        print("# batch_size : ", args.batch_size)
        print("# epochs : ", args.epochs)
        print("# init_epochs : ", args.init_epochs)
        print("# wadvg, wadvd, wadvd_real, wadvd_gray, wadvd_fake, wadvd_smooth, wcon, wgray, wcol, wtv: ",
              args.wadvg, args.wadvd, args.wadvd_real, args.wadvd_gray, args.wadvd_fake, args.wadvd_smooth,
              args.wcon, args.wgray, args.wcol, args.wtv)
        print("# lr_init, lr_g, lr_d : ", args.lr_init, args.lr_g, args.lr_d)
        print(f"# training_rate G -- D: {args.training_rate} : 1")
        print()

        self.args = args
        self.data_loader = data_loader
        self.val_loader = val_loader

        self.dataset = args.dataset
        self.epoch = 0
        self.training_rate = args.training_rate
        self.save_freq = args.save_freq

        self.gen_weights_path = os.path.join(args.checkpoint_dir, 'gen')
        self.dis_weights_path = os.path.join(args.checkpoint_dir, 'dis')

        if not os.path.exists(self.gen_weights_path):
            os.makedirs(self.gen_weights_path)

        if not os.path.exists(self.dis_weights_path):
            os.makedirs(self.dis_weights_path)

        generator = Generator()
        discriminator = Discriminator(args.ch, args.n_layers, args.sn)

        if torch.cuda.is_available():
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            if len(os.getenv('CUDA_VISIBLE_DEVICES').split(',')) > 1:
                print(' Using %d GPU(s)' % len(os.getenv('CUDA_VISIBLE_DEVICES').split(',')))
                generator = nn.DataParallel(generator)
                discriminator = nn.DataParallel(discriminator)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.gen_optimizer = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
        self.dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

        loss_logger = LossLogger()
        loss_content = ContentLoss()
        loss_grayscale = GrayscaleLoss()
        loss_color = ColorLoss()
        loss_total_variation = TotalVariationLoss()
        loss_generator = GeneratorLoss(args, loss_content, loss_grayscale, loss_color, loss_total_variation, loss_logger)
        loss_discriminator = DiscriminatorLoss(args, loss_logger)

        self.loss_logger = loss_logger

        self.add_module('loss_content', loss_content)
        self.add_module('loss_generator', loss_generator)
        self.add_module('loss_discriminator', loss_discriminator)

    def process(self):
        self.generator.train()

        self.loss_logger.reset()

        max_iter = len(self.data_loader)

        epoch_init = time.time()

        if self.epoch < self.args.init_epochs:
            # Train with content loss only
            set_lr(self.gen_optimizer, self.args.lr_init)
            for index, images in enumerate(self.data_loader):
                start = time.time()
                img = images[0]
                if torch.cuda.is_available():
                    img = img.cuda()

                self.gen_optimizer.zero_grad()

                fake_img = self.generator(img)
                loss = self.loss_content(img, fake_img)
                loss.backward()
                self.gen_optimizer.step()

                self.loss_logger.update_loss_content(loss)
                avg_content_loss = self.loss_logger.avg_loss_init()
                print(
                    f'[Init Training G] Epoch: {self.epoch:3d} Iteration: {index + 1}/{max_iter} content loss: {avg_content_loss:2f} time: {time.time() - start:.2f}')

                if time.time() - epoch_init > 10800:
                    self.save(index)
                    self.epoch += 1
                    return

            set_lr(self.gen_optimizer, self.args.lr_g)
            if self.epoch % self.save_freq == 0:
                self.save(max_iter)
            self.epoch += 1
            return

        j = self.training_rate
        for index, images in enumerate(self.data_loader):
            # To cuda
            start = time.time()
            img, anime, anime_gray, anime_smt_gray = images
            if torch.cuda.is_available():
                img = img.cuda()
                anime = anime.cuda()
                anime_gray = anime_gray.cuda()
                anime_smt_gray = anime_smt_gray.cuda()

            if j == self.training_rate:
                # Train discriminator
                self.discriminator.train()
                self.dis_optimizer.zero_grad()
                fake_img = self.generator(img).detach()

                # Add some Gaussian noise to images before feeding to D
                if self.args.d_noise:
                    fake_img += gaussian_noise()
                    anime += gaussian_noise()
                    anime_gray += gaussian_noise()
                    anime_smt_gray += gaussian_noise()

                fake_d = self.discriminator(fake_img)
                real_anime_d = self.discriminator(anime)
                real_anime_gray_d = self.discriminator(anime_gray)
                real_anime_smt_gray_d = self.discriminator(anime_smt_gray)

                loss_d = self.loss_discriminator(
                    fake_d, real_anime_d, real_anime_gray_d, real_anime_smt_gray_d)

                loss_d.backward()
                self.dis_optimizer.step()

            self.discriminator.eval()

            # ---------------- TRAIN G ---------------- #
            self.gen_optimizer.zero_grad()

            fake_img = self.generator(img)
            fake_d = self.discriminator(fake_img)

            loss_g = self.loss_generator(fake_img, img, fake_d, anime_gray)

            loss_g.backward()
            self.gen_optimizer.step()

            avg_adv, avg_gram, avg_color, avg_content, avg_tv = self.loss_logger.avg_loss_G()

            if j == self.training_rate:
                avg_adv_d = self.loss_logger.avg_loss_D()
                print(
                    f'Epoch: {self.epoch:3d} Iteration: {index + 1}/{max_iter} loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} total variation {avg_tv:2f} / loss D: {avg_adv_d:2f} time: {time.time() - start:.2f}')
            else:
                print(
                    f'Epoch: {self.epoch:3d} Iteration: {index + 1}/{max_iter} loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} total variation {avg_tv:2f} time: {time.time() - start:.2f}')

            j = j - 1
            if j < 1:
                j = self.training_rate

            if time.time() - epoch_init > 18000:
                self.save(index)
                epoch_init = time.time()

        if self.epoch % self.save_freq == 0:
            self.save(max_iter)
        self.epoch += 1

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.dataset)
            files = [os.path.join(self.gen_weights_path, f) for f in os.listdir(self.gen_weights_path) if
                     f.endswith('.pth')]
            file = max(files, key=lambda f: int(f.split('_')[-3]) * 1000000 + int(f.split('_')[-2]))

            if torch.cuda.is_available():
                data = torch.load(file)
            else:
                data = torch.load(file, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.epoch = data['epoch'] + 1
            print("Generator loaded")

        # load discriminator only when training
        if os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.dataset)
            files = [os.path.join(self.dis_weights_path, f) for f in os.listdir(self.dis_weights_path) if
                     f.endswith('.pth')]
            file = max(files, key=lambda f: int(f.split('_')[-2]))

            if torch.cuda.is_available():
                data = torch.load(file)
            else:
                data = torch.load(file, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])
            self.dis_optimizer.load_state_dict(data['optimizer'])
            print("Discriminator loaded")

    def save(self, iter=None):
        if not iter:
            print(f'* Saving checkpoint at epoch {self.epoch}')
            torch.save({
                'epoch': self.epoch,
                'generator': self.generator.state_dict(),
                'optimizer': self.gen_optimizer.state_dict(),
            }, os.path.join(self.gen_weights_path, f'{self.dataset}_{self.epoch}_gen.pth'))
            print('Generator saved on %s' % self.gen_weights_path)

            torch.save({
                'discriminator': self.discriminator.state_dict(),
                'optimizer': self.dis_optimizer.state_dict()
            }, os.path.join(self.dis_weights_path, f'{self.dataset}_{self.epoch}_dis.pth'))
            print('Discriminator saved on %s' % self.dis_weights_path)
        else:
            print(f'* Saving checkpoint at epoch {self.epoch}')
            torch.save({
                'epoch': self.epoch,
                'generator': self.generator.state_dict(),
                'optimizer': self.gen_optimizer.state_dict(),
            }, os.path.join(self.gen_weights_path, f'{self.dataset}_{self.epoch}_{iter}_gen.pth'))
            print('Generator saved on %s' % self.gen_weights_path)

            torch.save({
                'discriminator': self.discriminator.state_dict(),
                'optimizer': self.dis_optimizer.state_dict()
            }, os.path.join(self.dis_weights_path, f'{self.dataset}_{self.epoch}_{iter}_dis.pth'))
            print('Discriminator saved on %s' % self.dis_weights_path)

        save_samples(self.generator, self.val_loader, self.epoch, self.args, iter)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def gaussian_noise():
    return torch.normal(gaussian_mean, gaussian_std)


def save_samples(generator, loader, epoch, args, iter=None):
    '''
    Generate and save images
    '''
    generator.eval()

    max_iter = len(loader)

    fake_imgs = []

    for index, img in enumerate(loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                img = img.cuda()
            fake_img = generator(img)
            fake_img = fake_img.detach().cpu().numpy()
            # Channel first -> channel last
            fake_img = fake_img.transpose(0, 2, 3, 1).squeeze(0)
            fake_imgs.append(denormalize_input(fake_img, dtype=np.int16))
            print(f'Val: {index + 1}/{max_iter}')

        if index + 1 == max_iter:
            break

    if iter:
        save_path = os.path.join(args.save_image_dir, args.dataset, f'{epoch}_{iter}')
    else:
        save_path = os.path.join(args.save_image_dir, args.dataset, f'{epoch}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, img in enumerate(fake_imgs):
        save_file = os.path.join(save_path, f'{i:03d}.png')
        print(f'* Saving {save_file}')
        cv2.imwrite(save_file, cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB))
