import torch
import torch.nn as nn
import torchvision.models as models

vgg_mean = torch.tensor([123.68, 116.779, 103.939]).float()
vgg_std = torch.tensor([0.229, 0.224, 0.225]).float()

if torch.cuda.is_available():
    vgg_mean = vgg_mean.cuda()
    vgg_std = vgg_std.cuda()


class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.vgg = self.get_vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()
        self.vgg.eval()

        self.mean = vgg_mean.view(-1, 1, 1)
        self.std = vgg_std.view(-1, 1, 1)

    def forward(self, x):
        return self.vgg(self.normalize_vgg(x))

    @staticmethod
    def get_vgg19(last_layer='conv4_4'):
        vgg = models.vgg19(pretrained=torch.cuda.is_available()).features
        model_list = []

        i = 0
        j = 1
        for layer in vgg.children():
            if isinstance(layer, nn.MaxPool2d):
                i = 0
                j += 1

            elif isinstance(layer, nn.Conv2d):
                i += 1

            name = f'conv{j}_{i}'

            if name == last_layer:
                model_list.append(layer)
                break

            model_list.append(layer)

        model = nn.Sequential(*model_list)
        return model

    def normalize_vgg(self, image):
        '''
        Expect input in range -1 1
        '''
        image = ((image + 1.0) / 2.0) * 255
        return image - self.mean

# class Vgg(nn.Module):
#     def __init__(self):
#         super(Vgg, self).__init__()
#         features = models.vgg19(pretrained=True).features.cuda()
#         self.relu1_1 = nn.Sequential()
#         self.relu1_2 = nn.Sequential()
#
#         self.relu2_1 = nn.Sequential()
#         self.relu2_2 = nn.Sequential()
#
#         self.relu3_1 = nn.Sequential()
#         self.relu3_2 = nn.Sequential()
#         self.relu3_3 = nn.Sequential()
#         self.relu3_4 = nn.Sequential()
#
#         self.relu4_1 = nn.Sequential()
#         self.relu4_2 = nn.Sequential()
#         self.relu4_3 = nn.Sequential()
#         self.relu4_4 = nn.Sequential()
#
#         self.relu5_1 = nn.Sequential()
#         self.relu5_2 = nn.Sequential()
#         self.relu5_3 = nn.Sequential()
#         self.relu5_4 = nn.Sequential()
#
#         for x in range(2):
#             self.relu1_1.add_module(str(x), features[x])
#
#         for x in range(2, 4):
#             self.relu1_2.add_module(str(x), features[x])
#
#         for x in range(4, 7):
#             self.relu2_1.add_module(str(x), features[x])
#
#         for x in range(7, 9):
#             self.relu2_2.add_module(str(x), features[x])
#
#         for x in range(9, 12):
#             self.relu3_1.add_module(str(x), features[x])
#
#         for x in range(12, 14):
#             self.relu3_2.add_module(str(x), features[x])
#
#         for x in range(14, 16):
#             self.relu3_2.add_module(str(x), features[x])
#
#         for x in range(16, 18):
#             self.relu3_4.add_module(str(x), features[x])
#
#         for x in range(18, 21):
#             self.relu4_1.add_module(str(x), features[x])
#
#         for x in range(21, 23):
#             self.relu4_2.add_module(str(x), features[x])
#
#         for x in range(23, 25):
#             self.relu4_3.add_module(str(x), features[x])
#
#         for x in range(25, 27):
#             self.relu4_4.add_module(str(x), features[x])
#
#         for x in range(27, 30):
#             self.relu5_1.add_module(str(x), features[x])
#
#         for x in range(30, 32):
#             self.relu5_2.add_module(str(x), features[x])
#
#         for x in range(32, 34):
#             self.relu5_3.add_module(str(x), features[x])
#
#         for x in range(34, 36):
#             self.relu5_4.add_module(str(x), features[x])
#
#         # don't need the gradients, just want the features
#         for param in self.parameters():
#             param.requires_grad = False
#
#     def forward(self, x):
#         relu1_1 = self.relu1_1(x)
#         relu1_2 = self.relu1_2(relu1_1)
#
#         relu2_1 = self.relu2_1(relu1_2)
#         relu2_2 = self.relu2_2(relu2_1)
#
#         relu3_1 = self.relu3_1(relu2_2)
#         relu3_2 = self.relu3_2(relu3_1)
#         relu3_3 = self.relu3_3(relu3_2)
#         relu3_4 = self.relu3_4(relu3_3)
#
#         relu4_1 = self.relu4_1(relu3_4)
#         relu4_2 = self.relu4_2(relu4_1)
#         relu4_3 = self.relu4_3(relu4_2)
#         relu4_4 = self.relu4_4(relu4_3)
#
#         relu5_1 = self.relu5_1(relu4_4)
#         relu5_2 = self.relu5_2(relu5_1)
#         relu5_3 = self.relu5_3(relu5_2)
#         relu5_4 = self.relu5_4(relu5_3)
#
#         out = {
#             'relu1_1': relu1_1,
#             'relu1_2': relu1_2,
#
#             'relu2_1': relu2_1,
#             'relu2_2': relu2_2,
#
#             'relu3_1': relu3_1,
#             'relu3_2': relu3_2,
#             'relu3_3': relu3_3,
#             'relu3_4': relu3_4,
#
#             'relu4_1': relu4_1,
#             'relu4_2': relu4_2,
#             'relu4_3': relu4_3,
#             'relu4_4': relu4_4,
#
#             'relu5_1': relu5_1,
#             'relu5_2': relu5_2,
#             'relu5_3': relu5_3,
#             'relu5_4': relu5_4,
#         }
#         return out
#
