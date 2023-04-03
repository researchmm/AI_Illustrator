import torch
import clip
import torchvision
import torch.nn.functional as F
from models.stylegan2 import generator_discriminator
from myutils import print_wt


class my_preprocess(torch.nn.Module):
    def __init__(self, in_size=1024):
        super(my_preprocess, self).__init__()
        self.in_size = in_size
        if self.in_size not in [1024, 512, 256, 224]:
            raise ValueError('No such size.')
        if self.in_size != 224:
            avg_kernel_size = in_size // 32
            self.upsample = torch.nn.Upsample(scale_factor=7)
            self.avgpool = torch.nn.AvgPool2d(kernel_size=avg_kernel_size)
            self.normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                              (0.26862954, 0.26130258, 0.27577711))
        else:
            self.normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                              (0.26862954, 0.26130258, 0.27577711))

    def forward(self, img):
        if self.in_size != 224:
            return self.normalize(self.avgpool(self.upsample(img)))
        else:
            return self.normalize(img)


class total_loss(torch.nn.Module):
    def __init__(self, pixel_weight=0.3, recons_weight=1, reg_weight=0.3, DataKind='ffhq'):
        super(total_loss, self).__init__()

        # 待使用的StyleGAN部分
        if DataKind == 'ffhq':
            self.stylegan = generator_discriminator.StyleGANv2Generator(out_size=1024, style_channels=512, bgr2rgb=True)
            StyleGAN_total_checkpoint = torch.utils.model_zoo.load_url('http://download.openmmlab.com/mmgen/stylegan2'
                                                                       '/official_weights/stylegan2-ffhq-config-f'
                                                                       '-official_20210327_171224-bce9310c.pth',
                                                                       map_location=torch.device('cpu'))
        elif DataKind == 'church':
            self.stylegan = generator_discriminator.StyleGANv2Generator(out_size=256, style_channels=512, bgr2rgb=True)
            StyleGAN_total_checkpoint = torch.utils.model_zoo.load_url('http://download.openmmlab.com/mmgen/stylegan2'
                                                                       '/official_weights/stylegan2-church-config-f'
                                                                       '-official_20210327_172657-1d42b7d1.pth',
                                                                       map_location=torch.device('cpu'))
        elif DataKind == 'cat':
            self.stylegan = generator_discriminator.StyleGANv2Generator(out_size=256, style_channels=512, bgr2rgb=True)
            StyleGAN_total_checkpoint = torch.utils.model_zoo.load_url('http://download.openmmlab.com/mmgen/stylegan2'
                                                                       '/official_weights/stylegan2-cat-config-f'
                                                                       '-official_20210327_172444-15bc485b.pth',
                                                                       map_location=torch.device('cpu'))
        else:
            raise ValueError('No such kind of data')
        StyleGAN_total_state_dict = StyleGAN_total_checkpoint['state_dict']
        modified_state_dict = {}
        pre_fix = 'generator_ema'
        for k, v in StyleGAN_total_state_dict.items():
            if k[0:len(pre_fix)] != pre_fix:
                continue
            modified_state_dict[k[len(pre_fix) + 1:]] = v
        self.stylegan.load_state_dict(modified_state_dict)
        self.stylegan.eval()
        self.stylegan.cuda()
        for param in self.stylegan.parameters():
            param.requires_grad = False

        # 待使用的CLIP部分
        self.clip, _ = clip.load('ViT-B/32', device='cuda')
        self.clip.eval()
        self.clip.cuda()
        if DataKind == 'church':
            self.preprocess = my_preprocess(in_size=256)
        elif DataKind == 'ffhq':
            self.preprocess = my_preprocess(in_size=1024)
        elif DataKind == 'cat':
            self.preprocess = my_preprocess(in_size=256)
        else:
            raise ValueError('No such kind of data')
        self.face_component_resize = torchvision.transforms.Resize(
            size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.face_component_preprocess = my_preprocess(in_size=224)

        # latent的pixel级loss部分
        self.pixel_loss = torch.nn.L1Loss()

        self.cos_sim = torch.nn.CosineSimilarity()

        self.pixel_weight = pixel_weight
        self.recons_weight = recons_weight
        self.reg_weight = reg_weight

        print_weight = True
        if print_weight:
            print_wt('Loss function built. The loss weights are:')
            print_wt('  Pixel weight: {}'.format(self.pixel_weight))
            print_wt('  Semantic Reconstruction weight: {}'.format(self.recons_weight))
            print_wt('  Regularization weight: {}'.format(self.reg_weight))

    def forward(self, target_style_latent, pred_style_latent, input_clip):

        rebuild_image_pred = (self.stylegan(pred_style_latent) + 1) / 2
        rebuild_image_true = (self.stylegan(target_style_latent) + 1) / 2

        total_loss = 0

        # 语义重建一致性损失
        if self.recons_weight != 0:
            rebuild_clip_pred = self.clip.encode_image(self.preprocess(rebuild_image_pred))
            recons_loss = torch.mean(1 - self.cos_sim(input_clip, rebuild_clip_pred))
            total_loss += self.recons_weight * recons_loss
        else:
            recons_loss = torch.zeros([1], device=torch.device('cuda'))

        # latent直接损失
        if self.pixel_weight != 0:
            pixel_loss = self.pixel_loss(target_style_latent, pred_style_latent)
            total_loss += self.pixel_weight * pixel_loss
        else:
            pixel_loss = torch.zeros([1], device=torch.device('cuda'))

        # 均值&标准差正则项
        if self.reg_weight != 0:
            pred_latent_mean = torch.mean(pred_style_latent, dim=1)
            pred_latent_std = torch.std(pred_style_latent, dim=1)
            regularization_loss = torch.mean(torch.abs(pred_latent_mean)) + \
                                  torch.mean(torch.abs(pred_latent_std - torch.ones(pred_latent_std.shape[0]).cuda()))
            total_loss += self.reg_weight * regularization_loss
        else:
            regularization_loss = torch.zeros([1], device=torch.device('cuda'))

        loss_dict = {
            'pixel': pixel_loss,
            'recons': recons_loss,
            'reg': regularization_loss,
        }
        return total_loss, loss_dict
