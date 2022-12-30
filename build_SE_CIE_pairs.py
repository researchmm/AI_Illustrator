import os
import clip
import argparse

import torch
import torchvision
import torch.nn.functional as F
from myutils import print_wt
from models.stylegan2 import generator_discriminator
from PIL import Image

num_data = 10000
batch_num = 15
name = 'test'
# curr_data = '/home/v-yiyangma/local_codes/ASC/Style_CLIP_pairs_train_1.pth'
curr_data = ''
index = 1
kind = 'cat'


class my_preprocess(torch.nn.Module):
    def __init__(self, in_size=1024):
        super(my_preprocess, self).__init__()
        self.in_size=in_size
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-data', type=int, default=num_data)
    parser.add_argument('--batch-num', type=int, default=batch_num)
    parser.add_argument('--name', type=str, default=name)
    parser.add_argument('--curr-data', type=str, default=curr_data)
    parser.add_argument('--index', type=int, default=index)
    parser.add_argument('--kind', type=str, default=kind)
    args = parser.parse_args()

    print_wt('Manuscript starts.')

    print_wt('Going to build {} pairs of {} to {}.'.format(args.num_data, args.kind, args.name))
    StyleGAN_Gen = generator_discriminator.StyleGANv2Generator(out_size=256, style_channels=512, bgr2rgb=True)
    # StyleGAN_total_checkpoint = torch.utils.model_zoo.load_url('http://download.openmmlab.com/mmgen/stylegan2'
    #                                                            '/official_weights/stylegan2-ffhq-config-f'
    #                                                            '-official_20210327_171224-bce9310c.pth')
    # StyleGAN_total_checkpoint = torch.utils.model_zoo.load_url('http://download.openmmlab.com/mmgen/stylegan2'
    #                                                        '/official_weights/stylegan2-church-config-f'
    #                                                        '-official_20210327_172657-1d42b7d1.pth')
    StyleGAN_total_checkpoint = torch.utils.model_zoo.load_url('http://download.openmmlab.com/mmgen/stylegan2'
                                                               '/official_weights/stylegan2-cat-config-f'
                                                               '-official_20210327_172444-15bc485b.pth')
    StyleGAN_total_state_dict = StyleGAN_total_checkpoint['state_dict']
    modified_state_dict = {}
    pre_fix = 'generator_ema'
    for k, v in StyleGAN_total_state_dict.items():
        if k[0:len(pre_fix)] != pre_fix:
            continue
        modified_state_dict[k[len(pre_fix) + 1:]] = v
    StyleGAN_Gen.load_state_dict(modified_state_dict)
    StyleGAN_Gen.cuda()
    StyleGAN_Gen.eval()

    model, preprocess = clip.load('ViT-B/32', device='cuda')
    model.eval()
    toimage = torchvision.transforms.ToPILImage()

    if os.path.exists(args.curr_data):
        curr_embeds = torch.load(args.curr_data, map_location=torch.device('cpu'))
        style_embeds = curr_embeds['Style_latent']
        clip_embeds = curr_embeds['CLIP_latent']
        if len(style_embeds.shape) != 2 or style_embeds.shape[1] != 512 or len(clip_embeds.shape) != 2 or clip_embeds.shape[1] != 512 or style_embeds.shape[0] != clip_embeds.shape[0]:
            assert ValueError('Load error!')
        print_wt('Generating to an existing data-bank with size {}.'.format(style_embeds.shape[0]))
    else:
        style_embeds = None
        clip_embeds = None
        print_wt('Generating to a new data-bank.')

    save_path = './SE_CIE_pairs/Style_CLIP_pairs_' + args.kind + '_' + args.name + '_w_norm_512.pth'
    print_wt('Save path is {}.'.format(save_path))


    _preprocess = my_preprocess(in_size=256)
    _preprocess.cuda()
    _preprocess.eval()

    print_wt('Start generating.')
    with torch.no_grad():
        iters = 0
        for i in range(args.num_data // args.batch_num):
            iters += 1
            z = torch.randn([args.batch_num, 512], device=torch.device('cuda'))
            imgs = (StyleGAN_Gen(z) + 1) / 2
            imgs = torch.clamp(imgs, 0, 1)
            imgs = _preprocess(imgs)
            # print(imgs.shape)
            img_embeds = model.encode_image(imgs).float()
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            img_embeds *= 512 ** 0.5
            # print(img_embeds.norm(dim=-1))
            # cos_sim = F.cosine_similarity(img_embeds[0].unsqueeze(0), img_embeds[1].unsqueeze(0))
            # print(cos_sim)

            if style_embeds is None:
                style_embeds = z.cpu()
            else:
                style_embeds = torch.cat([style_embeds, z.cpu()], dim=0)

            if clip_embeds is None:
                clip_embeds = img_embeds.cpu()
            else:
                clip_embeds = torch.cat([clip_embeds, img_embeds.cpu()], dim=0)

            # print(style_embeds.dtype, clip_embeds.dtype)

            if iters % 10 == 0:
                print_wt('{} iters end. {} pairs generated.'.format(iters, iters * args.batch_num))

        print_wt('Generation ends. Now, number of total pairs is {}.'.format(style_embeds.shape[0]))
        style_embeds = style_embeds.cpu()
        clip_embeds = clip_embeds.cpu()
        pairs = {
            'Style_latent':style_embeds,
            'CLIP_latent':clip_embeds
        }
        torch.save(pairs, save_path)
        print_wt('New data saved to {}.'.format(save_path))


