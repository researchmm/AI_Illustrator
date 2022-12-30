import os
import time
import clip
import random
import pickle
import torch
import torchvision
import argparse
from myutils import print_wt
from models.stylegan2 import generator_discriminator
from models import clip2style
from build_SE_CIE_pairs import my_preprocess
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', type=str, default='human')
    parser.add_argument('--projector_path', type=str, default='./pretrained_projectors/c2s_human.pth')
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--strength', type=float, default=1.75)
    parser.add_argument('--random_generate', action='store_true')
    parser.add_argument('--random_layer', type=int, default=1)
    args = parser.parse_args()

    print_wt('Manuscript starts.')
    model, _ = clip.load("ViT-B/32", device=torch.device('cuda'))
    projection = clip2style.LatentMapping_C2S()
    projection.load_state_dict(torch.load(args.projector_path))
    projection.cuda()
    projection.eval()

    if args.kind == 'human':
        StyleGAN_Gen = generator_discriminator.StyleGANv2Generator(out_size=1024, style_channels=512, bgr2rgb=True)
        StyleGAN_total_checkpoint = torch.utils.model_zoo.load_url(
            'http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f'
            '-official_20210327_171224-bce9310c.pth')
        clip_image_embeds = torch.load(
            './SE_CIE_pairs/Style_CLIP_pairs_ffhq_train_w_norm_512.pth')
        source_text = 'A normal human face.'

    elif args.kind == 'cat':
        StyleGAN_Gen = generator_discriminator.StyleGANv2Generator(out_size=256, style_channels=512, bgr2rgb=True)
        StyleGAN_total_checkpoint = torch.utils.model_zoo.load_url(
            'http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f'
            '-official_20210327_172444-15bc485b.pth')
        clip_image_embeds = torch.load(
            './SE_CIE_pairs/Style_CLIP_pairs_cat_train_w_norm_512.pth')
        source_text = 'A Cat.'

    elif args.kind == 'church':
        StyleGAN_Gen = generator_discriminator.StyleGANv2Generator(out_size=256, style_channels=512, bgr2rgb=True)
        StyleGAN_total_checkpoint = torch.utils.model_zoo.load_url(
            'http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f'
            '-official_20210327_172657-1d42b7d1.pth')
        clip_image_embeds = torch.load(
            './SE_CIE_pairs/Style_CLIP_pairs_church_train_w_norm_512.pth')
        source_text = 'A normal church.'

    else:
        raise ValueError('No such kind.')

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
    preprocess = my_preprocess()

    avg_clip_embed = torch.mean(clip_image_embeds['CLIP_latent'], dim=0).unsqueeze(0).cuda()
    avg_clip_embed = avg_clip_embed / avg_clip_embed.norm(dim=-1, keepdim=True)
    avg_clip_embed *= 512 ** 0.5

    generate_rank = 1
    while True:
        target_text = input('Input your description: ')
        texts = clip.tokenize([source_text, target_text]).cuda()

        text_clip_embed = model.encode_text(texts).float()
        text_clip_embed = text_clip_embed / text_clip_embed.norm(dim=-1, keepdim=True)
        text_clip_embed *= 512 ** 0.5
        delta_text_clip_embed = (text_clip_embed[1] - text_clip_embed[0]).unsqueeze(0)

        generate_strength = args.strength
        input_image_clip_embed = avg_clip_embed + generate_strength * delta_text_clip_embed
        input_image_clip_embed = input_image_clip_embed / input_image_clip_embed.norm(dim=-1, keepdim=True)
        input_image_clip_embed *= 512 ** 0.5

        pred_style = projection(input_image_clip_embed)
        rand_style = torch.randn((1, 512), device=torch.device('cuda'))
        if args.random_generate:
            generated_image = (StyleGAN_Gen([rand_style, pred_style], inject_index=args.random_layer) + 1) / 2
        else:
            generated_image = (StyleGAN_Gen([pred_style]) + 1) / 2

        save_dir = args.save_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torchvision.utils.save_image(generated_image, os.path.join(save_dir, str(generate_rank) + '.png'))
        print_wt('Image saved at {}.'.format(generate_rank))
        generate_rank += 1
