import argparse
import os
import sys

from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.distributed as dist
import clip

from myutils import print_wt, avg_loss
from models.stylegan2 import generator_discriminator
from models import clip2style
from data_pipes import data_clip2style
import loss
import build_demo

TRAIN_LATENT_PATH = '/home/v-yiyangma/local_codes/PCM-Frame/training_datas/Style_CLIP_pairs_cat_train_w_norm_512.pth'
TEST_LATENT_PATH = '/home/v-yiyangma/local_codes/PCM-Frame/training_datas/Style_CLIP_pairs_cat_test_w_norm_512.pth'
SAVE_PATH = '/home/v-yiyangma/models/PCM-Frame/new_clip2style_cat.pth'
SAVE_FILEFOLD = '/home/v-yiyangma/models/PCM-Frame/'
LOAD_PATH = '/home/v-yiyangma/models/PCM-Frame/saved/c2s_ablation_wl1.pth'

BATCH_SIZE = 8
MAX_ITERS = 75000
PER_VALID_ITER = 2000
PER_REPORT_ITER = 100
VALID_ITERS = 200
GET_DEMOS = 5
INIT_LEARNING_RATE = 1e-4
DATA_KIND = 'cat'

parser = argparse.ArgumentParser()
parser.add_argument('--train-latent-path', type=str, default=TRAIN_LATENT_PATH)
parser.add_argument('--test-latent-path', type=str, default=TEST_LATENT_PATH)
parser.add_argument('--save-path', type=str, default=SAVE_PATH)
parser.add_argument('--save-filefold', type=str, default=SAVE_FILEFOLD)
parser.add_argument('--load-path', type=str, default=LOAD_PATH)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('--max-iters', type=int, default=MAX_ITERS)
parser.add_argument('--per-valid-iter', type=int, default=PER_VALID_ITER)
parser.add_argument('--per-report-iter', type=int, default=PER_REPORT_ITER)
parser.add_argument('--valid-iters', type=int, default=VALID_ITERS)
parser.add_argument('--get-demos', type=int, default=GET_DEMOS)
parser.add_argument('--init-learning-rate', type=float, default=INIT_LEARNING_RATE)
parser.add_argument('--data-kind', type=str, default=DATA_KIND)

parser.add_argument('--local-rank', type=int, default=-1)

args = parser.parse_args()

if __name__ == '__main__':
    args.local_rank = int(os.environ['LOCAL_RANK'])
    print_wt('Process No.{} starts.'.format(args.local_rank))

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    projection = clip2style.LatentMapping_C2S()
    if args.pretrained:
        if not os.path.exists(args.load_path):
            raise ValueError('No such path {} to load model.'.format(args.load_path))
        projection_checkpoint = torch.load(args.load_path, map_location=torch.device('cpu'))
        modified_projection_checkpoint = {}
        for k, v in projection_checkpoint.items():
            if k[0:len('module')] == 'module':
                modified_projection_checkpoint[k[len('module') + 1:]] = v
            else:
                modified_projection_checkpoint[k] = v
        projection.load_state_dict(modified_projection_checkpoint)
        if args.local_rank == 0:
            print_wt('Model loaded.')
    else:
        if args.local_rank == 0:
            print_wt('New model built.')
    projection = torch.nn.SyncBatchNorm.convert_sync_batchnorm(projection)
    projection = projection.cuda()
    projection = torch.nn.parallel.DistributedDataParallel(
        projection,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False
    )

    train_dataset = data_clip2style.clip2style_dataset(data_path=args.train_latent_path)
    test_dataset = data_clip2style.clip2style_dataset(data_path=args.test_latent_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, rank=args.local_rank
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, rank=args.local_rank
    )
    train_loader = DataLoader(train_dataset, num_workers=12, batch_size=args.batch_size, shuffle=False,
                              sampler=train_sampler)
    test_loader = DataLoader(test_dataset, num_workers=12, batch_size=args.batch_size, shuffle=False,
                             sampler=test_sampler)
    if args.local_rank == 0:
        print_wt('Dataloader built.')

    optimizer = torch.optim.Adam(projection.parameters(), lr=args.init_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iters, eta_min=1e-7)
    if args.local_rank == 0:
        print_wt('Optimizer built. Initial learning rate is {}.'.format(args.init_learning_rate))

    loss_fn = loss.total_loss(DataKind=args.data_kind)
    if args.local_rank == 0:
        print_wt('Loss built.')

    if args.local_rank == 0:
        print_wt('Training starts. Going to train {} iterations.'.format(args.max_iters))
    curr_iters = 0
    epochs = 0
    best_valid_loss = 100
    best_valid_iter = -1
    while True:
        train_sampler.set_epoch(epochs)
        epochs += 1
        train_loss = 0
        train_iters = 0
        train_loss_dict = {
            'pixel': 0,
            'recons': 0,
            'reg': 0,
        }
        projection.train()
        if args.local_rank == 0:
            print_wt('Epoch {} Starts.'.format(epochs))
        for style_latent, clip_latent in train_loader:
            optimizer.zero_grad()
            curr_iters += 1
            train_iters += 1
            style_latent = style_latent.cuda()
            clip_latent = clip_latent.cuda()
            pred_style_latent = projection(clip_latent)
            loss, loss_dict = loss_fn(style_latent, pred_style_latent, clip_latent)
            for key, value in train_loss_dict.items():
                train_loss_dict[key] += avg_loss(loss_dict[key], dist.get_world_size()).item()
            loss.backward()
            optimizer.step()
            train_loss += avg_loss(loss, dist.get_world_size()).item()
            if train_iters % args.per_report_iter == 0 and args.local_rank == 0:
                print_wt('  {} iters end.'.format(train_iters))
                print_wt('    Avg loss is {}.'.format(train_loss / train_iters))
                for key, value in train_loss_dict.items():
                    print_wt('      {} loss is {}.'.format(key, value / train_iters))
            scheduler.step()

            if curr_iters % args.per_valid_iter == 0 or curr_iters == args.max_iters:
                if args.local_rank == 0:
                    print_wt('  Start to valid at iteration {}.'.format(curr_iters))
                eval_iters = 0
                eval_loss = 0
                eval_loss_dict = {
                    'pixel': 0,
                    'recons': 0,
                    'ID': 0,
                    'reg': 0,
                    'eyes': 0,
                    'mouth': 0
                }
                projection.eval()
                demo_style_latent = []
                demo_clip_latent = []
                demo_pred_style_latent = []
                demo_count = 0
                for style_latent, clip_latent in test_loader:
                    with torch.no_grad():
                        eval_iters += 1
                        style_latent = style_latent.cuda()
                        clip_latent = clip_latent.cuda()
                        pred_style_latent = projection(clip_latent)
                        loss, loss_dict = loss_fn(style_latent, pred_style_latent, clip_latent)
                        for key, value in eval_loss_dict.items():
                            eval_loss_dict[key] += avg_loss(loss_dict[key], dist.get_world_size()).item()
                        eval_loss += avg_loss(loss, dist.get_world_size()).item()
                        if demo_count < args.get_demos:
                            demo_style_latent.append(style_latent[0])
                            demo_clip_latent.append(clip_latent[0])
                            demo_pred_style_latent.append(pred_style_latent[0])
                            demo_count += 1
                        if eval_iters % 100 == 0 and args.local_rank == 0:
                            print_wt('  valid {} iters end.'.format(eval_iters))
                        if eval_iters >= args.valid_iters:
                            break

                if args.local_rank == 0:
                    # for i in range(len(demo_pred_style_latent)):
                    #     build_demo.build_demo_from_single_latent_set(demo_style_latent[i], demo_pred_style_latent[i], curr_iters, i)

                    print_wt('  Valid loss is {}.'.format(eval_loss / eval_iters))
                    for key, value in eval_loss_dict.items():
                        print_wt('      {} loss is {}.'.format(key, value / eval_iters))

                    print_wt('  learning rate is {} now.'.format(optimizer.param_groups[0]['lr']))
                    print_wt('  Number of total iterations is {}.'.format(curr_iters))

                    torch.save(projection.module.state_dict(), args.save_filefold + str(curr_iters) + '_c2s.pth')
                    if eval_loss / eval_iters < best_valid_loss:
                        best_valid_iter = curr_iters
                        best_valid_loss = eval_loss / eval_iters
                        torch.save(projection.module.state_dict(), args.save_path)
                        print_wt('Best model saved at iteration {}.'.format(curr_iters))

            if curr_iters >= args.max_iters:
                if args.local_rank == 0:
                    print_wt('Training ends. Number of total iterations is {}.'.format(curr_iters))
                    print_wt('Best model saved at iteration {} with valid loss {}.'.format(best_valid_iter, best_valid_loss))
                sys.exit()

        if args.local_rank == 0:
            print_wt('{} epochs end.'.format(epochs))

