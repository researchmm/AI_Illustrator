import torch
from torch.utils.data.dataset import Dataset

class clip2style_dataset(Dataset):
    def __init__(self, data_path):
        super(clip2style_dataset, self).__init__()
        latents = torch.load(data_path, map_location=torch.device('cpu'))
        self.style_latents = latents['Style_latent']
        self.clip_latents = latents['CLIP_latent']
        if self.style_latents.shape[0] != self.clip_latents.shape[0]:
            print('Data error. Unmatched numbers of style latents and clip latents.')
            raise ValueError

    def __getitem__(self, item):
        return self.style_latents[item], self.clip_latents[item]

    def __len__(self):
        return self.style_latents.shape[0]
