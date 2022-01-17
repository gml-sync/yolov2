from glob import glob
from pathlib import Path
import os

import numpy as np
import cv2
import einops

import torch
import torch.utils.data as data
from torch import nn
from torch.autograd import Variable

DEVICE = 'cuda'

class RestorationDataset(data.Dataset):
    def __init__(self):
        self.feature_list = [] # input
        self.image_list = [] # gt
        self.desc_list = [] # input range - min, max

        dataset_root = "visualize"

        self.feature_list = sorted(glob(dataset_root + "/*features.png"))
        self.image_list = sorted(glob(dataset_root + "/*image.jpg"))
        self.desc_list = sorted(glob(dataset_root + "/*range.txt"))

    def __getitem__(self, index):
        index = index % len(self.image_list)
        index = index % 3



        # distort input features with ffmpeg
        # options: -y allow overwriting without confirmation
        #         -qp quality param, higher is worse. Take values [0, 22, 27, 32, 37]

        rand_qps = [0, 22, 27, 32, 37]
        qp_idx = np.random.randint(len(rand_qps))
        rand_qp = rand_qps[qp_idx]

        rand_filename = str(np.random.randint(1000000)).zfill(6)
        # encode
        os.system(f"ffmpeg -loglevel quiet -y -i {self.feature_list[index]} -c:v libx264 -qp {rand_qp} h264_{rand_filename}.mkv")
        # decode
        os.system(f"ffmpeg -loglevel quiet -i h264_{rand_filename}.mkv -r 1/1 output_{rand_filename}_%03d.bmp")

        h264_feat_path = f"output_{rand_filename}_001.bmp"
        min_feat, max_feat = 0, 0
        with open(self.desc_list[index], "r") as description:
            min_feat, max_feat = map(float, description.read().split())
        features = cv2.imread(h264_feat_path).astype(np.float32) / 255
        features = features * (max_feat - min_feat) + min_feat
        features = features[:, :, 0]
        features = einops.rearrange(features, '(i1 h) (i2 w) -> (i1 i2) h w', h=80, w=80) # 80 is specific to layer 5
        image = cv2.imread(self.image_list[index]).astype(np.float32) / 255

        features = torch.from_numpy(features).float()
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        os.system(f"rm h264_{rand_filename}.mkv {h264_feat_path}")

        return features, image
        
    def __len__(self):
        return len(self.image_list)

train_dataset = RestorationDataset()
f, i = train_dataset[0]
print(f.shape, i.shape)
print(train_dataset.feature_list[1000],
    train_dataset.image_list[1000],
    train_dataset.desc_list[1000])
print("Length of dataset:", len(train_dataset))
print(f"images: {len(train_dataset.image_list)}, features: {len(train_dataset.feature_list)}, desc: {len(train_dataset.desc_list)}")

class RestorationDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = self.upsample(256)
        self.bottle1_1 = self.bottleneck_block(128)
        self.bottle1_2 = self.bottleneck_block(128)

        self.up2 = self.upsample(128)
        self.bottle2 = self.bottleneck_block(64)

        self.up3 = self.upsample(64)
        self.conv_out = torch.nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def __call__(self, x):
        # print(f"x: {x.size()}")
        up1 = self.up1(x)
        # print(f"up1: {up1.size()}")
        # p = self.bottle1_1(up1)
        # print(f"bottle1_1: {p.size()}")
        bottle1_1 = self.bottle1_1(up1) + up1
        bottle1_2 = self.bottle1_2(bottle1_1) + bottle1_1

        up2 = self.up2(bottle1_2)
        bottle2 = self.bottle2(up2) + up2

        up3 = self.up3(bottle2)
        out = self.conv_out(up3)

        return out
    
    def upsample(self, in_channels):
        block = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(in_channels // 2),
            torch.nn.SiLU()
            )
        
        return block

    def bottleneck_block(self, in_channels):

        block = nn.Sequential(
                            torch.nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
                            torch.nn.BatchNorm2d(in_channels // 2),
                            torch.nn.SiLU(),
                            torch.nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1),
                            torch.nn.BatchNorm2d(in_channels),
                            torch.nn.SiLU()
                            )
        return block



model = RestorationDecoder()
random_tensor = torch.randn(2, 256, 80, 80) # b c h w
r = model(random_tensor)
print("Input shape:", random_tensor.shape)
print("Output shape:", r.shape)


import json
from pathlib import Path

def read_json(filename):
    data_dict = None
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            data_dict = json.load(f)
        except:
            pass

    return data_dict


def write_json(data_dict, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)


def checkpoint_load_path(path):
    '''
    path - str. Path of checkpoint
    returns path or backup path based on saved json.
    '''
    p = Path(path)
    info = Path(p.parent / (p.stem + '.json'))
    if not info.exists():
        return path

    checkpoint_info = read_json(info)
    if checkpoint_info is None: # JSON corruption means both checkpoints are good
        return path

    if checkpoint_info['backup_latest']:
        return p.parent / (p.stem + '.back')
    else:
        return path

def checkpoint_save_path(path, save_json=False):
    '''
    path - str. Path of checkpoint
    returns path or backup path based on saved json.
        JSON save must be done AFTER checkpoint saving
    '''
    p = Path(path)
    save_to_back = False

    info = Path(p.parent / (p.stem + '.json'))
    if info.exists():
        checkpoint_info = read_json(info)
        if not checkpoint_info is None:
            # JSON corruption means both checkpoints are good. Assume JSON is fine
            if not checkpoint_info['backup_latest']:
                # Latest checkpoint is not backup
                save_to_back = True

    if save_json:
        write_json({'backup_latest': save_to_back}, info)

    if not save_to_back:
        return path
    else:
        return p.parent / (p.stem + '.back')




import time

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.to(DEVICE)

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    print("WHAT??")
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc.item()*dataloader.batch_size # without .item pytorch does not free CUDA memory!
                running_loss += loss.item()*dataloader.batch_size

                if step % 1 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024), flush=True)
                    #print(torch.cuda.memory_summary())

                    # Save model
                    model_path = Path("checkpoints/restore_model.pth")
                    PATH = checkpoint_save_path(model_path)

                    checkpoint = {
                        'model': model,
                        'optimizer': optimizer,
                        'step': step
                    }
                    torch.save(checkpoint, PATH)
                    checkpoint_save_path(PATH, save_json=True)

                    gt_image = y.detach()[0].permute(1,2,0).cpu().numpy()
                    pred = outputs.detach()[0].permute(1,2,0).cpu().numpy()
                    outputs_dir = Path("outputs")
                    cv2.imwrite(str(outputs_dir / f"{step}_pred.jpg"), np.clip(pred * 255, 0, 255).astype(np.uint8))
                    cv2.imwrite(str(outputs_dir / f"{step}_gt_image.jpg"), (gt_image * 255).astype(np.uint8))
                
                if step > 6:
                    break

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            #clear_output(wait=True)
            # print('Epoch {}/{}'.format(epoch, epochs - 1))
            # print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss


# Create directories
model_dir = Path("checkpoints")
model_dir.mkdir(parents=True, exist_ok=True)

outputs_dir = Path("outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)




model = RestorationDecoder()
print("Parameters:", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Load model weights
model_path = Path("checkpoints/restore_model.pth")
path = checkpoint_load_path(model_path)
if Path(path).exists():
    checkpoint = torch.load(path, map_location=torch.device(DEVICE))
    if 'model' in checkpoint: # New format, full save
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        step = checkpoint['step']
        print('Continue from', step, 'step')

train_dataset = RestorationDataset()

train_loader = data.DataLoader(train_dataset, batch_size=1, 
        pin_memory=False, shuffle=True, num_workers=1, drop_last=True) # batch size 16, workers 4

train_loss, valid_loss = train(model, train_loader, None, loss_fn, optimizer, loss_fn, epochs=2)

# speed: ~10 sec per 100 images 