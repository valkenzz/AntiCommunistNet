import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import sys
import time
import torch
import torch.nn

from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer
import numpy as np
import random

# config
dataset_path = './dataset/'
#pretrained_path = './xception-b5690688.pth'
pretrained_path = './nonmaisnimportquoi'

batch_size = 12
gpu_ids = [0]
max_epoch = 100
loss_freq = 40
mode = 'Original' # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = './logSave'
ckpt_name = 'logGENERAL'


if __name__ == '__main__':
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'real'), size=256, frame_num=4000, augment=True)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8)
    
    len_dataloader = dataloader_real.__len__()
    print(len_dataloader)
    dataset_img, total_len =  get_dataset(name='train', size=256, root=dataset_path, frame_num=4000, augment=True)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8
    )

    # init checkpoint and logger
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    best_val = 0.
    ckpt_model_name = 'best.pkl'
    
    # train
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0
    epoch = 0
    
    while epoch < max_epoch:

        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)
        
        logger.debug(f'No {epoch}')
        i = 0

        while i < len_dataloader:
            
            i += 1
            model.total_steps += 1

            try:
                data_real = real_iter.next()
                data_fake = fake_iter.next()
            except StopIteration:
                break
            # -------------------------------------------------
            
            if data_real.shape[0] != data_fake.shape[0]:
                continue

            bz = data_real.shape[0]
            
            data = torch.cat([data_real,data_fake],dim=0)
            label = torch.cat([torch.zeros(bz).unsqueeze(dim=0),torch.ones(bz).unsqueeze(dim=0)],dim=1).squeeze(dim=0)

            # manually shuffle
            idx = list(range(data.shape[0]))
            random.shuffle(idx)
            data = data[idx]
            label = label[idx]

            data = data.detach()
            label = label.detach()

            model.set_input(data,label)
            loss = model.optimize_weight()

            if model.total_steps % loss_freq == 0:
                logger.debug(f'loss: {loss} at step: {model.total_steps}')
#            print('un ici')
#            print(i)
#            print(len_dataloader) 
#            if i % int(len_dataloader / 10) == 0:
                model.model.eval()
           # auc, r_acc, f_acc = evaluate(model, dataset_path, mode='valid') 
           # logger.debug(f'(Val @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
                auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
                logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}') 
                model.model.train()
        epoch = epoch + 1

    model.model.eval()
    auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
    logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
