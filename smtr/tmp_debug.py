# %%
import pytorch_lightning as pl
import torch_geometric
import wandb
from pytorch_lightning.loggers import WandbLogger
import time
import torch
from smtr_e3nn_model import SMTR, create_transform, Ligand_dataset

# %%
learning_rate = 1e-5
max_epoch = 5
if_new_train = True
check_point = '20220516_114821.ckpt'
accelerator = 'cpu'
if torch.cuda.is_available():
    accelerator = 'gpu'


# %%
train_dataset = '/home/huabei/Projects/smtr/dataset/data_train.txt'
val_dataset = '/home/huabei/Projects/smtr/dataset/data_test.txt'
transform = create_transform()
# data
train_dataset = Ligand_dataset(train_dataset, transform=transform)
val_dataset = Ligand_dataset(val_dataset, transform=transform)
train_dataloader = torch_geometric.loader.DataLoader(train_dataset)
val_dataloader = torch_geometric.loader.DataLoader(val_dataset)


# %%
if if_new_train:
    # file_position = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # os.mkdir(file_position)
    smnn = SMTR(learning_rate=learning_rate)
else:
    smnn = SMTR().load_from_checkpoint(checkpoint_path=check_point)
project = 'smtr_e3nn'
wandb.init(project=project)
wandb_logger = WandbLogger(save_dir='.')
trainer = pl.Trainer(logger=wandb_logger, max_epochs=max_epoch,
                        auto_scale_batch_size=True, accelerator='cpu', devices=1)



# %%
trainer.fit(smnn, train_dataloader, val_dataloader)
trainer.save_checkpoint(time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".ckpt")

# %%



