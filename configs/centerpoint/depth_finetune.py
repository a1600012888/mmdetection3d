_base_ =  './depth.py'

model = dict(pretrained=None,
             finetune=True)

load_from = './epoch_20.pth'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6)
