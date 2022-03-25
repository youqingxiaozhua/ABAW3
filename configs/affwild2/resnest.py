_base_ = [
    '../_base_/datasets/ABAW3.py',
    '../_base_/default_runtime.py'
]

num_classes = 4

ce_head = dict(
    type='LinearClsHead',
    num_classes=num_classes,
    in_channels=2048,
    loss=[
        dict(type='CrossEntropyLoss', loss_name='ce_loss', loss_weight=1.0),
        ],
    topk=(1,),
)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeStFaceX',
        num_layers=50, 
        drop_ratio=0.4,
        feat_dim=512, out_h=7, out_w=7,
        init_cfg=dict(type='Pretrained', checkpoint='weights/face/FaceX-ResNeSt50.pt', prefix='backbone.')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=ce_head)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
)

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing', min_lr=0,
            warmup='linear',
            warmup_iters=100,
            warmup_ratio=0.1,
            warmup_by_epoch=False,)

runner = dict(max_iters=10000)

