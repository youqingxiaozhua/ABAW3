_base_ = [
    '../_base_/datasets/ABAW3.py',
    '../_base_/default_runtime.py'
]

ce_head = dict(
    type='LinearClsHead',
    num_classes=8,
    in_channels=512,
    loss=dict(type='CrossEntropyLoss', loss_name='ce_loss', loss_weight=1.0),
    topk=(1,),
)

model = dict(
    type='ImageClassifier',
    # freeze=('backbone', ),
    backbone=dict(
        type='IRSE',
        input_size=(112, 112),
        num_layers=50,
        mode='ir',
        init_cfg=dict(type='Pretrained', checkpoint='weights/backbone_ir50_ms1m_epoch120.pth')
        # init_cfg=dict(type='Pretrained', checkpoint='weights/backbone_ir152_ms1m_epoch_112.pth')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=ce_head)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
)

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing', min_lr=0,
            warmup='linear',
            warmup_iters=100,
            warmup_ratio=0.1,
            warmup_by_epoch=False,)

runner = dict(max_iters=15000)
evaluation = dict(interval=5000)

