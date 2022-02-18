_base_ = [
    # '../_base_/datasets/AffWild2.py',
    '../_base_/datasets/ABAW3.py',
    '../_base_/default_runtime.py'
]

au_head = dict(
    type='MultiLabelLinearClsHead',
    in_channels=512,
    num_classes=12,
    loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
)
ce_head = dict(
    type='LinearClsHead',
    num_classes=8,
    in_channels=512,
    loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    topk=(1,),
)
va_head = dict(
    type='LinearRegHead',
    loss=dict(type='MSELoss', loss_weight=10),
    dims=(512, 2),
)

model = dict(
    type='ImageClassifier',
    # backbone=dict(
    #     type='IRSE',
    #     input_size=(112, 112),
    #     num_layers=50,
    #     mode='ir',
    #     init_cfg=dict(type='Pretrained', checkpoint='weights/backbone_ir50_ms1m_epoch120.pth')
    #     # init_cfg=dict(type='Pretrained', checkpoint='weights/backbone_ir152_ms1m_epoch_112.pth')
    #     ),
    backbone=dict(
        type='IResNet',
        depth=100,
        # init_cfg=dict(type='Pretrained', checkpoint='weights/insightface_ms1mv3_arcface_r50_fp16.pth')
        # init_cfg=dict(type='Pretrained', checkpoint='weights/insightface_glint360k_cosface_r50_fp16_0.1.pth')
        # init_cfg=dict(type='Pretrained', checkpoint='weights/IResNet_100_MS1MV3.pth')
        init_cfg=dict(type='Pretrained', checkpoint='weights/IResNet_100_glint360k.pth')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=ce_head)

data = dict(
    samples_per_gpu=256//2,
    workers_per_gpu=8,
)

# evaluation = dict(interval=30)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing', min_lr=0,
            warmup='linear',
            warmup_iters=1000,
            warmup_ratio=0.1,
            warmup_by_epoch=False,)
# runner = dict(type='EpochBasedRunner', max_epochs=100)




