_base_ = [
    '../_base_/datasets/ABAW3.py',
    '../_base_/default_runtime.py'
]

num_classes = 4

ce_head = dict(
    type='LinearClsHead',
    num_classes=num_classes,
    in_channels=768,
    loss=[
        dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=num_classes,
            loss_name='ce_loss',
            loss_weight=1.0)
        ],
    topk=(1,),
)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformerFaceX',
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.3,
        qkv_bias=True,
        qk_scale=None,
        ape=False, patch_norm=True, use_checkpoint=False,
        init_cfg=dict(type='Pretrained', checkpoint='weights/Swin-S-MS1M-Epoch_17.pth', prefix='backbone.')
    ),
    head=ce_head)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
)

# optimizer
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })
optimizer = dict(type='AdamW', lr=5e-4, eps=1e-8,
    betas=(0.9, 0.999), weight_decay=0.05, paramwise_cfg=paramwise_cfg)    # swin
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing', min_lr=0,
            warmup='linear',
            warmup_iters=100,
            warmup_ratio=0.1,
            warmup_by_epoch=False,)

runner = dict(max_iters=20000)


