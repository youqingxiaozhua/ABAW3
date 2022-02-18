# dataset settings
dataset_type = 'AffWild2'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_size = 112

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=img_size),
    # dict(type='RandomRotate', prob=0.5, degree=6),
    dict(type='RandomResizedCrop', size=img_size, scale=(0.8, 1.0), ratio=(1. / 1., 1. / 1.)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomGrayscale', gray_prob=0.2),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.4,
    #             contrast=0.4,
    #             saturation=0.4,
    #             hue=0.1)
    #     ],
    #     p=0.8),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='GaussianBlur',
    #             sigma_min=0.1,
    #             sigma_max=2.0)
    #     ],
    #     p=0.5),
    dict(
        type='RandomErasing',
        erase_prob=0.5,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label', 
    # 'au_label'
    ]),
    dict(type='Collect', keys=['img', 'gt_label', 
    # 'au_label'
    ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(256, -1)),
    dict(type='Resize', size=img_size),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label', ]),
    dict(type='Collect', keys=['img', ])
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     # dict(type='Resize', size=224),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(img_size, img_size),
#         img_ratios=[1.0,],
#         flip=True,
#         num=2,
#         transforms=[
#             # dict(type='RandomRotate', prob=0.5, degree=3),
#             dict(type='RandomResizedCrop', size=img_size, scale=(0.9, 1.0), ratio=(1. / 1., 1. / 1.)),
#             dict(type='RandomFlip'),
#             # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

base_path = 'data/Aff-Wild2/ABAW3/'
# image_path = base_path + 'cropped_aligned_224'
image_path = base_path + 'cropped_aligned'

task = 'EXPR_Classification'
# task = 'AU_Detection'
# task = 'VA_Estimation'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.15,
        method='sqrt',
        # method='reciprocal',
        dataset=dict(
            type=dataset_type,
            data_prefix=image_path,
            ann_file=base_path + f'Third ABAW Annotations/{task}_Challenge/Train_Set/',
            pipeline=train_pipeline),
    ),
    # train=dict(
    #     type=dataset_type,
    #     data_prefix=image_path,
    #     ann_file=base_path + f'annotations/{task}/Train_Set/',
    #     pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=image_path,
        ann_file=base_path + f'Third ABAW Annotations/{task}_Challenge/Validation_Set',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=image_path,
        ann_file=base_path + f'Third ABAW Annotations/{task}_Challenge/Validation_Set',
        pipeline=test_pipeline),
    vis=dict(
        type=dataset_type,
        data_prefix=image_path,
        ann_file=base_path + 'EmoLabel/ViTPooling_improved.csv',
        pipeline=test_pipeline),
    # all=dict(
    #     type=dataset_type,
    #     data_prefix=image_path,
    #     ann_file=base_path + 'EmoLabel/all.txt',
    #     pipeline=test_pipeline
    # )
)

# lr_config = dict(
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.1,
#     warmup_by_epoch=False,
# )

# bs, iter
# 64, 192
# 128, 96
# 256, 48
# 512, 24

if task == 'EXPR_Classification':
    metrics = ['accuracy', 'f1_score', 'class_accuracy']
    save_best='f1_score.mean'
    rule='greater'
elif task == 'AU_Detection':
    metrics = ['f1_score']
    save_best='f1_score.mean'
    rule='greater'
elif task == 'VA_Estimation':
    metrics = ['MSE', 'CCC']
    save_best='ccc_va'
    rule='greater'
else:
    raise ValueError('invalid task value')

evaluation = dict(interval=2000, metric=metrics, metric_options=dict(average_mode='none'), save_best=save_best, rule=rule)
checkpoint_config = dict(create_symlink=False, max_keep_ckpts=1, by_epoch=False, interval=2000)
runner = dict(type='IterBasedRunner', max_iters=40000)  # 10k = 1 epoch when bs=256
lr_config = dict(by_epoch=False,)

