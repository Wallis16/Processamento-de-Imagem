"""      
Utilizando o torchreid para realizar o processo de Person Re-Identification usando a linguagem python. Neste arquivo estao os parametros e o dataset utilizado."""

import torchreid

#escolhendo o dataset e definindo parametros

datamanager = torchreid.data.ImageDataManager(
    root='data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)

#definindo o backbone

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

model = model.cuda()

#definindo o otimizador

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

#definindo o tamanho dos steps

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

#carregando a engine

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

#definindo o path onde vai ficar salvo o modelo e tambem escolhendo o numero de epochs

engine.run(
    save_dir='log/resnet50',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False
)
