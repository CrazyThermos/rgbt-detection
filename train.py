import yaml
from dataset.base_dataset import create_dataloader
from utils.general import LOCAL_RANK
# # train on the GPU or on the CPU, if a GPU is not available
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # our dataset has two classes only - background and person
# num_classes = 2
# # use our dataset and defined transformations
# dataset = None
# dataset_test = None

# # split the dataset in train and test set
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-50])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# # define training and validation data loaders
# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=2,
#     shuffle=True,
#     num_workers=4,
#     collate_fn=utils.collate_fn
# )

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test,
#     batch_size=1,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=utils.collate_fn
# )

# # get the model using our helper function
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

# # move model to the right device
# model.to(device)

# # construct an optimizer
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
#     params,
#     lr=0.005,
#     momentum=0.9,
#     weight_decay=0.0005
# )

# # and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=3,
#     gamma=0.1
# )

# # let's train it for 5 epochs
# num_epochs = 5

# for epoch in range(num_epochs):
#     # train for one epoch, printing every 10 iterations
#     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
#     # update the learning rate
#     lr_scheduler.step()
#     # evaluate on the test dataset
#     evaluate(model, data_loader_test, device=device)
if __name__ == '__main__':
    hyp = "configs/hyp.scaratch-low.yaml"
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
        if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            hyp['anchors'] = 3
    ndataloader, ndataset = create_dataloader(path="../../datasets/TEST",
                                            imgsz=640,
                                            batch_size=1,
                                            stride=32,
                                            single_cls=False,
                                            hyp=hyp,
                                            augment=True,
                                            cache="ram",
                                            rect=False,
                                            rank=LOCAL_RANK,
                                            workers=8,
                                            image_weights=False,
                                            quad=False,
                                            prefix='train:',
                                            shuffle=True,
                                            seed=0)
    pbar = enumerate(ndataloader)
    for i, (imgs, targets, paths, _) in pbar:  
        print(imgs)
    pass