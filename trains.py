import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Pytorch.photo_detection.photo_detection_YOLOv3.yolo3_pytorch.nets.yolo3 import YoloBody
from Pytorch.photo_detection.photo_detection_YOLOv3.yolo3_pytorch.utils.dataloader import yolo_dataset_collate, YoloDataset
from Pytorch.photo_detection.photo_detection_YOLOv3.yolo3_pytorch.nets.yolo_training import YOLOLoss, Generator
from Pytorch.photo_detection.photo_detection_YOLOv3.yolo3_pytorch.utils.config import Config
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def args_parse():
    parser = argparse.ArgumentParser(description="训练的参数")
    parser.add_argument('-input_size', default=(416, 416), type=int, help='input image size', dest='input_size')
    # parser.add_argument('-mosaic', default=True, type=bool, help='use mosaic data augumentation')
    # parser.add_argument('-anchors', default="./model_data/yolo_anchors.txt", type=str, help='anchor file')
    # parser.add_argument('-classes', default="./model_data/voc_classes.txt", type=str, help='classes name')
    parser.add_argument('-annotation', default="./2007_train.txt", type=str, help='annotation file')
    parser.add_argument('-model', default="./model_data/yolo_weights.pth", type=str, help='model file')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-batch_size', default=16, type=int, help='train data batch size')
    parser.add_argument('-epochs', default=50, type=int, help='train epoch size')
    args = parser.parse_args()
    return args


def parse_lines(path):  # loads the lines
    with open(path) as f:
        lines_name = f.readlines()
    return lines_name


def get_classes(path):  # loads the classes
    class_names = parse_lines(path)
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(path):
    with open(path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train():
    args = args_parse()
    # anchors = get_anchors(args.anchors)  # get anchors
    # num_classes = len(get_classes(args.classes))  # get classes
    # 绘制模型
    # model = YoloBody(len(anchors[0]), num_classes)
    model = YoloBody(Config)

    # 使用预训练模型，如果显卡不够大的话可以使用预训练模型来微调
    print("Load pretrained model into state dict...")
    model_list = model.state_dict()
    pretrained_dict = torch.load(args.model, map_location="cuda:0")  # Load pretrained model

    pretrained_dicts = {}
    for k, v in pretrained_dict.items():
        # print(k)
        if np.shape(model_list[k]) == np.shape(v):
            pretrained_dicts[k] = v

    model_list.update(pretrained_dicts)
    model.load_state_dict(model_list)
    print("Finished!")

    model.train()
    model.to(device)

    # input_shape = args.input_size
    # for i in range(3):
    #     yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes,
    #                                 (input_shape[1], input_shape[0]), label_smooth=False))
    yolo_losses = []  # creat loss function
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), cuda=True))

    val_slits = 0.1  # 训练集验证集分配
    lines = parse_lines(args.annotation)
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    val_num = int(len(lines) * val_slits)
    train_num = len(lines) - val_num

    # witer = SummaryWriter(log_dir="./log", flush_secs=60)  # 进行训练可视化
    # graph_inputs = torch.from_numpy(np.random.rand(1, 3, input_shape[1], input_shape[0])).type(torch.FloatTensor).cuda()
    # witer.add_graph(model, graph_inputs)

    batch_size = args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), args.lr)  # 优化器
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)  # 学习率余弦退火衰减
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    train_dataset = YoloDataset(lines[:train_num], (Config["img_h"], Config["img_w"]))
    val_dataset = YoloDataset(lines[train_num:], (Config["img_h"], Config["img_w"]))
    train = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    val = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    # train_epoch_size = max(1, train_num//batch_size)
    train_epoch_size = train_num // batch_size
    val_epoch_size = val_num // batch_size

    for param in model.backbone.parameters():  # 冻结部分网络
        param.requires_grad = False

    epochs = args.epochs
    for epoch in range(0, epochs):
        total_loss = 0
        val_loss = 0
        start = time.time()
        model.train()
        with tqdm(total=train_epoch_size, desc=f"Epoch {epoch + 1} / {epochs}") as pbar:  # 进行训练
            for i, (data, targets) in enumerate(train):
                if i >= train_epoch_size:
                    break
                images = torch.from_numpy(data).cuda()
                targets = [torch.from_numpy(ann) for ann in targets]

                optimizer.zero_grad()
                outputs = model(images)

                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets)
                    losses.append(loss_item[0])
                loss = sum(losses)
                loss.backward()
                optimizer.step()

                total_loss += loss
                spend_time = time.time() - start

                dicts = {"step/s": spend_time, "lr": get_lr(optimizer), "total_loss": total_loss.item() / (i + 1)}
                pbar.set_postfix(dicts)  # 进度条右提示
                pbar.update(1)

                writer = SummaryWriter(log_dir="./loges", flush_secs=60)  # 进行训练可视化
                # 将loss写入tensorboard，每一步都写
                writer.add_scalar('Train_loss', loss, (epoch * train_epoch_size + i))

        start_time = time.time()
        model.eval()
        print("Start validation")
        with tqdm(total=val_epoch_size, desc=f'Epoch {epoch + 1} / {epochs}') as pbar:
            for i, (val_data, targets_val) in enumerate(val):
                if i >= val_epoch_size:
                    break
                with torch.no_grad():
                    images_val = torch.from_numpy(val_data).to(device)
                    targets_val = [torch.from_numpy(ann) for ann in targets_val]
                    optimizer.zero_grad()
                    outputs = model(images_val)

                    losses = []
                    for i in range(3):
                        loss_item = yolo_losses[i](outputs[i], targets_val)
                        losses.append(loss_item[0])
                    loss = sum(losses)
                    val_loss += loss

                spend_time = time.time() - start_time
                dicts = {"step/s": spend_time, "val_loss": val_loss.item() / (i + 1)}
                pbar.set_postfix(dicts)  # 进度条右提示
                pbar.update(1)

        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(epochs))
        print(
            'Total Loss: %.4f || Val Loss: %.4f ' % (
            total_loss / (train_epoch_size + 1), val_loss / (val_epoch_size + 1)))

        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(), 'log/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % ((epoch + 1), total_loss / (train_epoch_size + 1), val_loss / (val_epoch_size + 1)))
        lr_scheduler.step()


if __name__ == "__main__":
    train()
