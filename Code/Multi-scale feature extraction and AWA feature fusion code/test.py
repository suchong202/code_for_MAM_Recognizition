import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets import get_model_from_name
from utils.callbacks import LossHistory
from utils.dataloader_fusion import DataGenerator, detection_collate
from utils.utils import (download_weights, get_classes, get_lr_scheduler,
                         set_optimizer_lr, show_config, weights_init)
from utils.utils_fit import fit_one_epoch,test_one_epoch

if __name__ == "__main__":

    mode = 'adaptive'

    classes_path = 'model_data/cls_classes.txt'
    input_shape = [224, 224]
    backbone = "mobilenetv2"

    model_path = r'logs\xx.pth' # 放入训练完成后保存在logs的文件夹中的权重文件


    train_annotation_path = "cls_train_fusion.txt"
    val_annotation_path = 'cls_val_fusion.txt'
    test_annotation_path = 'cls_test_fusion.txt'

    class_names, num_classes = get_classes(classes_path)
    pretrained=False
    if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        model = get_model_from_name[backbone](num_classes=num_classes, pretrained=pretrained)
    else:
        model = get_model_from_name[backbone](input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)


    from fusion_model import FusionModel
    model_train = FusionModel(model, num_classes, mode=mode)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    num_test = len(test_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)
    train_dataset = DataGenerator(train_lines, input_shape, False,False)
    val_dataset = DataGenerator(val_lines, input_shape, False,False)
    test_dataset = DataGenerator(test_lines, input_shape, False,False)


    train_sampler = None
    val_sampler = None
    test_sampler = None
    shuffle = True
    gen =     DataLoader(train_dataset, shuffle=False, batch_size=22, num_workers=0,
                        pin_memory=True,
                        drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=22, num_workers=0,
                         pin_memory=True,
                         drop_last=True, collate_fn=detection_collate, sampler=val_sampler)
    gen_test= DataLoader(test_dataset, shuffle=False, batch_size=22, num_workers=0,
                         pin_memory=True,
                         drop_last=True, collate_fn=detection_collate, sampler=test_sampler)
    if model_path != "":

        print('Load weights {}.'.format(model_path))

        model_dict = model_train.state_dict()

        pretrained_dict = torch.load(model_path, map_location="cpu")
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            # print("new_k:",new_k)
            if new_k in model_dict.keys() and np.shape(model_dict[new_k]) == np.shape(v):
                # print("k:", k, v)
                temp_dict[new_k] = v
                load_key.append(new_k)
            else:
                no_load_key.append(new_k)
        model_dict.update(temp_dict)
        model_train.load_state_dict(model_dict)

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    total_loss = 0
    total_accuracy = 0

    val_loss = 0
    val_accuracy = 0
    from tqdm import tqdm
    import torch.nn.functional as F
    print('Start Test')
    pbar = tqdm(total=1, desc=f'Epoch {1}/{1}', postfix=dict, mininterval=0.3)
    model_train.eval()
    accuracy_all=[]
    for iteration, batch in enumerate(gen_test):
        images, targets = batch

        print(images[0], targets[0])
        print(images[1], targets[1])
        outputs = model_train(images)
        print("outputs:", outputs)
        # ----------------------#
        #   计算损失
        # ----------------------#
        loss_value = nn.CrossEntropyLoss()(outputs, targets)
        total_loss += loss_value.item()
        accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
        total_accuracy += accuracy.item()
        accuracy_all.append(total_accuracy / (iteration + 1))
        pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                            'accuracy': total_accuracy / (iteration + 1)})
        pbar.update(1)
    pbar.close()
    print('Finish Test')
