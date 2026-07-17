from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_metric_learning import losses as metric_loss
from lib.infoNCE import *
import warnings
import numpy as np
import pickle

np.set_printoptions(precision=3)
import time
import os

# Set your GPU device here (default: 0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd

from lib.Uncertainty import *
from lib.Memory import *
from lib.Memory_new import *
from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config

from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from lib.FReMuRe import FReMuRe
from lib.ds_track import get_sequence

from datetime import datetime
from collections import Counter

def get_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H%M%S")
    return formatted_time

def compute_freq_tensor(dataset1, dataset2, rel_type='attention', num_classes=3):
    """
    Compute frequency tensor for a given relation type from dataset.
    Args:
        dataset: AG_dataset_train
        rel_type: 'attention', 'spatial', or 'contacting'
        num_classes: number of relation categories
    Returns:
        freq_tensor: torch.Tensor of shape [num_classes]
    """
    all_labels = []
    for ann in dataset1.gt_annotations:
        for frame_ann in ann:
            for obj_pair in frame_ann:
                if rel_type + '_relationship' in obj_pair:
                    all_labels.extend(obj_pair[rel_type + '_relationship'].tolist())
                else:
                    continue
    for ann in dataset2.gt_annotations:
        for frame_ann in ann:
            for obj_pair in frame_ann:
                if rel_type + '_relationship' in obj_pair:
                    all_labels.extend(obj_pair[rel_type + '_relationship'].tolist())
                else:
                    continue
    counter = Counter(all_labels)
    freq_list = [counter[i] for i in range(num_classes)]
    freq_array = np.array(freq_list, dtype=np.float32)
    freq_array = freq_array / (freq_array.sum() + 1e-12)  # Normalize to sum to 1
    freq_tensor = torch.tensor(freq_array)
    return freq_tensor

warnings.filterwarnings("ignore", category=DeprecationWarning)  # 表示忽略警告，即不会显示任何警告信息
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  # 过滤 NumPy 中的弃用警告
"""------------------------------------some settings----------------------------------------"""
conf = Config()
print('The Ceckpoint 保存在:', conf.save_path)  # output/
if not os.path.exists(conf.save_path):  # False 如果路径不存在则创建
    os.mkdir(conf.save_path)
conf.save_path = conf.save_path + get_time() + '-' + conf.mode + '/'
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)

model_save_path = conf.save_path + 'models/'  # output/2025-xx-xx-xxxxxx-predcls/models/

arg_file = open(conf.save_path + 'configurations.txt', mode='w')
print('---------------------------配置参数---------------------------\n', flush=True)
for i in conf.args:
    str_print = '{} : {}'.format(i, conf.args[i])
    print(str_print, flush=True)
    arg_file.write(str_print + '\n')


print('---------------------------数据集导入---------------------------\n', flush=True)
AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")

print('---------------------------构建模型---------------------------\n', flush=True)

# 冻结参数，只做前向推理：
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True,
                           mode=conf.mode).to(device=gpu_device)
object_detector.eval()
freq_attention = None
freq_spatial = None
freq_contact = None
if conf.freq:
    freq_attention = compute_freq_tensor(AG_dataset_train, AG_dataset_test, rel_type='attention', num_classes=3).to(gpu_device)
    freq_spatial = compute_freq_tensor(AG_dataset_train, AG_dataset_test, rel_type='spatial', num_classes=6).to(gpu_device)
    freq_contact = compute_freq_tensor(AG_dataset_train, AG_dataset_test, rel_type='contacting', num_classes=17).to(gpu_device)

model = FReMuRe(mode=conf.mode,
                attention_class_num=len(AG_dataset_train.attention_relationships),  # 3
                spatial_class_num=len(AG_dataset_train.spatial_relationships),  # 6
                contact_class_num=len(AG_dataset_train.contacting_relationships),  # 17
                obj_classes=AG_dataset_train.object_classes,
                enc_layer_num=conf.enc_layer,  # 1
                dec_layer_num=conf.dec_layer,  # 3
                obj_mem_compute=conf.obj_mem_compute,  # 计算目标记忆幻觉[分离/联合/无]
                rel_mem_compute=conf.rel_mem_compute,  # 计算关系记忆幻觉[分离/联合/无]
                take_obj_mem_feat=conf.take_obj_mem_feat,  # false
                mem_fusion=conf.mem_fusion,  # late
                selection=conf.mem_feat_selection,  # manual
                selection_lambda=conf.mem_feat_lambda,  # 0.5
                obj_head=conf.obj_head,  # liner
                rel_head=conf.rel_head,  #
                K=conf.K,  # 6，混合模型数量
                tracking=conf.tracking,
                freq_attention=freq_attention,
                freq_spatial=freq_spatial,
                freq_contact=freq_contact).to(device=gpu_device)  # false

print('---------------------------构建evaluator---------------------------\n', flush=True)
evaluator = BasicSceneGraphEvaluator(mode=conf.mode,
                                     AG_object_classes=AG_dataset_train.object_classes,
                                     AG_all_predicates=AG_dataset_train.relationship_classes,
                                     AG_attention_predicates=AG_dataset_train.attention_relationships,
                                     AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                     AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                     iou_threshold=0.5,
                                     # output_dir = conf.save_path,
                                     constraint='with')

# loss function, default Multi-label margin loss
print('---------------------------构建loss function---------------------------\n', flush=True)
weights = torch.ones(len(model.obj_classes))
weights[0] = conf.eos_coef
if conf.obj_head != 'gmm':
    ce_loss_obj = nn.CrossEntropyLoss(weight=weights.to(device=gpu_device), reduction='none')
else:
    ce_loss_obj = nn.NLLLoss(weight=weights.to(device=gpu_device), reduction='none')

if conf.rel_head != 'gmm':
    ce_loss_rel = nn.CrossEntropyLoss(reduction='none')
else:
    ce_loss_rel = nn.NLLLoss(reduction='none')  # 负对数似然损失

if conf.mlm:
    mlm_loss = nn.MultiLabelMarginLoss(reduction='none')
else:
    bce_loss = nn.BCELoss(reduction='none')

if conf.obj_con_loss == 'euc_con':
    con_loss = metric_loss.ContrastiveLoss(pos_margin=0, neg_margin=1)
    # con_loss = EucNormLoss()
    # con_loss.train()
elif conf.obj_con_loss == 'info_nce':
    con_loss = SupConLoss(temperature=0.1)
    con_loss.train()

# optimizer
print('---------------------------构建优化器---------------------------\n', flush=True)
for name, value in model.named_parameters():
    if 'object_classifier' in name and conf.mode == 'predcls':
        value.requires_grad = False  # 停止张量的梯度计算，适用于冻结模型参数或中间变量不需要参与梯度计算的情况。

# learned_params = [
#         {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
#         # {
#         #     "params": [p for n, p in model.named_parameters() if "object_classifier" in n and p.requires_grad],
#         #     "lr": 1e-5,
#         # },
#     ]

learned_params = model.parameters()
optimizer = None
if conf.optimizer == 'adamw':
    optimizer = AdamW(learned_params, lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(learned_params, lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(learned_params, lr=conf.lr, momentum=0.9, weight_decay=0.01)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
                              threshold_mode="abs", min_lr=1e-7)

# some parameters
tr = []  # 存loss的
best_recall = 0
best_Mrecall = 0

# 记录日志
if not conf.no_logging:
    log = open(conf.save_path + 'logs.txt', mode='a')  # 新写入的内容会添加到文件末尾
    log.write('*' * 60 + '\n')
    log_val = open(conf.save_path + 'log_val.txt', mode='a')
    log_val.write('*' * 60 + '\n')

print('---------------------------开始训练---------------------------\n', flush=True)
for epoch in range(conf.nepoch):  # 10
    unc_vals = uncertainty_values(obj_classes=len(model.obj_classes),  # 37
                                  attention_class_num=model.attention_class_num,  # 3
                                  spatial_class_num=model.spatial_class_num,  # 6
                                  contact_class_num=model.contact_class_num)  # 17
    model.train()
    object_detector.is_train = True

    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    max_batch = {}
    for b in range(len(dataloader_train)):
        data = next(train_iter)
        im_data = copy.deepcopy(data[0].to(device=gpu_device))
        im_info = copy.deepcopy(data[1].to(device=gpu_device))
        gt_boxes = copy.deepcopy(data[2].to(device=gpu_device))
        num_boxes = copy.deepcopy(data[3].to(device=gpu_device))
        gt_annotation = AG_dataset_train.gt_annotations[data[4]]


        # prevent gradients to FasterRCNN
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

        if conf.tracking:
            get_sequence(entry, gt_annotation, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)

        pred = model(entry, phase='train', unc=False)

        if conf.obj_unc or conf.rel_unc or conf.obj_mem_compute or conf.rel_mem_compute:
            uncertainty_computation(data, AG_dataset_train,
                                    object_detector, model, unc_vals, gpu_device,
                                    conf.save_path,
                                    obj_unc=conf.obj_unc, obj_mem=conf.obj_mem_compute,
                                    background_mem=False, rel_unc=conf.rel_unc,
                                    tracking=conf.tracking)

        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]

        if conf.rel_head == 'gmm':
            attention_distribution = torch.log(attention_distribution + 1e-12)

        if conf.obj_head == 'gmm' and conf.mode != 'predcls':
            pred['distribution'] = torch.log(pred['distribution'] + 1e-12)

        attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(
            device=attention_distribution.device).squeeze()
        if conf.mlm:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(
                device=attention_distribution.device)
            contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(
                device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

        else:
            # bce loss
            spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(
                device=attention_distribution.device)
            contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(
                device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contacting_gt"][i]] = 1

        losses = {}
        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            losses['object_loss'] = ce_loss_obj(pred['distribution'], pred['labels'])
            loss_weighting = conf.obj_loss_weighting
            if loss_weighting is not None:
                num = torch.exp(unc_vals.obj_batch_unc[loss_weighting].sum(-1))
                den = num.sum()
                weights = 1 + (num / den).to(device=gpu_device)
                losses['object_loss'] = weights * losses['object_loss']
            losses['object_loss'] = losses['object_loss'].mean()
            if conf.obj_con_loss:
                losses['object_contrastive_loss'] = conf.lambda_con * con_loss(pred['object_mem_features'],
                                                                               pred['labels'])

        losses["attention_relation_loss"] = ce_loss_rel(attention_distribution, attention_label)
        if conf.mlm:
            losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
            losses["contacting_relation_loss"] = mlm_loss(contact_distribution, contact_label)

        else:
            losses["spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label)
            losses["contacting_relation_loss"] = bce_loss(contact_distribution, contact_label)

        loss_weighting = conf.rel_loss_weighting

        # === Frequency-based weighting (tail class emphasis) ===
        freq_keys = {
            'attention': 'freq_attention',
            'spatial': 'freq_spatial',
            'contacting': 'freq_contact'
        }

        for rel in ['attention', 'spatial', 'contacting']:
            freq_key = freq_keys[rel]
            freq_tensor = entry.get(freq_key, None)  # 取出当前关系的 freq_tensor
            loss_key = rel + '_relation_loss'

            # if freq_tensor is not None:
            #     loss_val = losses[loss_key]  # [B] or [B, C]
            #     N = loss_val.shape[0]
            #
            #     freq_weight = torch.log(1.0 / (freq_tensor + 1e-6))
            #
            #     # === 自动对齐长度 ===
            #     if freq_weight.shape[0] != N:
            #         freq_weight = F.pad(freq_weight, (0, N - freq_weight.shape[0]))[:N]  # [N]
            #
            #     # === 自动匹配维度并加权 ===
            #     if loss_val.dim() == 2:
            #         freq_weight = freq_weight.unsqueeze(1)  # [N, 1]
            #     losses[loss_key] = loss_val * freq_weight.to(loss_val.device)

            if loss_weighting is not None:
                num = torch.exp(unc_vals.rel_batch_unc[rel][loss_weighting].sum(-1))
                den = num.sum() + 1e-12
                weights = 1 + (num / den).to(device=gpu_device)

                if rel != 'attention':
                    weights = weights.unsqueeze(-1).repeat(1, losses[rel + '_relation_loss'].shape[-1])

                losses[rel + '_relation_loss'] = weights * losses[rel + '_relation_loss']
            losses[rel + '_relation_loss'] = losses[rel + '_relation_loss'].mean()

        optimizer.zero_grad()
        loss = sum(losses.values())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()
        losses['total_loss'] = loss
        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
        log_iter = conf.log_iter
        if (b + 1) % log_iter == 0 and (b + 1) >= log_iter:
            time_per_batch = (time.time() - start) / log_iter
            str_print = "\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                      time_per_batch,
                                                                                      len(dataloader_train) * time_per_batch / 60)
            print(str_print, flush=True)

            if not conf.no_logging:
                log.write(str_print + '\n')

                mn = pd.concat(tr[-log_iter:], axis=1).mean(1)
                print(mn, flush=True)
                for k in list(mn.keys()):
                    str_print = '{} : {:5f}'.format(k, mn[k])
                    log.write(str_print + '\n')
            # mn.to_csv(os.path.join(conf.save_path, 'training_loss.csv'),header=None)
            start = time.time()

    if not conf.no_logging:
        if conf.obj_unc or conf.rel_unc:
            if not os.path.exists(conf.save_path + 'epoch_wise_cls_unc/'):
                os.mkdir(conf.save_path + 'epoch_wise_cls_unc/')
            with open(conf.save_path + 'epoch_wise_cls_unc/cls_unc_obj_{}.pkl'.format(epoch), 'wb') as file:
                pickle.dump(unc_vals.cls_obj_uc, file)
            with open(conf.save_path + 'epoch_wise_cls_unc/cls_unc_rel_{}.pkl'.format(epoch), 'wb') as file:
                pickle.dump(unc_vals.cls_rel_uc, file)

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        for b in range(len(dataloader_test)):
            data = next(test_iter)

            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            if conf.tracking:
                get_sequence(entry, gt_annotation, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)
            pred = model(entry, phase='test', unc=False)
            evaluator.evaluate_scene_graph(gt_annotation, pred)
        print('-----------' * 3, flush=True)

    recall = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    mrecall = evaluator.calc_mrecall()[20]
    if not conf.no_logging:
        log_val.write('epoch {} validation results:'.format(epoch) + '\n')
        evaluator.print_stats(log_val)
    if recall > best_recall:
        best_recall = recall
        str_print = 'new best recall of {} at epoch {}'.format(best_recall, epoch)
        if epoch > 0 and conf.rel_mem_compute is not None:
            if len(model.object_classifier.obj_memory) == 0:
                object_memory = []
            else:
                object_memory = model.object_classifier.obj_memory.to('cpu')
            rel_memory = model.rel_memory
            if len(rel_memory) != 0:
                rel_memory = {k: rel_memory[k].to('cpu') for k in rel_memory.keys()}
        else:
            object_memory = []
            rel_memory = []
        print(str_print + '\n', flush=True)
        if not conf.no_logging:
            log_val.write(str_print + '\n')
            torch.save({"state_dict": model.state_dict(),
                        'object_memory': object_memory,
                        'rel_memory': rel_memory}, os.path.join(model_save_path, "best_recall_model.tar".format(epoch)))
    if mrecall > best_Mrecall:
        best_Mrecall = mrecall
        str_print = 'new best Mrecall of {} at epoch {}'.format(best_Mrecall, epoch)
        print(str_print + '\n', flush=True)
        if not conf.no_logging:
            log_val.write(str_print + '\n')
            if epoch > 0 and conf.rel_mem_compute is not None:
                object_memory = model.object_classifier.obj_memory.to('cpu')
                rel_memory = model.rel_memory
                rel_memory = {k: rel_memory[k].to('cpu') for k in rel_memory.keys()}
            else:
                object_memory = []
                rel_memory = []
            torch.save({"state_dict": model.state_dict(),
                        'object_memory': object_memory,
                        'rel_memory': rel_memory},
                       os.path.join(model_save_path, "best_Mrecall_model.tar".format(epoch)))
    evaluator.reset_result()
    scheduler.step(mrecall)

    if conf.rel_mem_compute or conf.obj_mem_compute:
        print('computing memory \n', flush=True)
        rel_class_num = {'attention': model.attention_class_num,
                         'spatial': model.spatial_class_num,
                         'contacting': model.contact_class_num}
        if conf.tracking:
            obj_feature_dim = 2048 + 200 + 128
        else:
            obj_feature_dim = 1024
        if conf.freq:
            rel_memory, obj_memory = memory_computation(unc_vals, conf.save_path, rel_class_num={
                'attention': model.attention_class_num, 'spatial': model.spatial_class_num,
                'contacting': model.contact_class_num}, obj_class_num=len(model.obj_classes),
                obj_mem=conf.obj_mem_compute, obj_unc=conf.obj_unc, rel_weight_type='al', obj_weight_type='al',
                freq_obj_tensor=None, freq_rel_tensor_dict={'attention': freq_attention, 'spatial': freq_spatial,
                                                            'contacting': freq_contact},
                temperature=0.5)  # 温度越小，强化作用越强
        else:
            rel_memory, obj_memory = memory_computation(unc_vals, conf.save_path, rel_class_num, len(model.obj_classes),
                                                        obj_feature_dim=obj_feature_dim, rel_feature_dim=1936,
                                                        obj_weight_type=conf.obj_mem_weight_type,
                                                        rel_weight_type=conf.rel_mem_weight_type,
                                                        obj_mem=conf.obj_mem_compute, obj_unc=conf.obj_unc,
                                                        include_bg_mem=False)

        model.object_classifier.obj_memory = obj_memory.to(gpu_device)
        model.rel_memory = {k: rel_memory[k].to(gpu_device) for k in rel_memory.keys()}
