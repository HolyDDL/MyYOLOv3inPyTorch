import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import yolov3
from CocoDataSet import COCOset
from tqdm import tqdm

# class Dataset():
    
#     def __init__(self, img_dir, ann_dir, batch_size = 1, num_workers = 0) -> None:
#         '''
#             Parameters:
#                 img_dir: 存储对应图片的目录
#                 ann_dir: 存储相应ann文件的json文件
#                 batch_size: 在img大小不同时, 只能=1载入
#                 num_workers: 读取数据进程数
#         '''
#         self.img_dir = img_dir
#         self.ann_dir = ann_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
    
#     def get_dataset(self):
#         # 如果入dataset的img的size相同, 可以直接用trans
#         trans = torchvision.transforms.Compose([torchvision.transforms.Resize((416,416)), torchvision.transforms.ToTensor()])
#         det = dset.CocoDetection(self.img_dir, self.ann_dir,transform=trans)
#         print(f'Number of samples: {len(det)}')
#         return det
    
#     def load_data(self):
#         data = self.get_dataset()
#         loaded = DataLoader(data, self.batch_size, num_workers=self.num_workers)
#         return loaded
    
#     # 获得每一个batch下的targets
#     def every_batch_targets(self, ori_targets) -> torch.Tensor: 
#         '''
#             Parameters:
#                 ori_targets: 传入每个数据集对应的targetInfo
#             Return:
#                 返回本batch下的targets[num_box, 5 = xyxy+class]
#         '''
#         # print(f'This img has {len(ori_targets)} bbox')
#         targets = []
#         for box in ori_targets:
#             '''
#                 box中描述框坐标的[lefttop_x, lefttop_y, w ,h]
#             '''
#             ltx, lty, rbx, rby = box['bbox'][0], box['bbox'][1], box['bbox'][0]+box['bbox'][2], box['bbox'][1]+box['bbox'][3]
#             label = box['category_id']
#             box_info = [ltx, lty, rbx, rby, label]
#             targets.append(box_info)
#         return torch.Tensor(targets).data.requires_grad_(False)
        
class YOLOLoss(nn.Module):

    def __init__(self, num_classes, input_size: tuple, device, coord_weight: float = 5, noobj_weight: float = 0.5) -> None:
        super().__init__()
        self.anch_13 = [[116,90],[156,198],[373,326]]
        self.anch_26 = [[30,61],[62,45],[59,119]]
        self.anch_52 = [[10,13],[16,30],[33,23]]
        self.num_classes = num_classes
        self.input_size = input_size
        self.box_attrs = 5 + num_classes
        self.num_anchors = len(self.anch_13)
        self.coord_weight = coord_weight
        self.noobj_weight = noobj_weight
        self.device = device

    def forward(self, attrs: list[torch.Tensor], targets: torch.Tensor, img_size: tuple):
        '''
            Parameters:
                attrs: 卷积后的特征图的list [attr0, attr1, attr2]
                targets: 该图片的targets信息 [num_bboxes, 5 = xyxy+class]
                img_size: (W, H)
        '''
        bs = targets.shape[0]
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        # print(f'Shape of the image is {img_size} (H, W)')
        # 得到real_box的mid_x, mid_y
        real_box = targets.clone().requires_grad_(False).to(self.device)
        real_box[:, 0] += (real_box[:, 2] / 2.)
        real_box[:, 1] += (real_box[:, 3] / 2.)
        all_attr_loss = 0
        for i, input in  enumerate(attrs):
            attr_loss = 0
            # 对三种特征图分别处理
            # 输入单个特征提取结果结果
            batch_size = input.size(0)
            height_size = input.size(2)
            width_size = input.size(3)
            num_grid = height_size
            if num_grid == 13:
                anchor = self.anch_13
            elif num_grid == 26:
                anchor = self.anch_26
            elif num_grid == 52:
                anchor = self.anch_52

            # 调整输入尺寸, 改为 1,3,52,52,85
            # batch_size, 3锚框下的, (52,52)某一个点的, 85维度张量
            pred = input.view(batch_size, self.num_anchors, self.num_classes+5, height_size, width_size).permute(0, 1, 3, 4, 2).contiguous()
            # 锚框的调整参数, 并将之归一化
            x_offset = torch.sigmoid(pred[..., 0])
            y_offset = torch.sigmoid(pred[..., 1])
            height_anchor = pred[...,2]
            width_anchor = pred[...,3]
            # 获得是否有物体置信度
            have_conf = torch.sigmoid(pred[..., 4])
            # 种类置信度张量
            class_conf = torch.sigmoid(pred[..., 5:])
            # 生成特征图网格
            # repeat沿着指定的维度重复tensor
            grid_x = torch.linspace(0, width_size - 1, width_size).repeat(height_size, 1).repeat(
                    batch_size * self.num_anchors, 1, 1).view(x_offset.shape).type(torch.FloatTensor).to(self.device)
            grid_y = torch.linspace(0, width_size - 1, width_size).repeat(height_size, 1).repeat(
                    batch_size * self.num_anchors, 1, 1).view(y_offset.shape).type(torch.FloatTensor).to(self.device)
            # 缩放比例, 即几个像素点对应特征图上的一个点
            scale_h = self.input_size[1] / height_size
            scale_w = self.input_size[0] / width_size
            # 特征层中锚框大小相对于一个特征点的大小
            scale_anchors = [(anchor_w / scale_w, anchor_h / scale_h) for anchor_w, anchor_h in anchor]
            # 生成特征图中锚框的大小
            anchor_w = torch.FloatTensor(scale_anchors).index_select(1,torch.LongTensor([0])).to(self.device)
            anchor_h = torch.FloatTensor(scale_anchors).index_select(1,torch.LongTensor([1])).to(self.device)
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, height_size * width_size).view(width_anchor.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, height_size * width_size).view(height_anchor.shape)
            # 调整锚框
            # 生成一个只有调整前4参数的张量 1,3,52,52,4用于生成调整锚框
            # torch.Tensor.data == torch.Tensor.detach(), 截断梯度, 在传播时对detach不进行参数更新
            pred_box = torch.FloatTensor(pred[..., :4].shape).to(device)
            pred_box[..., 0] = x_offset + grid_x
            pred_box[..., 1] = y_offset + grid_y
            pred_box[..., 2] = torch.exp(width_anchor) + anchor_w
            pred_box[..., 3] = torch.exp(height_anchor) + anchor_h
            _scale = torch.Tensor([width_size, height_size, width_size, height_size]).type(torch.FloatTensor).to(device)
            # 归一化
            pred_box /= _scale
            # 得到预测类
            class_conf, class_pred = torch.max(class_conf, dim=-1, keepdim=True)
            have_conf = torch.unsqueeze(have_conf, dim=-1)
            # 得到解码框 shape = [batch_size, num_anchors, 13, 13, 7 = xywh+have_conf+class_conf+class_pred]
            decode_boxes = torch.cat((pred_box, have_conf, class_conf, class_pred), dim=-1)
            grid_x, grid_y =  self.choose_pb(decode_boxes, real_box, num_grid)
            pb_loss, pb_ciou = self.pb_loss(grid_x, grid_y, decode_boxes, real_box)
            nt_loss = self.nt_loss(decode_boxes,real_box,pb_ciou)
            attr_loss = nt_loss + pb_loss
            all_attr_loss += attr_loss
        return all_attr_loss

    def nt_loss(self, decode_boxes, real_box, pb_ciou):
        # real_box : [bs, 5]
        real_xyxy = torch.clone(real_box).requires_grad_(False)
        real_xyxy[:, 0] -= (real_xyxy[:, 2] / 2.)
        real_xyxy[:, 1] -= (real_xyxy[:, 3] / 2.)
        real_xyxy[:, 2] += real_xyxy[:, 0]
        real_xyxy[:, 3] += real_xyxy[:, 1]
        bs_loss = 0
        for bs, each_ciou in enumerate(pb_ciou):
            each = torch.clone(decode_boxes[bs]).view(-1, 7)
            each[:, 0] -= (each[:, 2] / 2.)
            each[:, 1] -= (each[:, 3] / 2.)
            each[:, 2] += each[:, 0]
            each[:, 3] += each[:, 1]
            real = real_xyxy[bs, :4].unsqueeze(dim=0)
            iou = torchvision.ops.box_iou(real, each[:, :4]).squeeze().to(self.device)
            iou += 1e-7
            c_iou = each[:, 5] * each[:, 4].squeeze() * iou
            temp = 1 - c_iou
            nt_ciou = torch.log(1 - c_iou).sum()
            bs_pt_ciou = torch.log(1 - each_ciou).sum()
            conf_loss = - self.noobj_weight * (nt_ciou - bs_pt_ciou)
            if self.check_loss(conf_loss):
                print('In noobj_conf_loss')
                exit()
            bs_loss += conf_loss
        return bs_loss 

    def pb_loss(self, grid_x, grid_y, decode_boxes, real_box: torch.Tensor):
        # real_box : [bs, 5]
        real_xyxy = torch.clone(real_box).requires_grad_(False)
        real_xyxy[:, 0] -= (real_xyxy[:, 2] / 2.)
        real_xyxy[:, 1] -= (real_xyxy[:, 3] / 2.)
        real_xyxy[:, 2] += real_xyxy[:, 0]
        real_xyxy[:, 3] += real_xyxy[:, 1]
        bs_loss = 0
        pb_ciou = []
        for bs, (x, y) in enumerate(zip(grid_x, grid_y)):
            # 每一个bs下的坐标为(x, y)
            # 得到each.shape = [num_anchors, 7] 7=xywh+have_conf+class_conf+class_pred
            # 意义为负责检测该物体的网格内的锚框们
            each = torch.clone(decode_boxes[bs, :, x, y, :])
            xywh_loss = self.xywh_loss(each, real_box[bs, :])
            each[:, 0] -= (each[:, 2] / 2.)
            each[:, 1] -= (each[:, 3] / 2.)
            each[:, 2] += each[:, 0]
            each[:, 3] += each[:, 1]
            # iou = [num_anchors]
            real = real_xyxy[bs, :4].unsqueeze(dim=0)
            iou = torchvision.ops.box_iou(real, each[:, :4]).squeeze().to(self.device)
            iou += 1e-7
            # c_iou = pt_box[:, 5].squeeze() * pt_box[:, 4].squeeze() * iou
            c_iou = each[:, 5] * each[:, 4].squeeze() * iou
            pb_ciou.append(c_iou)
            conf_loss = - torch.log(c_iou).sum()
            if self.check_loss(conf_loss):
                print('In pb_conf_loss')
                exit() 
            pos_weight = torch.ones(3).requires_grad_(False).to(self.device)
            real_cat = (real_box[bs, 4].repeat(3) / self.num_classes).float()
            bce = nn.BCELoss(pos_weight,reduction='sum')
            pred_cat = (each[:, 6] / self.num_classes).float()
            class_loss = bce(pred_cat, real_cat)
            if self.check_loss(class_loss):
                print('In cod_closs_loss')
                exit()
            bs_loss += xywh_loss
            bs_loss += conf_loss
            bs_loss += class_loss
        return bs_loss, pb_ciou
    def xywh_loss(self, each: torch.Tensor, real_box: torch.Tensor):
        '''
            Parameters:
                real_box: 每一个bs下的真实坐标 [5], 注意传入real_box而非real_xyxy
        '''
        xx = torch.square(each[:, 0] - real_box[0])
        yy = torch.square(each[:, 1] - real_box[1])
        ww = torch.square(torch.sqrt(each[:, 2]) - torch.sqrt(real_box[2]))
        hh = torch.square(torch.sqrt(each[:, 3]) - torch.sqrt(real_box[3]))
        loss = self.coord_weight * (xx.sum() + yy.sum() + ww.sum() + hh.sum())
        if self.check_loss(loss):
            print('In pb_xywh_loss')
            exit()
        return loss

    def choose_pb(self, decode_boxes: torch.Tensor, real_boxes: torch.Tensor, num_grid: int):
        '''
            Return:
                返回sample对应在网格中的坐标
                grid_X: [bs]
                grid_y: [bs]
        '''
        grid_x = (real_boxes[:, 0] * num_grid).int().requires_grad_(False)
        grid_y = (real_boxes[:, 1] * num_grid).int().requires_grad_(False)
        return grid_x, grid_y
            # for batch, batch_boxes in enumerate(real_box):
            #     batch_pbox_loss = 0
            #     batch_nbox_loss = 0
            #     batch_pbox_idx = []
            #     for i, sample in enumerate(batch_boxes):
            #         # sample.shape = 5 = xywh + class
            #         positive_idx = self.choose_pb_idx(sample, num_grid)
            #         cod_loss = self.coord_loss(decode_boxes[batch, :, positive_idx[0], positive_idx[1], :], sample)
            #         batch_pbox_loss += cod_loss
            #         batch_pbox_idx.append(positive_idx)
            #     for x in range(decode_boxes.shape[2]):
            #         for y in range(decode_boxes.shape[3]):
            #             noobj = True
            #             for idx in batch_pbox_idx:
            #                 if x == idx[0] and y == idx[1]:
            #                     noobj = False
            #                     break
            #             if noobj:
            #                 noobj_loss = self.noobj_loss(decode_boxes[batch, :, x, y, :], batch_boxes)
            #                 if self.check_loss(noobj_loss):
            #                     print('in noobj_loss')
            #                 batch_nbox_loss += noobj_loss
            #     # print(batch_nbox_loss)
            #     # print(batch_pbox_loss)
            #     batch_loss += batch_pbox_loss
            #     batch_loss += batch_nbox_loss
            # # print(batch_loss)
            # all_attr_loss += batch_loss
            # # print(all_attr_loss)
        # return all_attr_loss
    
    # 计算负样本的loss
    def noobj_loss(self, ori_nt_box: torch.Tensor, ori_real_boxes: torch.Tensor):
        '''
            Parameters:
                ori_nt_box: 指定的单个负样本 [num_anchors, 7]
                orI_real_box: 该batch下的全部真实样本 [num, 5]
        '''
        nt_box = torch.clone(ori_nt_box).requires_grad_(True)
        real_boxes = torch.clone(ori_real_boxes).requires_grad_(False)
        for each in [nt_box, real_boxes]:
            # 将xywh变为xyxy
            each[:, 0] -= (each[:, 2] /2.)
            each[:, 1] -= (each[:, 3] /2.)
            each[:, 2] += each[:, 0]
            each[:, 3] += each[:, 1]
        # iou.shape = [num_real_boxes, num_anchors]
        iou = torchvision.ops.box_iou(real_boxes[:,:4], nt_box[:,:4])
        iou += 1e-7
        c_iou = torch.mul(iou, nt_box[:, 5].squeeze() * nt_box[:, 4].squeeze())
        conf_loss = torch.log(1 - c_iou)
        return - conf_loss.sum() * self.noobj_weight
                    
    # 计算每一个正样本所在的网格内的三个锚框的loss
    def coord_loss(self, pt_box, real_box):
        '''
            Parameters:
                pt_box: 正样本box, shape = [num_anchors, 7]
                real_box: 真正box, shape= [5]
        '''
        x_loss = torch.square((pt_box[:, 0] - real_box[0])).sum()
        y_loss = torch.square((pt_box[:, 1] - real_box[1])).sum()
        w_loss = torch.square(torch.sqrt(pt_box[:, 2])-torch.sqrt(real_box[2])).sum()
        h_loss = torch.square(torch.sqrt(pt_box[:, 3])-torch.sqrt(real_box[3])).sum()
        lambda_coord_loss =  self.coord_weight * (x_loss + y_loss + w_loss + h_loss)
        if self.check_loss(lambda_coord_loss):
            print('in lambda_coord_loss')
        conf_loss = self.cod_conf_loss(real_box, pt_box)
        if self.check_loss(conf_loss):
            print('in cod_conf_loss')
        class_loss = self.cod_class_loss(real_box, pt_box)
        if self.check_loss(class_loss):
            print('in cod_class_loss')
        return lambda_coord_loss + conf_loss + class_loss

    def check_loss(self, loss: torch.Tensor) -> bool:
        if float(loss) < 0:
            print('Loss <0')
            return True
        elif loss.isnan():
            print('Loss is nan')
            return True
        elif loss.isinf():
            print('Loss is inf')
            return True
        return False

    def cod_class_loss(self, sample: torch.Tensor, pt_box: torch.Tensor):
        real_box = torch.clone(sample).requires_grad_(False)
        class_sub = pt_box[:, -1]-real_box[-1]
        for i,each in enumerate(class_sub):
            if each == 0:
                class_sub[i] = 1     
        loss = torch.log(torch.square(class_sub))
        return loss.sum()
    
    def cod_conf_loss(self, sample: torch.Tensor, ori_pt_box: torch.Tensor):
        real_box = torch.clone(sample).requires_grad_(False)
        pt_box = torch.clone(ori_pt_box).requires_grad_(True)
        real_box = real_box.unsqueeze(0)
        for each in [pt_box, real_box]:
            # 由xywh变为xyxy
            each[:, 0] -= (each[:, 2] /2.)
            each[:, 1] -= (each[:, 3] /2.)
            each[:, 2] += each[:, 0]
            each[:, 3] += each[:, 1]
        iou = torchvision.ops.box_iou(pt_box[:, :4], real_box[:, :4]).squeeze().to(self.device)
        for each in iou:
            if each == 0:
                iou += 1e-7
                break
        for each in iou :
            if each == 1:
                iou -= 1e-7
                break
        c_iou = pt_box[:, 5].squeeze() * pt_box[:, 4].squeeze() * iou
        conf_loss = torch.log(c_iou)
        return - conf_loss.sum()
    
    # 对于每一个真实样本进行选择正样本锚框
    def choose_pb_idx(self, real_sample: torch.Tensor, num_grid: int) -> tuple:
        '''
            Parameters:
                real_sample: 真实样本框 [5]
                num_grid: 图片上的网格数
            Return:
                真实样本所在的格子坐标tuple
        '''
        real_grid = real_sample.clone().requires_grad_(False)
        real_grid[:-1] *= num_grid
        x_gird = int(real_grid[0])
        y_grid = int(real_grid[1])
        return (x_gird, y_grid)    

class DrawTrainPic():

    def __init__(self, save_dir) -> None:
        '''
            Parameters:
                save_dir: 存储画完框的图片的目录
        '''
        self.save_dir = save_dir
    
    def draw(self, img, targets, names: dict, new_pic_name):
        '''
            Parameters:
                img: 需要标注框的image
                targets: 该img的bbox相关参数, [num_bboxes, 5 = ltx, lty, rbx, rby, label]
                names: label对应class号的解读
                new_pic_name: 存储画完框以后的图片的名字
        '''
        from names import LoadNames
        for target in targets:
            lftx = target[0]
            lfty = target[1]
            rtbx = target[2]
            rtby = target[3]
            color = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
            frame = cv2.rectangle(img, (int(lftx), int(lfty)),(int(rtbx), int(rtby)),color,2)
            name = LoadNames(int(target[-1]), 'train')
            font_size = cv2.getTextSize(name,0,0.4,1)
            frame = cv2.rectangle(frame, (int(lftx), int(lfty-font_size[0][1])), (int(lftx+font_size[0][0]), int(lfty)), color, -1)
            frame = cv2.putText(frame, name, (int(lftx), int(lfty)),fontFace=0, fontScale=0.4, thickness=1,
                            color=(color[2], color[0], color[1]))
            cv2.imwrite(f'{self.save_dir}{os.sep}{new_pic_name}.jpg', frame)
        print(f'The picture has saved as "{self.save_dir}{os.sep}{new_pic_name}.jpg"')


if __name__ == '__main__':
    import os

    import cv2
    
    # 判断设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to train')
    print(f'The number of GPU is {torch.cuda.device_count()}')
    
    # 加载数据集
    batch_size = 4
    trans = torchvision.transforms.Compose([torchvision.transforms.Resize([416,416]), torchvision.transforms.ToTensor()])
    data_val = COCOset('annotations_trainval2017','val2017','val', trans)
    valdata = DataLoader(data_val, batch_size)
    data_train = COCOset('annotations_trainval2017','val2017','val', trans)
    traindata = DataLoader(data_train, batch_size)
    # tensorboard
    writer = SummaryWriter('logs')
    # 训练参数初始化
    lr = 1e-2
    
    epoch = 10
    # 读取图像与标准化 #    
    input_size = (416, 416)
    num_classes = 80
    num_anchors = 3
    At = yolov3.Attribute(num_classes,num_anchors)
    At.to(device)
    dark = yolov3.DarkNet()
    dark.to(device)
    yolo = yolov3.Yolov3()
    yolo.to(device)
    At.train()
    dark.train()
    yolo.train()
    optim_at = torch.optim.Adam(At.parameters(), lr)
    optim_dark = torch.optim.Adam(dark.parameters(), lr)
    optim_yolo = torch.optim.Adam(yolo.parameters(), lr)
    for ep in range(epoch):
        ep_loss = 0
        with tqdm(total=len(traindata)) as pbar:
            pbar.set_description(f'Training in epoch {ep}')
            for batch, data in enumerate(traindata):
                # 每一个batch下的数据分开处理
                imgs, targets = data
                img_size = (imgs.shape[2], imgs.shape[3])
                # ----- 特征提取 ----- #
                # --- 重整图像 --- # 
                if (imgs.shape[-2], imgs.shape[-1]) == input_size:
                    input = imgs
                else:
                    print('Input size Error!')
                    exit()
                input = input.to(device)
                # --------------- #
                d1, d2, d3 = dark(input)
                out = yolo(d1,d2,d3)

                attr0 = At(out[0])
                attr1 = At(out[1])
                attr2 = At(out[2])
                attrs = [attr0, attr1, attr2]
                # ------------------- #
                # 损失函数 #
                Loss = YOLOLoss(num_classes, input_size, device)
                Loss.to(device)
                loss_value = Loss.forward(attrs, targets, img_size)
                ep_loss += loss_value
                optim_at.zero_grad()
                optim_dark.zero_grad()
                optim_yolo.zero_grad()
                loss_value.backward()
                optim_yolo.step()
                optim_dark.step()
                optim_at.step()
                # tensorboard
                writer.add_scalar('batch_train_loss',loss_value, batch)
                pbar.update(1)
        writer.add_scalar('epoch_train_loss', ep_loss)
        print(f'In epoch {ep}, train loss is {ep_loss}')
        # 验证
        At.eval()
        dark.eval()
        yolo.eval()
        with torch.no_grad():
            all_val_loss = 0
            for batch, data in enumerate(valdata):
                imgs, targets = data
                img_size = (imgs.shape[2], imgs.shape[3])
                # ----- 特征提取 ----- #
                # --- 重整图像 --- # 
                if (imgs.shape[-2], imgs.shape[-1]) == input_size:
                    input = imgs
                else:
                    print('Input size Error!')
                    exit()
                input = input.to(device)
                # --------------- #
                d1, d2, d3 = dark(input)
                out = yolo(d1,d2,d3)

                attr0 = At(out[0])
                attr1 = At(out[1])
                attr2 = At(out[2])
                attrs = [attr0, attr1, attr2]
                # ------------------- #
                # 损失函数 #
                Loss = YOLOLoss(num_classes, input_size, device)
                Loss.to(device)
                loss_value = Loss.forward(attrs, targets, img_size)
                all_val_loss += loss_value
            writer.add_scalar('Val_epoch_loss', all_val_loss)
            print(f'In epoch {ep}, val loss is {all_val_loss}')
        torch.save(At.state_dict(), r'models\LAST_AT.pt')
        torch.save(dark.state_dict(), r'models\LAST_dark.pt')
        torch.save(yolo.state_dict(), r'models\LAST_yolo.pt')
        if ep == 0:
            torch.save(At.state_dict(), r'models\BEST_AT.pt')
            torch.save(dark.state_dict(), r'models\BEST_dark.pt')
            torch.save(yolo.state_dict(), r'models\BEST_yolo.pt')
            judge_loss = all_val_loss
            print(f'Now the best epoch is {ep}')
        elif all_val_loss < judge_loss:
            torch.save(At.state_dict(), r'models\BEST_AT.pt')
            torch.save(dark.state_dict(), r'models\BEST_dark.pt')
            torch.save(yolo.state_dict(), r'models\BEST_yolo.pt')
            judge_loss = all_val_loss
            print(f'Now the best epoch is {ep}')
    writer.close()
            # # 如需要打印训练的框在图上的位置
            # dra = DrawTrainPic(r'save')
            # image = cv2.imread(r'val2017\000000000139.jpg')
            # dra.draw(image, targets, names,'new_pic')