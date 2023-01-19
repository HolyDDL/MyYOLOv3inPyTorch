import torch
import numpy as np
from torchvision.ops import nms
import yolov3
import cv2
import os

class DecodeBox():
    
    def __init__(self, num_classes, device, input_size: tuple, num_anchors = 3) -> None:
        '''
            Parameters:
                num_classes: 种类数量
                input_size: resize以后的标准图片大小
                num_anchors: 单个网格生成锚框数量
        '''
        # 将原图划分的方框数目对应的锚框大小, 每个点仅有三个框
        self.anch_13 = [[116,90],[156,198],[373,326]]
        self.anch_26 = [[30,61],[62,45],[59,119]]
        self.anch_52 = [[10,13],[16,30],[33,23]]
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.input_size = input_size
        self.device = device

    # 一次输入一张图片的[13,26,52]其中之一的分割网格下的特征图
    # 用于解码锚框, 返回锚框以及其对应的锚框参数和类别预测概率张量   
    def decode_box(self, all_input: list):
        '''
            Parameters:
                all_input: 卷积后的特征图的list [attr0, attr1, attr2]
        '''
        outputs = None
        for i, input in  enumerate(all_input):
            # 输入单个特征提取结果结果
            batch_size = input.size(0)
            height_size = input.size(2)
            width_size = input.size(3)
            if height_size == 13:
                anchor = self.anch_13
            elif height_size == 26:
                anchor = self.anch_26
            elif height_size == 52:
                anchor = self.anch_52

            # 调整输入尺寸, 改为 1,3,52,52,85
            # batch_size, 3锚框下的, (52,52)某一个点的, 85维度张量
            pred = input.view(batch_size, self.num_anchors, self.num_classes+5, height_size, width_size).permute(0, 1, 3, 4, 2).contiguous()
            # 锚框的调整参数, 并将之归一化
            x_offset = torch.sigmoid(pred[...,0])
            y_offset = torch.sigmoid(pred[...,1])
            height_anchor = pred[...,2]
            width_anchor = pred[...,3]
            # 获得是否有物体置信度
            have_conf = torch.sigmoid(pred[...,4])
            # 种类置信度张量
            class_conf = torch.sigmoid(pred[...,5:])

            ''' GPU计算窗口
                FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
                LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            '''
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
            pred_box = torch.FloatTensor(pred[..., :4].shape).to(self.device)
            pred_box[..., 0] = x_offset + grid_x
            pred_box[..., 1] = y_offset + grid_y
            pred_box[..., 2] = torch.exp(width_anchor) + anchor_w
            pred_box[..., 3] = torch.exp(height_anchor) + anchor_h

            # 归一化为小数形式, _sacle为归一化因子, 并将之转化为output = [batch_size, 锚框数量, 85] 格式
            _scale = torch.Tensor([width_size, height_size, width_size, height_size]).type(torch.FloatTensor).to(self.device)
            output = torch.cat((pred_box.view(batch_size, -1, 4) / _scale,
                                    have_conf.view(batch_size, -1, 1), class_conf.view(batch_size, -1, self.num_classes)), -1)
            outputs = output if outputs == None else torch.cat((outputs, output), dim=1)
        print(f'Before NMS, the number of all anchors is {outputs.shape[1]} in each batch')
        return outputs

    def correct_boxes(self, box_xy, box_wh, img_size: tuple):
        '''
            Parameters:
                box_xy: 所有box的xy [N,2]
                box_wh: 所有box的wh [N,2]
                img_size: 原图片大小
        '''
        # 实现xy换序为yx
        box_yx = box_xy[:, ::-1]
        box_hw = box_wh[:, ::-1]
        input_size = np.array(self.input_size)
        img_size = np.array(img_size)
        # 获得正方形416,416格式在原图片(长方形)长宽比例下的图片高宽大小
        # np.round, 四舍五入数据
        # 换序后的yx与scale相乘得到在[416, 新高]下的box_yx
        new_shape = np.round(img_size * np.min(input_size / img_size))
        scale = input_size / new_shape
        box_yx *= scale
        box_hw *= scale
        # 换算成左上右下坐标形式
        box_lefttop = box_yx - (box_hw / 2.)
        box_rightbottom = box_yx + (box_hw / 2.)
        # boxes承载原图上预测框的lefttop_x, lefttop_y, rightbottom_x, rightbottom_y
        boxes = np.concatenate([box_lefttop[:, 1:2], box_lefttop[:, :1], box_rightbottom[:,1:2], box_rightbottom[:, :1]],axis=-1)
        sc = np.concatenate([img_size, img_size], axis=-1)
        boxes *= sc
        return boxes


    # 进行非极大抑制
    def non_max_supperssion(self, decode_anchors, img_size: tuple, conf_threshold = 0.5, nms_threshold = 0.4) -> np.ndarray:
        '''
            parameters:
                decode_anchors: 单个网格下的特征图解码后的锚框张量
                img_size: 原图片宽高大小
                conf_threshold: 用于判定锚框包含物体置信度\times 锚框类别置信度 的阈值
                nms_threshold: 用于判定非极大抑制下的阈值
        '''
        # decode_anchors 是decode_box返回的锚框的坐标形式 张量大小为 batch_size, 锚框数量, 85(类别数量+4锚框xywh+1含有物体置信度)
        # torch.Tensor.new()是按照()中的tensor生成一个shape相同, device相同的空tensor
        # 将xyhw型坐标变成xyxy型坐标(左上角和右下角xy坐标)
        box_corner = decode_anchors.new(decode_anchors.shape)
        box_corner[:,:,0] = decode_anchors[:,:,0] - decode_anchors[:,:,2] / 2
        box_corner[:,:,1] = decode_anchors[:,:,1] - decode_anchors[:,:,3] / 2
        box_corner[:,:,2] = decode_anchors[:,:,0] + decode_anchors[:,:,2] / 2
        box_corner[:,:,3] = decode_anchors[:,:,1] + decode_anchors[:,:,3] / 2
        decode_anchors[:,:, :4] = box_corner[:,:, :4]
        
        # output数组每一个元素存储一个batch下的nms后输出
        output = [None for i in range(decode_anchors.shape[0])]
        for batch, pred_anchor in enumerate(decode_anchors):
            # 每一个batch下的xyxy形式锚框
            # pred_anchor = [锚框数量, 85]
            # class_conf表示classes的置信度, class_pred表示max的种类值
            class_conf, class_pred = torch.max(pred_anchor[:, 5:5+self.num_classes], dim=1, keepdim=True)
            # 计算每一个锚框中 含有物体置信度 * 分类别类别概率 的总概率, Tensor.squeeze()去掉多余的行列, 只保留一维数组
            # 计算结果相当于是所有锚框中是否存在物体的概率
            conf_mask = (pred_anchor[:, 4] * class_conf[:, 0] >= conf_threshold).squeeze()
            # 筛选含有物体锚框概率, 含有物体锚框中类别号, 含有物体锚框中类别概率(若通过mask, 即大于阈值, 留存, 否则删去)
            pred_anchor = pred_anchor[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            # 如果本batch没有锚框
            if not pred_anchor.size(0):
                continue
            # 预测锚框, [筛选后锚框数量, 7], 7 = 四个坐标xyxy + 含有物体置信度 + class分类置信度 + class号
            detect_anchors = torch.cat((pred_anchor[:, :5], class_conf.float(), class_pred.float()), dim=1)
            # 获取detect_anchors中的所有不同的种类torch.unique返回数组中所有不同的数字
            all_classes = detect_anchors[:, -1].cpu().unique()
            # GPU入口
            if pred_anchor.is_cuda:
                all_classes = all_classes.cuda()
                detect_anchors = detect_anchors.cuda()
            # 非极大抑制
            for each_class in all_classes:
                # 取出每一类下的相同的类对应的锚框
                detect_class = detect_anchors[detect_anchors[:, -1] == each_class]
                # nms返回一个递减的序列(按照score排列), [N, 7]
                keep = nms(detect_class[:, :4], detect_class[:, 4] * detect_class[:, 5], nms_threshold)
                nms_detect_anchors = detect_class[keep]
                # 拼接输出
                output[batch] = nms_detect_anchors if output[batch] == None else torch.cat((output[batch], nms_detect_anchors))
            
            # 将nms后的预测框返回到原图尺寸
            if output[batch] is not None:
                output[batch] = output[batch].cpu().numpy()
                # 返回到xywh形式
                box_xy, box_wh = (output[batch][:, :2] + output[batch][:, 2:4])/2, output[batch][:, 2:4] - output[batch][:, :2]
                output[batch][:, :4] = self.correct_boxes(box_xy, box_wh, img_size)
                # output: [batch_size, num_anchors, 7=xyxy+have_conf+class_conf+class_number]
                output = np.array(output)
                print(f'After NMS, the number of anchors is {output.shape[1]} in each batch')
        return output

class DrawPicture():

    def __init__(self, ori_img, save_path: str, conf_threshold = 0.95) -> None:
        '''
            Parameters:
                ori_img: 原始图片
                save_path: 保存路径
                conf_threshold: 最后一次筛选置信阈值
        '''
        self.ori_img = ori_img
        self.save_path = save_path
        self.threshold = conf_threshold

    def Draw(self, boxes):
        '''
            Parameters:
                boxes: 预测框 [batch_size, num_boxes, 7=xyxy+have_conf+class_conf+class_number]
        '''
        from names import LoadNames
        for batch in boxes:
            for box in batch:
                lfttop = (int(box[0]),int(box[1]))
                rtbottom = (int(box[2]), int(box[3]))        
                class_num = int(box[-1])
                conf = np.round(box[4] * box[5], decimals=3)
                color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                if conf >= self.threshold:
                    class_name = LoadNames(class_num)
                    frame = cv2.rectangle(self.ori_img, lfttop, rtbottom, color,thickness=2)
                    font_size = cv2.getTextSize(f'{class_name}  {conf:.3f}',0,1,2)
                    frame = cv2.rectangle(frame, (int(lfttop[0]), int(lfttop[1]-font_size[0][1])), (int(lfttop[0]+font_size[0][0]), int(lfttop[1])), color, -1)
                    frame = cv2.putText(frame, text=f'{class_name}  {conf:.3f}', org=lfttop, fontFace=0, fontScale=1, thickness=2,
                                        color=(color[1], color[2], color[0]))
                    cv2.imwrite(f'{self.save_path}{os.sep}new2.jpg', frame)

if __name__ == '__main__':


    # 读取图像与标准化 #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} to detect')
    ori_img_cv2 = np.array(cv2.imread('street.png'))
    img_size = tuple([ori_img_cv2.shape[0], ori_img_cv2.shape[1]])
    input_size = (416, 416)
    num_classes = 80
    num_anchors = 3
    input_img = cv2.resize(ori_img_cv2, input_size)
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.reshape(input_img,(1, 3, input_size[0], input_size[1]))
    input = torch.from_numpy(input_img).float().to(device)
    # 加载模型 #
    dark = yolov3.DarkNet().to(device)
    yolo = yolov3.Yolov3().to(device)
    At = yolov3.Attribute(num_classes,num_anchors).to(device)
    # dark.load_state_dict(torch.load('', map_location=device))
    # yolo.load_state_dict(torch.load('', map_location=device))
    # At.load_state_dict(torch.load('', map_location=device))
    # 特征提取 #
    
    d1, d2, d3 = dark(input)
    
    out = yolo(d1,d2,d3)
    
    attr0 = At(out[0])
    attr1 = At(out[1])
    attr2 = At(out[2])
    attrs = [attr0, attr1, attr2]
    # 解码预测框与得到nms后的真实box #
    dbox = DecodeBox(num_classes,device, input_size)
    decode_anchors = dbox.decode_box(attrs).data # 截断梯度
    nms_boxes = dbox.non_max_supperssion(decode_anchors, img_size)
    # 绘制框 #
    draw = DrawPicture(ori_img_cv2,'save')
    draw.Draw(nms_boxes)