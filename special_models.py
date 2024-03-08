import onnxruntime
import numpy as np
import cv2
import time
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from torch import nn
import torch
from utils import softmax
from conf import conf


class ResNet18:
    def __init__(self, path):
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 1)
        self.model.eval()

        checkpoint = torch.load(path, map_location='cpu')
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_heatmap(self, feature_map, h, w):
        feat = feature_map
        # score.backward()
        # grad = feat.grad
        # grad = F.adaptive_avg_pool2d(grad, (1, 1)).squeeze(0)
        feat = feat.squeeze(0)
        # feat = feat * grad
        feat = feat * self.model.fc.weight.unsqueeze(0).permute(2, 0, 1)

        heatmap = feat.detach().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        heatmap = (cv2.resize(heatmap, (w, h)) * 255).astype(np.uint8)
        return heatmap

    def forward(self, cv2img, cropper, idx):
        img = Image.fromarray(cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB))
        img = cropper.crop(img, idx)
        b_img = self.transform(img).unsqueeze(0)            # add batch axis
        with torch.no_grad():
            feat = nn.Sequential(*list(self.model.children())[0:-2])(b_img)
            pooled = self.model.avgpool(feat).view(1, -1)
            score = torch.sigmoid(self.model.fc(pooled))
        predict = int(score > conf.MODEL_POSITIVE_THRES[idx])

        h, w = img.size[1], img.size[0]
        if not predict:
            heatmap = np.zeros((h, w)).astype(np.uint8)
        else:
            heatmap = self.get_heatmap(feat, h, w)
        heatmap = cropper.putback(heatmap, idx)
        return {'score': score, 'pred': predict, 'hm': heatmap}


class YOLO:
    def __init__(self, config, model, labels, size=416, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size

        self.labels = labels
        try:
            self.net = cv2.dnn.readNetFromDarknet(config, model)
        except:
            raise ValueError("Couldn't find the models!\nDid you forget to download them manually (and keep in the correct directory, models/) or run the shell script?")

    def inference_from_file(self, file):
        mat = cv2.imread(file)
        return self.inference(mat)

    def inference(self, image):
        ih, iw = image.shape[:2]

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()
        inference_time = end - start

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([iw, ih, iw, ih])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                id = classIDs[i]
                confidence = confidences[i]

                results.append((id, self.labels[id], confidence, x, y, w, h))

        return results

    def forward(self, cv2img, cropper, idx):
        img = cv2img
        results = self.inference(img)
        center_points = []
        predict = 0
        top, left, height, width = cropper.get_crop_pos(idx)
        hm = np.zeros(img.shape[0:2])
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            if top < y + (h / 2) < (top + height):
                center_points.append((x + (w / 2), y + (h / 2)))
                # draw a bounding box rectangle on the image
                cv2.rectangle(hm, (x, y), (x + w, y + h), 1, 10)

        if len(center_points) == 1 or len(center_points) == 2 and (center_points[0][0] - center_points[1][0]) ** 2 + (
                center_points[0][1] - center_points[1][1]) ** 2 < 10000:
            predict = 1
        return {'score': 0, 'pred': predict, 'hm': hm.astype(np.uint8) * 255}


class OnnxTsmResnet50:
    def __init__(self, path):
        self.model = onnxruntime.InferenceSession(path)
        self.buffer = [
            np.zeros((1, 8, 56, 56), dtype=np.float32),
            np.zeros((1, 32, 56, 56), dtype=np.float32),
            np.zeros((1, 32, 56, 56), dtype=np.float32),

            np.zeros((1, 32, 56, 56), dtype=np.float32),
            np.zeros((1, 64, 28, 28), dtype=np.float32),
            np.zeros((1, 64, 28, 28), dtype=np.float32),
            np.zeros((1, 64, 28, 28), dtype=np.float32),

            np.zeros((1, 64, 28, 28), dtype=np.float32),
            np.zeros((1, 128, 14, 14), dtype=np.float32),
            np.zeros((1, 128, 14, 14), dtype=np.float32),
            np.zeros((1, 128, 14, 14), dtype=np.float32),
            np.zeros((1, 128, 14, 14), dtype=np.float32),
            np.zeros((1, 128, 14, 14), dtype=np.float32),

            np.zeros((1, 128, 14, 14), dtype=np.float32),
            np.zeros((1, 256, 7, 7), dtype=np.float32),
            np.zeros((1, 256, 7, 7), dtype=np.float32),
        ]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def inference(self, img: np.ndarray):
        name_lst = ["input"] + ["buffer_{}".format(i) for i in range(len(self.buffer))]
        content_lst = [img] + self.buffer
        logit, *self.buffer = self.model.run([], dict(zip(name_lst, content_lst)))
        return logit[0]

    def forward(self, cv2img, cropper, idx):
        img = Image.fromarray(cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB))
        img = cropper.crop(img, idx)
        b_img = self.transform(img).unsqueeze(0)

        logit = self.inference(b_img.numpy())
        prob = softmax(logit)
        if np.max(prob) < conf.HW_MODEL_POSITIVE_THRES:
            pred = 0
        else:
            pred = np.argmax(prob) + 1
        return {'score': float(np.max(prob)), 'pred': int(pred), 'hm': int(pred)}


# TSM相关
class TemporalShift(nn.Module):
    def __init__(self, net, n_div, init_buffer):
        super(TemporalShift, self).__init__()
        self.net = net
        self.fold_div = n_div

        self.buffer = init_buffer
        # self.register_buffer('buffer', init_buffer)

    def forward(self, x):
        x = self.shift(x, fold_div=self.fold_div)
        x = self.net(x)
        return x

    def shift(self, x, fold_div):
        # x (1, c, h, w)
        c = x.shape[1]
        fold = c // fold_div
        tmp = x[:, fold: 2 * fold].clone()

        out = x.clone()
        out[:, fold: 2 * fold] = self.buffer
        # print(x)
        # x[:, fold: 2 * fold] = 0
        self.buffer = tmp

        return out


# 22-02-28新加入 该模型支持GPU模式，隶属复杂度1，支持所有动作（含精细洗手）
class C1GPU_TsmResnet50:
    def __init__(self, path, action_type='common'):
        assert action_type in ('common', 'handwash')
        self.action_type = action_type
        init_buffers = [
            torch.zeros((1, 8, 56, 56)),
            torch.zeros((1, 32, 56, 56)),
            torch.zeros((1, 32, 56, 56)),

            torch.zeros((1, 32, 56, 56)),
            torch.zeros((1, 64, 28, 28)),
            torch.zeros((1, 64, 28, 28)),
            torch.zeros((1, 64, 28, 28)),

            torch.zeros((1, 64, 28, 28)),
            torch.zeros((1, 128, 14, 14)),
            torch.zeros((1, 128, 14, 14)),
            torch.zeros((1, 128, 14, 14)),
            torch.zeros((1, 128, 14, 14)),
            torch.zeros((1, 128, 14, 14)),

            torch.zeros((1, 128, 14, 14)),
            torch.zeros((1, 256, 7, 7)),
            torch.zeros((1, 256, 7, 7)),
        ]

        self.n_div = 8
        self.model = resnet50()

        if action_type == 'common':
            out_dim = 1
        elif action_type == 'handwash':
            out_dim = 12
        else:
            raise NotImplementedError
        self.model.fc = nn.Linear(2048, out_dim)
        self._get_shift_model(init_buffers)

        self.model.eval()
        self.device = torch.device('cuda') if conf.USE_GPU else torch.device('cpu')
        self.model = self.model.to(self.device)

        checkpoint = torch.load(path) if conf.USE_GPU else torch.load(path, map_location='cpu')
        checkpoint = {k.replace('module.base_model.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _make_block_shift(self, stage, init_buffers):
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            blocks[i].conv1 = TemporalShift(b.conv1, self.n_div, init_buffers[i])
        return nn.Sequential(*blocks)

    def _get_shift_model(self, init_buffers):
        self.model.layer1 = self._make_block_shift(self.model.layer1, init_buffers[0: 3])
        self.model.layer2 = self._make_block_shift(self.model.layer2, init_buffers[3: 7])
        self.model.layer3 = self._make_block_shift(self.model.layer3, init_buffers[7: 13])
        self.model.layer4 = self._make_block_shift(self.model.layer4, init_buffers[13: 16])

    def forward(self, cv2img, cropper, idx):
        img = Image.fromarray(cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB))
        # if idx==4:
        #     import pdb
        #     pdb.set_trace()
        img = cropper.crop(img, idx)
        b_img = self.transform(img).unsqueeze(0).to(self.device)            # add batch axis
        with torch.no_grad():
            feat = nn.Sequential(*list(self.model.children())[0:-2])(b_img) # feat.shape=[1,2048,7,7]
            pooled = self.model.avgpool(feat).view(1, -1) # jiarun: [1,2048]
            h, w = img.size[1], img.size[0]
            if self.action_type == 'common':
                score = torch.sigmoid(self.model.fc(pooled)).cpu()
                predict = int(score > conf.MODEL_POSITIVE_THRES[idx]) # jiarun: 当预测分数大于阈值的时候，才显示heatmap
                if not predict:
                    heatmap = np.zeros((h, w)).astype(np.uint8)
                else:
                    heatmap = self.get_heatmap(feat.cpu(), h, w)
                heatmap = cropper.putback(heatmap, idx)
            elif self.action_type == 'handwash':
                logit = self.model.fc(pooled).squeeze(0).cpu() # jiarun: [num_classes]
                prob = torch.softmax(logit, dim=0) # jiarun: [num_classes]
                score = float(torch.max(prob)) # jiarun: [1]

                if score < conf.HW_MODEL_POSITIVE_THRES :
                    predict = 0
                    heatmap = np.zeros((h, w)).astype(np.uint8)
                else:
                    predict = int(torch.argmax(prob) + 1)
                    heatmap = self.get_heatmap(feat.cpu(), h, w, predict-1)
                # heatmap = int(predict)

                heatmap = cropper.putback(heatmap, idx)

            else:
                raise NotImplementedError

        return {'score': score, 'pred': predict, 'hm': heatmap}

    def get_heatmap(self, feature_map, h, w, class_id=None):
        feat = feature_map # jiarun: [1,2048,7,7]

        # score.backward()
        # grad = feat.grad
        # grad = F.adaptive_avg_pool2d(grad, (1, 1)).squeeze(0)
        feat = feat.squeeze(0) # jiarun: [2028,7,7]
        # feat = feat * grad
        if class_id==None:
            weight = self.model.fc.weight.unsqueeze(0).mean(1,keepdim=True) # jiarun add this line for adapting to handwash: [1,1,2048]
        else:
            weight = self.model.fc.weight[class_id,:].reshape(1,1,-1) # jiarun: [1,1,2048]
        feat = feat * weight.permute(2, 0, 1).cpu() # fc.weight=[num_classes,2048]->[1,num_classes,2048]->[2048,1,num_classes]
        heatmap = feat.detach().numpy() # [2048,7,7]
        heatmap = np.mean(heatmap, axis=0) # [7,7]

        heatmap = np.maximum(heatmap, 0) #
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        heatmap = (cv2.resize(heatmap, (w, h)) * 255).astype(np.uint8)
        return heatmap


# 临时模型：”检查穿戴完整性“步骤暂无模型，该模型输出一段时间0（延迟）后持续输出1
class DummyModel:
    def __init__(self, zero_interval):
        self.zero_interval = zero_interval

    def forward(self, cv2img, cropper, idx):
        if self.zero_interval:
            self.zero_interval -= 1
            return {'score': 0, 'pred': 0, 'hm': np.zeros_like(cv2img).astype(np.uint8)}
        else:
            return {'score': 1, 'pred': 1, 'hm': np.zeros_like(cv2img).astype(np.uint8)}
