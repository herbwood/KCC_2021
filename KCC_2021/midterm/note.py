import cv2
import numpy as np
import torch
from torchsummary import summary 
from torch.nn import functional as F

class ModelOutputs_resnet():
    def __init__(self, model, target_layers, target_sub_layers):
        self.model = model
        self.target_layers = target_layers
        self.target_sub_layers = target_sub_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        self.gradients = []

        for name, module in self.model.named_children():
            x = module(x)
            if name == 'avgpool':
                x = torch.flatten(x, 1)
            if name in self.target_layers:
                for sub_name, sub_module in module[len(module)-1].named_children():
                    if sub_name in self.target_sub_layers:
                        x.register_hook(self.save_gradient)
                        target_feature_maps = x

        return target_feature_maps, x


class GradCam_resnet:
    def __init__(self, model, target_layer_names,target_sub_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:  # GPU일 경우 model을 cuda로 설정
            self.model = model.cuda()

        self.extractor = ModelOutputs_resnet(self.model, target_layer_names,target_sub_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        if self.cuda:  # GPU일 경우 input을 cuda로 변환하여 전달
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        probs,idx = 0, 0
        if index == None:
            index = np.argmax(output.cpu().data.numpy())  # index = 정답이라고 추측한 class index
            h_x = F.softmax(output,dim=1).data.squeeze()
            probs, idx = h_x.sort(0,True)


        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1  # 정답이라고 생각하는 class의 index 리스트 위치의 값만 1로
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)  # numpy배열을 tensor로 변환
        # requires_grad == True 텐서의 모든 연산에 대하여 추적
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features  # A^k

        target_cam = target.cpu().data.numpy()
        bz, nc, h,w = target_cam.shape

        target = target.cpu().data.numpy()[0, :]

        params = list(self.model.parameters())

        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        cam = weight_softmax[index].dot(target_cam.reshape((nc,h*w)))
        cam = cam.reshape(h,w)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))  # 224X224크기로 변환
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)


        weights = np.mean(grads_val, axis=(2, 3))[0, :]  # 논문에서의 global average pooling 식에 해당하는 부분
        grad_cam = np.zeros(target.shape[1:], dtype=np.float32)  # 14X14

        for i, w in enumerate(weights): # calcul grad_cam
            grad_cam += w * target[i, :, :]  # linear combination L^c_{Grad-CAM}에 해당하는 식에서 ReLU를 제외한 식

        grad_cam = np.maximum(grad_cam, 0)  # 0보다 작은 값을 제거
        grad_cam = cv2.resize(grad_cam, (224, 224))  # 224X224크기로 변환
        grad_cam = grad_cam - np.min(grad_cam)  #
        grad_cam = grad_cam / np.max(grad_cam)  # 위의 것과 해당 줄의 것은 0~1사이의 값으로 정규화하기 위한 정리

        return grad_cam, cam ,index,probs,idx 


if __name__ == "__main__":

    import os
    import sys

    model_root_dir = os.path.join('../model/', "retina_emd_diou_refine")
    sys.path.insert(0, model_root_dir)

    from config import config
    from network import Network

    model_file = r"C:\Users\June hyoung Kwon\PROJECTS\KCC_2021\model\retina_emd_diou_refine\outputs\model_dump\dump-49.pth"
    net = Network()
    net.eval()
    check_point = torch.load(model_file, map_location=torch.device('cpu'))
    net.load_state_dict(check_point['state_dict'])

    image = cv2.imread(r"C:\Users\June hyoung Kwon\PROJECTS\KCC_2021\tools\fpn_baseline_04\273271,1b86f000bc5b77bf.jpg")
    input = torch.tensor(image).float()
    cam = GradCam_resnet(net, "resnet50", "layer1")
    gc, c, i, p, i = cam()

