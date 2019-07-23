import io
#import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
from main import *
import glob
import os

imgPath = '/train/trainset/1/DMS/data/test/drink/'
savePath = 'cam_train/'

# input image
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
#model_id = 1
#if model_id == 1:
#    net = models.squeezenet1_1(pretrained=True)
#    finalconv_name = 'features' # this is the last conv layer of the network
#elif model_id == 2:
#    net = models.resnet18(pretrained=True)
#    finalconv_name = 'layer4'
#elif model_id == 3:
#    net = models.densenet161(pretrained=True)
#    finalconv_name = 'features'
#print(net.named_modules())

model_dir = 'checkpoint_no_normalize.pth.tar'
#net = models.resnet18()
net = Net()
net = torch.nn.DataParallel(net).cuda()
net.load_state_dict(torch.load(model_dir)['state_dict'])
net.eval()
#finalconv_name = 'module.model.13'
#print([i[0] for i in net.named_modules()])
#exit(0)

#for name, sub_module in net.named_modules():
#        print(name,sub_module)

#exit(0)
# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


for name, sub_module in net.named_modules():   
        if not isinstance(sub_module, nn.ModuleList) and \
                not isinstance(sub_module, nn.Sequential) and \
                type(sub_module) in nn.__dict__.values() and \
                not isinstance(sub_module, nn.Softmax):
            layer = sub_module
            #print(name)
            if name == 'module.model.13.3':
                hook = layer.register_forward_hook(hook_feature)


#print([k for k,v in net._modules.items()])
#print(type(net._modules),net._modules.get('module.model.13.1'))
#net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   #transforms.Grayscale(3),
   transforms.Resize((224,224)),
   transforms.ToTensor()
   #normalize
])

#imgPath = '../DMS/data/val_phone/phone/Feb152019190131Feb152019190131_125.jpg'
#response = requests.get(IMG_URL)

# download the imagenet category list
classes = {0:'bg',1:'drink',2:'phone'}


img_list = os.listdir(imgPath)

f = open('result.lst','a')

for image in img_list:
    print(image)
    img_pil = Image.open(os.path.join(imgPath,image))
    img_pil.save('test.jpg')

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()

    # output the prediction
    for i in range(0, 3):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread('test.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(os.path.join(savePath,image), result)
    s = image+' '+classes[idx[0]]+' '+str(probs[0]) +'\n'
    f.write(s)
    print('saved')

f.close()

