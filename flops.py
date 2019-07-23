import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from main import Net
from utils import get_flops
#from torch.autograd import Variable

model_dir = 'weights/checkpoint_quantize.pth.tar'
model = Net()
model = torch.nn.DataParallel(model).cuda(0)
#checkpoint = torch.load(model_dir)
#model.load_state_dict(checkpoint['state_dict'])
#model.eval()
print(get_flops(model))
