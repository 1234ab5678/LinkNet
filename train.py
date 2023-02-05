import math
import os
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn  
#from linkNet import LinkNet
from linkNet_ShuffleNet_0_5 import LinkNet
#from linkNet_Ghost import LinkNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image

# rotate_img
def rotate_image(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    # convet angle into rad
    rangle = np.deg2rad(angle)  # angle in radians
    # calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # map
    return cv2.warpAffine(
        src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
        flags=cv2.INTER_LANCZOS4)

def compute_class_weights(histogram, num_classes):
    classWeights = np.ones(num_classes, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(num_classes):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights

def focal_loss(input, target):
    '''
    :param input: 使用知乎上面大神给出的方案  https://zhuanlan.zhihu.com/p/28527749
    :param target:
    :return:
    '''
    n, c, h, w = input.size()

    target = target.long()
    inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    N = inputs.size(0)
    C = inputs.size(1)

    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    number_2 = torch.sum(target == 2).item()
    number_3 = torch.sum(target == 3).item()
    number_4 = torch.sum(target == 4).item()
    #number_5 = torch.sum(target == 5).item()

    frequency = torch.tensor((number_0, number_1, number_2, number_3, number_4), dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency, 5)

    weights = torch.from_numpy(classWeights).float()
    weights = weights[target.view(-1)]#这行代码非常重要

    gamma = 2

    P = F.softmax(inputs, dim=1)#shape [num_samples,num_classes]

    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)#shape [num_samples,num_classes]  one-hot encoding

    probs = (P * class_mask).sum(1).view(-1, 1)#shape [num_samples,]
    #print(probs)
    min_prob = torch.full(size=probs.shape, fill_value=0.1).cuda()
    probs = torch.maximum(probs, min_prob)
    log_p = probs.log()
    #print(log_p)
    #print('in calculating batch_loss',weights.shape,probs.shape,log_p.shape)

    # batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p
    batch_loss = -(torch.pow((1 - probs), gamma)) * log_p
    #print(batch_loss)
    #print(batch_loss.shape)

    loss = batch_loss.mean()
    return loss

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def process_image(img, min_side):
    size = img.shape
    h, w = size[0], size[1]
    #长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    #print('------------')
    # ------------------------------------------#
    #   水平翻转图像
    # ------------------------------------------#
    hflip = rand() < .5
    #print(hflip)
    if hflip:
        resize_img = cv2.flip(resize_img, 1)
    # ------------------------------------------#
    #   竖直翻转图像
    # ------------------------------------------#
    vflip = rand() < .5
    #print(vflip)
    if vflip:
        resize_img = cv2.flip(resize_img, 1)
    #print('------------')

    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
    return pad_img,int(top), int(bottom), int(left), int(right)

class DatasetFolder(Dataset):
    def __init__(self):
        self.train_list = []
        #files = glob('./trainval/*.jpg')
        #for line in files:
        #    self.train_list.append(line)
        path = './trainval/'
        for filename in os.listdir(path):
            format = filename[-3:]
            if(format == 'jpg'):
                self.train_list.append(path + filename)
    def __getitem__(self, index):
        item=self.train_list[index]
        bg_item = item[:-4] + '.png'
        img = cv2.imread(item)
        label = cv2.imread(bg_item,cv2.IMREAD_GRAYSCALE)
        img,top,bottom,left,right=process_image(img,512)
        label,top,bottom,left,right=process_image(label,512)
        img=np.transpose(img,(2, 0, 1))
        img = img / 255.
        img = torch.from_numpy(img)
        label =torch.from_numpy(label)
        return img, label
    def __len__(self):
        return len(self.train_list)

path="./models/model.pth"
net=LinkNet(5)
if os.path.exists(path):
    net.load_state_dict(torch.load(path, map_location='cpu'))
net.cuda()
#focalloss = FocalLoss(5)
#diceloss = DiceLoss()
initiallr = 1e-3
criterion=nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=initiallr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = torch.optim.Adam([{"params": net.parameters(), "initial_lr": lr, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 0}])
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min = lr * 0.01, last_epoch = -1)
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, last_epoch = -1)
Init_Epoch = 0
epochs= 300
train_dataset = DatasetFolder()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
#print(train_dataset.__len__())
resume = "" #./models/resume_model.pth
if resume != "":  # 接上次的断点继续训练
    print("-----------断点续训-------------")
    checkpoint = torch.load(resume, map_location='cpu')
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    Init_Epoch = start_epoch
    print("the training process from epoch{}...".format(start_epoch))

losses=[]
print("training...")
minloss = math.inf
max_score=0.
for epoch in range(Init_Epoch, epochs):
    net.train()
    for i, (input, lable) in enumerate(train_loader):
        input = input.type(torch.FloatTensor).cuda()
        lable = lable.type(torch.LongTensor).cuda()
        output = net(input)
        # print(np.shape(output))
        #loss = criterion(output, lable) + focal_loss(output, lable)
        loss = focal_loss(output, lable)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        #mean_loss = np.mean(losses[-100:]) * 1e5
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.6f}\t mean_loss {mean_loss:.6f}\t lr {lr:.6f}'.format(
                epoch, i, len(train_loader), loss=loss.item()*1e5,mean_loss=np.mean(losses[-100:])*1e5,
                lr=optimizer.param_groups[0]['lr']))
    scheduler.step()

    if (np.mean(losses[-100:])*1e5 < minloss):
        print('best model saved!')
        torch.save(net.state_dict(), "./models/best.pth")

    save_files = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch}
    torch.save(save_files, "./models/resume_model.pth")
    torch.save(net.state_dict(), "./models/model.pth")
    print("model save!!!!")