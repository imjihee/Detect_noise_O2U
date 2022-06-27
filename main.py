# -*- coding:utf-8 -*-

#sudo python main.py --n_epoch=250 --method=ours-base  --dataset=cifar100 --batch_size=128
import torch
import pathlib
import datetime
from pytz import timezone
import argparse, sys
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mask_data import Mask_Select

from curriculum import third_stage
from utils import evaluate, adjust_learning_rate
from resnet import ResNet50, ResNet101

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = '../results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.3)
parser.add_argument('--remove_rate', type = float, help = 'rate of the total dataset to be removed', default = None) 
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--fname',type=str, default = "", help='log folder name')

parser.add_argument('--dataset', type = str, help = 'mnist, minimagenet, cifar10, cifar100', default = 'cifar10')
parser.add_argument('--n_epoch1', type=int, default=10) #train epoch for stage 1. minimum 1
parser.add_argument('--n_epoch2', type=int, default=120) #train epoch for stage 2. original 250. minimum 2
parser.add_argument('--n_epoch3', type=int, default=200) #train epoch for stage 3. minimum 1
parser.add_argument('--seed', type=int, default=2)

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--network', type=str, default="resnet50")
parser.add_argument('--transforms', type=str, default="false")

parser.add_argument('--unstabitily_batch', type=int, default=16)

parser.add_argument('--curriculum', action='store_true')
args = parser.parse_args()
print(args)
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

network_map={'resnet50':ResNet50,'resnet101':ResNet101}
CNN=network_map[args.network]


transforms_map32 = {"true": transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()]), 'false': transforms.Compose([transforms.ToTensor()])}
transformer = transforms_map32[args.transforms]

#load dataset
if args.dataset=='cifar10':
	input_channel=3
	num_classes=10
	args.top_bn = False
	args.epoch_decay_start = 80
	train_dataset = CIFAR10(root=args.result_dir,
								download=True,
								train=True,
								transform=transformer,
								noise_type=args.noise_type,
				noise_rate=args.noise_rate
					)

	test_dataset = CIFAR10(root=args.result_dir,
								download=True,
								train=False,
								transform=transforms.ToTensor(),
								noise_type=args.noise_type,
					noise_rate=args.noise_rate
					)

if args.dataset=='cifar100':
	input_channel=3
	num_classes=100
	args.top_bn = False
	args.epoch_decay_start = 100
	train_dataset = CIFAR100(root=args.result_dir,
								download=True,
								train=True,
								transform=transformer,
								noise_type=args.noise_type,
				noise_rate=args.noise_rate
					)

	test_dataset = CIFAR100(root=args.result_dir,
								download=True,
								train=False,
								transform=transforms.ToTensor(),
								noise_type=args.noise_type,
				noise_rate=args.noise_rate
					)
if args.remove_rate is None:
	remove_rate=args.noise_rate
else:
	remove_rate=args.remove_rate
#
noise_or_not = train_dataset.noise_or_not
#print("*****", np.sum(noise_or_not))

"""
First Stage
"""
def first_stage(network,test_loader):

	train_loader_init = torch.utils.data.DataLoader(dataset=train_dataset,
													batch_size=128,
													num_workers=32,
													shuffle=True, pin_memory=False)
	stage = 1
	save_checkpoint=args.network+'_'+args.dataset+'_'+args.noise_type+str(args.noise_rate)+'.pt'

	ndata = train_dataset.__len__()
	optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
	
	for epoch in range(1, args.n_epoch1):
		# train models
		globals_loss = 0
		network.train()
		
		with torch.no_grad():
			accuracy = evaluate(test_loader, network)
		example_loss = np.zeros_like(noise_or_not, dtype=float) #sample 개수만큼 길이 가진 example_loss vector 생성
		lr=adjust_learning_rate(optimizer1,epoch,args.n_epoch1)

		for i, (images, labels, indexes) in enumerate(train_loader_init):
			images = Variable(images).cuda()
			labels = Variable(labels).cuda()

			logits = network(images)
			loss_1 = criterion(logits, labels)

			for pi, cl in zip(indexes, loss_1):
				example_loss[pi] = cl.cpu().data.item() #save loss of each samples

			globals_loss += loss_1.sum().cpu().data.item()
			loss_1 = loss_1.mean()

			optimizer1.zero_grad()
			loss_1.backward()
			optimizer1.step()

		print ("Stage %d - " % stage, "epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss /ndata, "test_accuarcy:%f" % accuracy)
		torch.save(network.state_dict(), save_checkpoint)

"""
Second Stage
"""
def second_stage(network,test_loader,max_epoch=args.n_epoch2):
	train_loader_detection = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=16,
											   num_workers=32,
											   shuffle=True,
														 )
	optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
	criterion=torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
	moving_loss_dic=np.zeros_like(noise_or_not)
	ndata = train_dataset.__len__()

	for epoch in range(1, max_epoch):
		# train models
		globals_loss=0
		network.train()
		with torch.no_grad():
			accuracy=evaluate(test_loader, network)
		example_loss= np.zeros_like(noise_or_not,dtype=float)

		#Learning Rate Scheduling
		t = (epoch % 40 + 1) / float(10) #40: lr frequency
		lr = (1 - t) * 0.1 + t * 0.001 #default _ 0.01: max, 0.001: min lr

		for param_group in optimizer1.param_groups:
			param_group['lr'] = lr

		for i, (images, labels, indexes) in enumerate(train_loader_detection):

			images = Variable(images).cuda()
			labels = Variable(labels).cuda()

			logits = network(images)
			loss_1 =criterion(logits,labels)

			for pi, cl in zip(indexes, loss_1):
				example_loss[pi] = cl.cpu().data.item()

			globals_loss += loss_1.sum().cpu().data.item()

			loss_1 = loss_1.mean()
			optimizer1.zero_grad()
			loss_1.backward()
			optimizer1.step() #training in an epoch finish

		example_loss = example_loss - example_loss.mean()
		moving_loss_dic = moving_loss_dic+example_loss #moving_loss_dic: ndarray, (50000,0)

		ind_1_sorted = np.argsort(moving_loss_dic) #moving_loss_dic를 오름차순 정렬하는 인덱스의 array 반환.
		loss_1_sorted = moving_loss_dic[ind_1_sorted]
		#loss 오름차순 순서대로 loss_1_sorted에 저장됨. ind_1_sorted에는 loss 오름차순 순서의 인덱스가 저장됨.
		remember_rate = 1 - remove_rate #남길 데이터 비율
		num_remember = int(remember_rate * len(loss_1_sorted)) #num_remember: 40000 @ remove_rate=0.2

		noise_accuracy=np.sum(noise_or_not[ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember) #제거할 데이터 중 노이즈 개수 / 총 노이즈 개수
		mask = np.ones_like(noise_or_not,dtype=np.float32)
		mask[ind_1_sorted[num_remember:]]=0 #지워야 할 인덱스에 대해 0 저장. mask[idx]=0

		top_accuracy_rm=int(0.9 * len(loss_1_sorted))
		top_accuracy= 1-np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(len(loss_1_sorted) - top_accuracy_rm)

		print ("Stage 2 - " + "epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata, "test_accuarcy:%f" % accuracy,"noise_accuracy:%f"%(1-noise_accuracy),"top 0.1 noise accuracy:%f"%top_accuracy)

	return mask, ind_1_sorted[:num_remember] #second stage finish

class Logger(object):
	def __init__(self, dir):
		td = datetime.datetime.now(timezone('Asia/Seoul'))
		file_name = td.strftime('%m-%d_%H.%M') + ".log"
		self.terminal = sys.stdout
		self.log = open(dir + "/" + file_name, "a")

	def write(self, temp):
		self.terminal.write(temp)
		self.log.write(temp)
	
	def flush(self):
		pass

"""main"""
output_d = "log/" + args.dataset + "/noise_" + str(args.noise_rate) + "_remove_" + str(args.remove_rate)
output_dir = pathlib.Path(output_d)
output_dir.mkdir(exist_ok=True, parents=True)
args.fname = output_d

sys.stdout = Logger(output_d)

print(args)

basenet= CNN(input_channel=input_channel, n_outputs=num_classes).cuda()

test_loader = torch.utils.data.DataLoader(
	dataset=test_dataset,batch_size=128,
	num_workers=32,shuffle=False, pin_memory=False)

#1
first_stage(network=basenet,test_loader=test_loader)
#2
filter_mask, ind_1_sorted = second_stage(network=basenet,test_loader=test_loader)
#3
third_stage(args, noise_or_not=noise_or_not, network=basenet,train_dataset=train_dataset, test_loader=test_loader, filter_mask=filter_mask, idx_sorted=ind_1_sorted.tolist())

#First stage --> get Filter mask from second stage --> first stage with Filter mask
