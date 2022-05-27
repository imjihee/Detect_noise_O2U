import torch
from torch.autograd import Variable
import torch.nn.functional as F

def adjust_learning_rate(optimizer, epoch,max_epoch=200):
	if epoch < 0.25 * max_epoch:
		lr = 0.01
	elif epoch < 0.5 * max_epoch:
		lr = 0.005
	else:
		lr = 0.001
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

def evaluate(test_loader, model1):
	model1.eval()
	correct1 = 0
	total1 = 0
	for images, labels, _ in test_loader:
		images = Variable(images).cuda()

		logits1 = model1(images)
		outputs1 = F.log_softmax(logits1, dim=1)
		_, pred1 = torch.max(outputs1.data, 1)
		total1 += labels.size(0)
		correct1 += (pred1.cpu() == labels).sum()
	acc1 = 100 * float(correct1) / float(total1)
	model1.train()

	return acc1