import torch
from torch.autograd import Variable
import numpy as np
from data.mask_data import Mask_Select
from utils import evaluate, adjust_learning_rate


from resnet import ResNet50, ResNet101
def third_stage(args, noise_or_not, network, train_dataset, test_loader, filter_mask=None):
    # third stage
    stage = 3
    train_loader_init = torch.utils.data.DataLoader(dataset=Mask_Select(train_dataset, filter_mask),
                                                    batch_size=128,
                                                    num_workers=32,
                                                    shuffle=True, pin_memory=False)

    save_checkpoint = args.network + '_' + args.dataset + '_' + args.noise_type + str(args.noise_rate) + '.pt'

    print("restore model from %s.pt" % save_checkpoint)
    network.load_state_dict(torch.load(save_checkpoint))
    ndata = train_dataset.__len__()
    optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()

    for epoch in range(1, args.n_epoch3):
        # train models
        globals_loss = 0
        network.train()
        with torch.no_grad():
            accuracy = evaluate(test_loader, network)
        example_loss = np.zeros_like(noise_or_not, dtype=float)  # sample 개수만큼 길이 가진 example_loss vector 생성
        lr = adjust_learning_rate(optimizer1, epoch, args.n_epoch3)  # lr 조정
        for i, (images, labels, indexes) in enumerate(train_loader_init):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logits = network(images)
            loss_1 = criterion(logits, labels)

            for pi, cl in zip(indexes, loss_1):
                example_loss[pi] = cl.cpu().data.item()  # save loss of each samples

            globals_loss += loss_1.sum().cpu().data.item()
            loss_1 = loss_1.mean()

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
        print("Stage %d - " % stage, "epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata,
              "test_accuarcy:%f" % accuracy)
        if filter_mask is None:
            torch.save(network.state_dict(), save_checkpoint)
