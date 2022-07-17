from time import sleep
import numpy as np
import pandas as pd
import pdb
import pickle
import datetime
from pytz import timezone

import torch
import collections
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn.functional as F

from data.mask_data import Mask_Select, Correct_label
from utils import evaluate, adjust_learning_rate
from ricap import RICAPCollactor, RICAPloss, ricap_dataset, ricap_criterion

def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)

"""
Third Stage: Curriculum Learning with Relatively Clean Data
"""
def third_stage(args, noise_or_not, network, train_dataset, test_loader, filter_mask, idx_sorted):
    # third stage
    stage = 3
    test_acc = []
    train_loss = []
    sf = True
    light=False
    if args.curriculum:
        sf = False #sf: shuffle

    clean_train_dataset = Mask_Select(train_dataset, filter_mask, idx_sorted, args.curriculum)

    if args.use_ricap:
        train_loader_init = torch.utils.data.DataLoader(
            dataset=ricap_dataset(clean_train_dataset),
            batch_size=128,
            num_workers=32,
            shuffle=True, pin_memory=False)
        criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
        #RICAPloss()
    else:
        train_loader_init = torch.utils.data.DataLoader(
            dataset=clean_train_dataset,
            batch_size=128,
            num_workers=32,
            shuffle=True, pin_memory=False)
        criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()

    ndata = train_dataset.__len__()
    optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    correct_label = np.zeros_like(noise_or_not, dtype=int)  # sample 개수만큼 길이 가진 example_loss vector 생성
    

    print("----------- Start Third Stage -----------")

    for epoch in range(1, args.n_epoch3):
        # train models
        globals_loss = 0
        network.train()
        with torch.no_grad():
            accuracy = evaluate(test_loader, network)
        
        print()
        lr = adjust_learning_rate(optimizer1, epoch, args.n_epoch3)  # lr 조정
        for i, (images, labels, indexes) in enumerate(train_loader_init):
            images = Variable(images).cuda()
            
            #logits = network(images.float()) #logits: torch.Size([128, 10])
            #loss_1 = ricap_criterion(logits, labels)
            if not args.use_ricap:
                labels = Variable(labels).cuda()
                logits = network(images)
                #pdb.set_trace()
                loss_1 = criterion(logits, labels)
            else:
                labels = Variable(labels).type(torch.float32).cuda()
                logits = network(images.float())
                loss_1 = ricap_criterion(logits, labels)

            if epoch==args.n_epoch3-1:
                outputs = F.log_softmax(logits, dim=1)
                for pi, cl in zip(indexes, torch.max(outputs.data, 1).indices):
                    idx, mx = pi.item(), cl.item()
                    correct_label[idx] = mx  # save correct label of each samples
                    #pdb.set_trace()
                    if light==False:
                        print("...make correct_label...")
                        light=True
            

            globals_loss += loss_1.sum().cpu().data.item()
            loss_1 = loss_1.mean()

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()

        
        print("Stage %d - " % stage, "epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata,
              "test_accuarcy:%f" % accuracy)
        

        test_acc.append(accuracy)
        train_loss.append(globals_loss/ndata)

    log_data = np.concatenate(([train_loss], [test_acc]), axis=0)
    export_toexcel(args, log_data, 3)
    counter = collections.Counter(correct_label)
    print("Correction Result:", counter)
    print("** stage 3 max test accuracy:", max(test_acc))

    lab_path = "log/correct_label/"+args.dataset+"_label_"+str(args.noise_rate)+"_"+str(args.remove_rate)+"_"+args.network
    with open(lab_path,"wb") as fp:
        pickle.dump(correct_label, fp)

    return network, correct_label
    
"""
Fourth Stage
"""
#Stage 4
def label_correction(args, network, corrected_label, train_dataset, test_loader):
    
    stage = 4
    test_acc = []
    train_loss = []

    correct_train_dataset = Correct_label(train_dataset, corrected_label)

    train_loader_init = torch.utils.data.DataLoader(
        dataset=correct_train_dataset,
        batch_size=128,
        num_workers=32,
        shuffle=True, pin_memory=False)
    criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()

    ndata = train_dataset.__len__()
    optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    print("----------- Start Fourth(4) Stage -----------")

    for epoch in range(1, args.n_epoch4):
        # train models
        globals_loss = 0
        network.train()
        with torch.no_grad():
            accuracy = evaluate(test_loader, network)
        lr = adjust_learning_rate(optimizer1, epoch, args.n_epoch4)  # lr 조정
        for i, (images, labels, indexes) in enumerate(train_loader_init):
            images = Variable(images).cuda()
            
            labels = Variable(labels).cuda()
            logits = network(images)
            #pdb.set_trace()
            loss_1 = criterion(logits, labels)

            globals_loss += loss_1.sum().cpu().data.item()
            loss_1 = loss_1.mean()

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()

        print("Stage %d - " % stage, "epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata,
              "test_accuarcy:%f" % accuracy)
        test_acc.append(accuracy)
        train_loss.append(globals_loss/ndata)

    log_data = np.concatenate(([train_loss], [test_acc]), axis=0)
    export_toexcel(args, log_data, 4)
    print("** stage 4 max test accuracy:", max(test_acc))


def export_toexcel(args, data, stage):
    df = pd.DataFrame(data)
    df = (df.T)
    
    td = datetime.datetime.now(timezone('Asia/Seoul'))

    xlsx_path = args.fname + '/acc_Stage_' + str(stage) + '_' + args.time_now + '.xlsx'
    writer1 = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')

    df.columns = ['train loss', 'test acc']
    df.to_excel(writer1)
    writer1.save()
    print("SAVE " + xlsx_path + " successfully")
