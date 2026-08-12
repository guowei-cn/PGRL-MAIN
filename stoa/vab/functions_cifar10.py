from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
MinMaxNormalization = MinMaxScaler()

def visualize(losses, poisoned_vector, path):
    clean_loss = losses[np.where(poisoned_vector == 0)]
    poison_loss = losses[np.where(poisoned_vector == 1)]
    fig = plt.figure()
    plt.hist([clean_loss, poison_loss], bins=20, label=['clean', 'poison'])
    plt.legend(loc='upper left')
    plt.xlabel('entropy')
    plt.ylabel('PDF')
    # log scale in y axis
    plt.yscale('log')
    plt.savefig('visualize/' + path)
    plt.close(fig)

def eval_train(args, model, pre_loss, eval_loader1, poisoned_vector, epoch, threshold=None, device=None):
    print('\n threshold: %.5f' % (threshold))
    model.eval()
    num_iter = (len(eval_loader1.dataset) // eval_loader1.batch_size) + 1
    losses = torch.zeros(len(eval_loader1.dataset))
    loss_ce_func = nn.CrossEntropyLoss(reduction='none')
    losses_ce = torch.zeros(len(eval_loader1.dataset))
    with torch.no_grad():
        for batch_idx, (inputs1, targets1, index) in enumerate(eval_loader1):
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            logits1 = model(inputs1)
            logits_softmax1 = torch.softmax(logits1, dim=1)
            #logits_softmax1 = logits_softmax1.float()
            for b in range(inputs1.size(0)):
                losses[index[b]] = -torch.mean(torch.mul(logits_softmax1[b, :], torch.log(logits_softmax1[b, :]))) # customised loss
                losses_ce[index[b]] = loss_ce_func(logits1[b].unsqueeze(0), targets1[b].unsqueeze(0))

                # if np.isnan(losses[index[b]]):
                #     print(logits_softmax1[b, :])
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            sys.stdout.flush()

    losses = losses.numpy()
    losses[np.isnan(losses)] = 0
    # auc_pred_entropy = roc_auc_score(1 - poisoned_vector, losses)
    losses_ce = losses_ce.numpy()
    # auc_pred_ce = roc_auc_score(1 - poisoned_vector, losses_ce)
    if len(pre_loss) != 0:
        losses = args.r * losses + (1 - args.r) * pre_loss
    else:
        losses = losses
    pre_loss = losses
    losses = MinMaxNormalization.fit_transform(losses.reshape(-1, 1))
    losses = np.squeeze(losses, 1)
    visualize(losses, poisoned_vector, str(epoch) + '.png')

    pred1 = losses > threshold

    correct = np.count_nonzero(np.equal(pred1, 1-poisoned_vector)) / len(poisoned_vector)
    Recall = np.sum((1-pred1)*poisoned_vector) / np.sum(poisoned_vector)
    print('\neval_train acc: %.5f recall: %.5f' % (correct, Recall))

    # True Positive: predicted poisoned, actually poisoned
    TP = np.sum((pred1 == 0) & (poisoned_vector == 1))

    # False Positive: predicted poisoned, actually benign
    FP = np.sum((pred1 == 0) & (poisoned_vector == 0))

    # True Negative: predicted benign, actually benign
    TN = np.sum((pred1 == 1) & (poisoned_vector == 0))

    # False Negative: predicted benign, actually poisoned
    FN = np.sum((pred1 == 1) & (poisoned_vector == 1))

    # Metrics
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall / Sensitivity
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    # Accuracy for reference
    accuracy = np.mean(pred1 == 1 - poisoned_vector)

    print('\neval_train acc: %.5f TPR: %.5f FPR: %.5f' % (accuracy, TPR, FPR))
    print('\n number of FN: %d' % (FN))
    return np.ones_like(losses), pred1, threshold, pre_loss


def warmup(args, epoch, net, optimizer, dataloader, device):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)

        #penalty = conf_penalty(outputs)
        L = loss #+ penalty

        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s| Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()

def train_poisoned_model(args, epoch, net, optimizer, labeled_trainloader, device):
    net.train()

    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, _, _, labels_x, w_x) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, labels_x, w_x = inputs_x.to(device), labels_x.to(device), w_x.to(device)

        logits = net(inputs_x)
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * labels_x, dim=1))
        loss = Lx

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                         % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            Lx.item(), 0))
        sys.stdout.flush()


def test(epoch, net1, net2, test_loader1, test_loader2, test_log, device):
    net1.eval()
    net2.eval()
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader1):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1, _ = net1(inputs)
            _, predicted1 = torch.max(outputs1, 1)
            total1 += targets.size(0)
            correct1 += predicted1.eq(targets).cpu().sum().item()

        for batch_idx, (inputs, targets, _) in enumerate(test_loader2):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs2 = net2(inputs)
            _, predicted2 = torch.max(outputs2, 1)
            total2 += targets.size(0)
            correct2 += predicted2.eq(targets).cpu().sum().item()

    acc1 = 100. * correct1 / total1
    acc2 = 100. * correct2 / total2
    print("| Test Epoch #%d\t Accuracy: %.2f%%\t Accuracy: %.2f%%" % (epoch, acc1, acc2))
    test_log.write('Epoch:%d   Accuracy1:%.2f   Accuracy2:%.2f\n' % (epoch, acc1, acc2))
    test_log.flush()
    return acc1, acc2

def save_state(epoch, net, optimizer, poisoned_index, noisy_idx, pre_loss, threshold, path):
    saved_dict = {
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "poisoned_index": poisoned_index,
        "noisy_index": noisy_idx,
        "pre_loss": pre_loss,
        "threshold": threshold
    }
    torch.save(saved_dict, path)

def load_state(net, optimizer, path):
    ckpt = torch.load(path)
    net.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    poisoned_index = ckpt["poisoned_index"]
    epoch = ckpt["epoch"]
    noisy_idx = ckpt["noisy_index"]
    threshold = None
    pre_loss = None
    if "threshold" in ckpt.keys():
        threshold = ckpt["threshold"]
    if "pre_loss" in ckpt.keys():
        pre_loss = ckpt["pre_loss"]

    return epoch, poisoned_index, noisy_idx, pre_loss, threshold
