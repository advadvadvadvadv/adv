# van, rce, lap
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from dataloaders8 import dataloaders
from utils6 import *
from torchsummary import summary
import time, math, numpy as np, matplotlib.pyplot as plt, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial
from models import *


def get_model(datashape, modelname, n_classes):
    if modelname == 'resnext50':
        return ResNext50(datashape, n_classes)
    elif modelname == 'onerep':
        return  OneRepResNet(datashape, n_classes)
    elif modelname == 'resnet110':
        return  ResNet110(datashape, n_classes)
    elif modelname == 'wide':
        return  WideResNet(datashape, n_classes)
    elif modelname == 'efficientnet':
        return EfficientNet(datashape, n_classes)
    else:
        raise NotImplementedError()



class ReverseCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes):
        super(ReverseCrossEntropyLoss, self).__init__()
        self.n_classes = n_classes
        self.logsoftmax = nn.LogSoftmax(dim = 1)
    def forward(self, out, y) :
        out = self.logsoftmax(out)
        reverse_label_one_hot = (functional.one_hot(y, self.n_classes).float() - 1) / (self.n_classes - 1)
        rce =  torch.sum(out * reverse_label_one_hot, dim = 1)
        return rce.mean()

def train(model, optimizer, scheduler, criterion, trainloader, valloader, testloader, rce, pnorm, lmt0, lml0, tau, uzs, con, nepochs, clip = 0):
    train_loss, train_acc, val_loss, val_acc, lmt, lml, t0, it, uzawa = [], [], [], [], lmt0, lml0, time.time(), 0, uzs > 0 and tau > 0
    lmt = 1 / lml if uzawa else lmt
    print('\n--- Begin trainning\n')
    for e in range(nepochs):
        model.train()
        t1, loss_meter, acc_meter, time_meter = time.time(), AverageMeter(), AverageMeter(), AverageMeter()
        for j, (x, y) in enumerate(trainloader):
            t2, it = time.time(), it + 1
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, rs = model(x)
            classification_loss = criterion(out, y)
            if uzawa and it % uzs == 0 :
                lml += tau * classification_loss.item()
                lmt = 1 / lml
                if con:
                    continue
            if lmt > 0 : 
                transport = sum([torch.mean(r ** pnorm) for r in rs])
                loss = classification_loss + lmt * transport
            else :
                loss = classification_loss
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            num = len(y)
            _, pred = torch.max(- out.data, 1) if rce else torch.max(out.data, 1)
            update_meters(y, pred, loss.item(), loss_meter, acc_meter, t = time.time() - t2, time_meter = time_meter)
            if j % 500 == 0 :
                m = (e + 1, nepochs, j + 1, len(trainloader), loss_meter.avg, acc_meter.avg, lml, time_meter.avg)
                print('[Ep {:^5}/{:^5} Batch {:^5}/{:^5}] Train loss {:9.4f} Train top1acc {:.4f} Lambda loss {:9.4f} Batch time {:.4f}s'.format(*m))
        train_loss.append(loss_meter.avg)
        train_acc.append(acc_meter.avg)
        optimizer.zero_grad()
        vlo, vac = test(model, criterion, rce, valloader)
        val_loss.append(vlo)
        val_acc.append(vac)
        m = (e + 1, nepochs, vlo, vac, time.time() - t1, time.time() - t0)
        print('\n[***** Ep {:^5}/{:^5} over ******] Valid loss {:9.4f} Valid top1acc {:.4f} Epoch time {:9.4f}s Total time {:.4f}s\n'.format(*m))
        scheduler.step()
    test_loss, test_acc = test(model, criterion, rce, testloader)
    return train_loss, val_acc

def test(model, criterion, rce, loader):
    model.eval()
    loss_meter, acc_meter, = AverageMeter(), AverageMeter()
    for j, (x, y) in enumerate(loader):
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            out, residus = model(x)
            ent = criterion(out, y)
            num = len(y)
            _, pred = torch.max(- out.data, 1) if rce else torch.max(out.data, 1)
            update_meters(y, pred, ent.item(), loss_meter, acc_meter)
    return loss_meter.avg, acc_meter.avg
    

def experiment(dataset, modelname, pnorm, nfilters, learningrate, lambdatransport, lambdaloss0, tau, uzawasteps, batchnorm, bias, rce, timestep, clip, classifier,
               nblocks, con, nepochs, init, initname, initgain, trainsize, valsize, testsize, noise, batchsize, relu, residu, weightdecay, seed = None, experiments = False):

    t0 = time.time()
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    trainloader, valloader, testloader, datashape, n_classes, mean, std = dataloaders(dataset, batchsize, trainsize, valsize, testsize, noise)

    if nepochs > 100:
        if rce :
            folder = modelname + '-' + dataset + '-' + time.strftime("%Y%m%d-%H%M%S") + '-rce'
        else :
            folder = modelname + '-' + dataset + '-' + time.strftime("%Y%m%d-%H%M%S") + '-lt' + str(lambdatransport) + '-ll' + str(lambdaloss0) + '-ta' + str(tau) + ('-prerelu' if not residu else '')
        make_folder(folder)
        stdout0 = sys.stdout
        sys.stdout = open(os.path.join(folder, 'log.txt'), 'wt')
    frame = inspect.currentframe()
    names, _, _, values = inspect.getargvalues(frame)
    print('experiment from main.py with parameters')
    for name in names:
        print('%s = %s' % (name, values[name]))
    if rce:
        print('rce loss means no transport loss')
        lambdaloss0, tau, lambdatransport, uzawasteps = 1, 0, 0, 0
    if uzawasteps == 0 and (lambdaloss0 != 1 or tau > 0):
        print('us = 0 means no transport loss. lambda loss is fixed to 1, tau to 0, and lambda transport to 0')
        lambdaloss0, tau, lambdatransport = 1, 0, 0
    if uzawasteps > 0 and lambdatransport != 1:
        print('us > 0 means uzawa. lambda transport is fixed to 1')
        lambdatransport = 1
    print('train batches', len(trainloader), 'val batches', len(valloader))

    model = get_model(datashape, modelname, n_classes)
    if torch.cuda.device_count() > 1:
        print('\n---', torch.cuda.device_count(), 'GPUs \n')
        model = nn.DataParallel(model)
    initialization = partial(initialize, initname, initgain)
    if init:
        for name, module in model.named_modules():
            module.apply(initialization) 
    model.to(device)
    print(model)
    
    criterion = nn.CrossEntropyLoss() if not rce else ReverseCrossEntropyLoss(n_classes)
    optimizer = optim.SGD(model.parameters(), lr = learningrate, momentum = 0.9, weight_decay = weightdecay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150, 225, 250], gamma = 0.1)

    train_loss, val_acc = train(model, optimizer, scheduler, criterion, trainloader, valloader, testloader, rce, pnorm, lambdatransport, lambdaloss0, tau, uzawasteps, con, nepochs, clip)
    
    if experiments and nepochs > 5:
        print('--- Train Loss \n', train_loss, '\n--- Val Acc1 \n', val_acc)
        print('--- Min Train Loss \n', min(train_loss), '\n--- Max Val Acc \n', max(val_acc))
        sys.stdout.close()
        sys.stdout = stdout0

    if not experiments and nepochs > 5:
        print('here')
        torch.save(model.state_dict(), os.path.join(folder, 'weights.pth'))
    return train_loss, val_acc, time.time() - t0

def experiments(parameters, average):
    t0, j, f = time.time(), 0, 110
    sep = '-' * f 
    accs = []
    nparameters = len(parameters)
    nexperiments = int(np.prod([len(parameters[i][1]) for i in range(nparameters)]))
    print('\n' + sep, 'main.py')
    print(sep, nexperiments, 'experiments ' + ('to average ' if average else '') + 'over parameters:')
    pprint.pprint(parameters, width = f, compact = True)
    for params in product([values for name, values in parameters]) :
        j += 1
        print('\n' + sep, 'experiment %d/%d with parameters:' % (j, nexperiments))
        pprint.pprint([parameters[i][0] + ' = ' + str(params[i]) for i in range(nparameters)], width = f, compact = True)
        tr_loss, vl_acc, t1 = experiment(*params, True)
        accs.append(np.max(vl_acc))
        print(sep, 'experiment %d/%d over. took %.1f s. total %.1f s' % (j, nexperiments, t1, time.time() - t0))

    if average:
        acc = np.mean(accs)
        confint = st.t.interval(0.95, len(accs) - 1, loc = acc, scale = st.sem(accs))
        print('\nall test acc', accs)
        print('\naverage test acc', acc)
        print('\nconfint', confint)

        
    print(('\n' if not average else '') + sep, 'total time for %d experiments: %.1f s' % (j, time.time() - t0))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-dat", "--dataset", required = True, choices = ['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet2012', 'tinyimagenet', 'imagenetdownloader'], nargs = '*')
    parser.add_argument("-mod", "--modelname", required = True, choices = ['resnext29', 'resnext50', 'onerep', 'resnet110', 'avgpool', 'wide', 'efficientnet'], nargs = '*')
    parser.add_argument("-pno", "--pnorm", type = int, default = [2], nargs = '*')
    parser.add_argument("-nfl", "--nfilters", type = int, default = [64], nargs = '*')
    parser.add_argument("-lrr", "--learningrate", type = float, default = [0.1], nargs = '*')
    parser.add_argument("-lmt", "--lambdatransport", type = float, default = [0], nargs = '*')
    parser.add_argument("-lml", "--lambdaloss0", type = float, default = [1], nargs = '*')
    parser.add_argument("-tau", "--tau", type = float, default = [0], nargs = '*')
    parser.add_argument("-uzs", "--uzawasteps", type = int, default = [0], nargs = '*')
    parser.add_argument("-btn", "--batchnorm", type = int, default = [1], choices = [0, 1], nargs = '*')
    parser.add_argument("-bia", "--bias", type = int, default = [0], choices = [0, 1], nargs = '*')
    parser.add_argument("-rce", "--reversecrossentropy", type = int, default = [0], choices = [0, 1], nargs = '*')
    parser.add_argument("-tst", "--timestep", type = float, default = [1], nargs = '*')
    parser.add_argument("-clp", "--clip", type = float, default = [0], nargs = '*')
    parser.add_argument("-cla", "--classifier", default = ['3Lin'], choices = ['1Lin', '2Lin', '3Lin'], nargs = '*')
    parser.add_argument("-nbl", "--nblocks", type = int, default = [9], nargs = '*')
    parser.add_argument("-con", "--continue", type = int, default = [0], choices = [0, 1], nargs = '*')
    parser.add_argument("-nep", "--nepochs", type = int, default = [300], nargs = '*')
    parser.add_argument("-ini", "--init", type = int, default = [1], choices = [0, 1], nargs = '*')
    parser.add_argument("-inn", "--initname", default = ['kaiming'], choices = ['orthogonal', 'normal', 'kaiming'], nargs = '*')
    parser.add_argument("-ing", "--initgain", type = float, default = [0.01], nargs = '*')
    parser.add_argument("-trs", "--trainsize", type = float, default = [None], nargs = '*')
    parser.add_argument("-vls", "--valsize", type = float, default = [None], nargs = '*')
    parser.add_argument("-tss", "--testsize", type = float, default = [None], nargs = '*')
    parser.add_argument("-noi", "--noisestd", type = float, default = [0], nargs = '*')
    parser.add_argument("-bas", "--batchsize", type = int, default = [128], nargs = '*')
    parser.add_argument("-rel", "--relu", type = int, default = [1], choices = [0, 1], nargs = '*')
    parser.add_argument("-res", "--residu", type = int, default = [1], choices = [0, 1], nargs = '*')
    parser.add_argument("-wdc", "--weightdecay", type = float, default = [0.0001], nargs = '*')
    parser.add_argument("-see", "--seed", type = int, default = [None], nargs = '*')
    parser.add_argument("-exp", "--experiments", action = 'store_true')
    parser.add_argument("-avg", "--averageexperiments", action = 'store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.experiments or args.averageexperiments:
        parameters = [(name, values) for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
        experiments(parameters, args.averageexperiments)
    else :
        parameters = [values[0] for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
        experiment(*parameters, False)