
from __future__ import print_function
import torch, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform

def get_random_noise_params(dataset, attack_name):
    min_pixel = - 2.42906570435
    max_pixel = 2.75373125076
    if dataset == 'cifar10':
        if attack_name == 'fgm':
            random_noise_size = 0.25 / 4
        elif attack_name == 'bim' or attack_name == 'pgd':
            random_noise_size = 0.13 / 2
        elif attack_name == 'df':
            random_noise_size = 0.25 / 4
        elif attack_name == 'cw2':
            random_noise_size = 0.05 / 2
        else:
            random_noise_size = 0.13 / 2
    elif dataset == 'cifar100':
        if attack_name == 'fgm':
            random_noise_size = 0.25 / 8
        elif attack_name == 'bim' or attack_name == 'pgd':
            random_noise_size = 0.13 / 4
        elif attack_name == 'df':
            random_noise_size = 0.13 / 4
        elif attack_name == 'cw2':
            random_noise_size = 0.05 / 2
        else:
            random_noise_size = 0.13 / 2
    else:
        if attack_name == 'fgm':
            random_noise_size = 0.25 / 4
        elif attack_name == 'bim' or attack_name == 'pgd':
            random_noise_size = 0.13 / 2
        elif attack_name == 'df':
            random_noise_size = 0.126
        elif attack_name == 'cw2':
            random_noise_size = 0.05 / 1
        else:
            random_noise_size = 0.13 / 2
    return random_noise_size, min_pixel, max_pixel


def get_maha_params(model, datashape, n_classes, trainloader):
    t0 = time.time()
    print('getting mahalanobis train stats...')
    temp_x = torch.rand(2, datashape[1], datashape[2], datashape[3]).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    n_stages = len(temp_list)
    feature_list = np.empty(n_stages)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    sample_mean, precision = sample_estimator(model, n_classes, feature_list, trainloader)
    print('mahalanobis train stats took', time.time() - t0, 'seconds')
    return sample_mean, precision, n_stages

def sample_estimator(model, n_classes, feature_list, trainloader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    n_stages = len(feature_list)
    num_sample_per_class = np.empty(n_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(n_stages):
        temp_list = []
        for j in range(n_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for data, target in trainloader:
        total += data.size(0)
        with torch.no_grad():
            data = data.cuda()
            data = Variable(data)
            output, out_features = model.feature_list(data)
            
            # get hidden features
            for i in range(n_stages):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
                
            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()
            
            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1                
                num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(n_classes, int(num_feature)).cuda()
        for j in range(n_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], dim = 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(n_stages):
        X = 0
        for i in range(n_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    # print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score_adv(model, x, n_classes, sample_mean, precision, layer, magnitude):
    data = Variable(x, requires_grad = True)
    out_features = model.intermediate_forward(data, layer)
    out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
    out_features = torch.mean(out_features, 2)
    gaussian_score = 0
    for i in range(n_classes):
        batch_sample_mean = sample_mean[layer][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = - 0.5 * torch.mm(torch.mm(zero_f, precision[layer]), zero_f.t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1,1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean[layer].index_select(0, sample_pred)
    zero_f = out_features - Variable(batch_sample_mean)
    pure_gau = - 0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer])), zero_f.t()).diag()
    loss = torch.mean(-pure_gau)
    loss.backward()
    gradient =  torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
    tempInputs = torch.add(data.data, gradient, alpha = - magnitude)
    with torch.no_grad():
        noise_out_features = model.intermediate_forward(Variable(tempInputs), layer)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
    for i in range(n_classes):
        batch_sample_mean = sample_mean[layer][i]
        zero_f = noise_out_features.data - batch_sample_mean
        term_gau = - 0.5 * torch.mm(torch.mm(zero_f, precision[layer]), zero_f.t()).diag()
        if i == 0:
            noise_gaussian_score = term_gau.view(-1, 1)
        else:
            noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)      
    noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim = 1)
    return noise_gaussian_score.cpu().numpy()