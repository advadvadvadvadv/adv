Adversarial detection using Transport and Mahalanobis detectors

```
python3 adv28.py -dat [DATASET] -mod [MODEL] -trt [TRAINTYPE] -set [SETTING] -att [ATTACKNAMES] 
```
DATASET from 'cifar10', 'cifar100', 'tinyimagenet'
MODEL from 'resnext50', 'resnet110', 'wide'
TRAINTYPE from 'van', 'rce', 'lap'
SETTING from 'seen', 'unseen'
ATTACKNAMES from 'fgm', 'pgd', 'bim', 'df', 'cw', 'hsj', 'ba', 'wb', 'bb', 'wbf', 'wbs', 'all'

So to detect seen white-box attacks on a LAP-ResNet110 on CIFAR10

```
python3 adv28.py -dat cifar10 -mod resnet110 -trt lap -set seen -att wb
```

Network training

To train a vanilla network
```
python3 adv_train_network_c.py -dat cifar100 -mod resnext50
```

To train a LAP network
```
python3 adv_train_network_c.py -dat cifar100 -mod resnext50 -lmt 1 -tau 1 -uzs 1 
```

To train an RCE network
```
python3 adv_train_network_c.py -dat cifar100 -mod resnext50 -rce 1 -lrr 0.05 -inn orthogonal -ing 0.05
```
