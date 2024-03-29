Adversarial detection using Transport and Mahalanobis detectors

```
python3 adv31.py -dat [DATASET] -mod [MODEL] -trt [TRAINTYPE] -set [SETTING] -att [ATTACKS] 
```
DATASET from cifar100, cifar10, tinyimagenet

MODEL from resnext50, resnet110, wide

TRAINTYPE from van, rce, lap

SETTING from seen, unseen

ATTACKS from fgm, pgd, bim, df, cw, auto, hsj, ba, wb, bb, wbf, wbs, all


So to detect seen white-box attacks on a LAP-ResNet110 on CIFAR10

```
python3 adv31.py -dat cifar10 -mod resnet110 -trt lap -set seen -att wb
```

Network training

To train a vanilla network
```
python3 adv_train_network_c.py -dat cifar100 -mod resnext50 -see 0
```

To train a LAP network
```
python3 adv_train_network_c.py -dat cifar100 -mod resnext50 -lmt 1 -tau 1 -uzs 1 -see 0
```

To train an RCE network
```
python3 adv_train_network_c.py -dat cifar100 -mod resnext50 -rce 1 -lrr 0.05 -inn orthogonal -ing 0.05 -see 0
```

OOD detection using Transport and Mahalanobis detectors
```
python3 ood3.py -ind cifar10 -ood1 [OOD1] -ood2 svhn -mod resnet110 -trt van -bas 100 -ivs 0.9 -its 0.1 -o1vs 0.9 -o1ts 0.1 -o2ts 0.03845
```

OOD1 can be cifar100 or an attack from fgm, pgd, bim, df, cw, hsj, ba



adv29 uses ART for AutoAttack while adv31 uses the author's original code. In adv29 the attacker does not know about RCE training and so misinterprets the output of the network which leads to better adversarial detection, while in adv31 the attackers knows to flip the logits. adv31nss tests the natural scene statistics detector.
