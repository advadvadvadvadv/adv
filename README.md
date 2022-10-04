Seen attacks

without regularization

```
python3 adv26b.py -dat cifar10 -mod resnet110 -att wbf -lmt 0.0 -tau 0.0 -bas 100 -vls 0.9 -tss 0.1 -see 0
```

with regularization
```
python3 adv26b.py -dat cifar10 -mod resnet110 -att wbf -lmt 1.0 -tau 1.0 -bas 100 -vls 0.9 -tss 0.1 -see 0
```


Unseen attacks

without regularization

```
python3 adv26b-gen.py -dat cifar10 -mod resnet110 -att wb -lmt 0.0 -tau 0.0 -bas 100 -vls 0.9 -tss 0.1 -see 0
```

with regularization
```
python3 adv26b-gen.py -dat cifar10 -mod resnet110 -att wb -lmt 1.0 -tau 1.0 -bas 100 -vls 0.9 -tss 0.1 -see 0
```

Training

To train a vanilla network
```
python3 adv_train_network_b.py -dat cifar100 -mod resnext50
```

To train a LAP network
```
python3 adv_train_network_b.py -dat cifar100 -mod resnext50 -lmt 1 -tau 1 -uzs 1 
```
