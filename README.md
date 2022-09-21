Seen attacks

without regularization

```
python3 adv26b.py -dat cifar10 -mod resnet110 -att wbf -lmt 0 -tau 0 -bas 100 -vls 0.9 -tss 0.1 -see 0
```

with regularization
```
python3 adv26b.py -dat cifar10 -mod resnet110 -att wbf -lmt 1 -tau 1 -bas 100 -vls 0.9 -tss 0.1 -see 0
```

Unseen attacks

without regularization

```
python3 adv26b-gen.py -dat cifar10 -mod resnet110 -att wb -lmt 0 -tau 0 -bas 100 -vls 0.9 -tss 0.1 -see 0
```

with regularization
```
python3 adv26b-gen.py -dat cifar10 -mod resnet110 -att wb -lmt 1 -tau 1 -bas 100 -vls 0.9 -tss 0.1 -see 0
```
