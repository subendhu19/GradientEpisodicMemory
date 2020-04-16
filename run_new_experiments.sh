#!/usr/bin/env bash

MY_PYTHON="python"
MNIST_ROTA="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda no  --seed 0"
MNIST_PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda no  --seed 0"
CIFAR_100i="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar100.pt           --cuda yes --seed 0"

echo EWC
$MY_PYTHON main.py $MNIST_ROTA --model ewc --lr 0.1 --n_memories 256 --memory_strength 1000
$MY_PYTHON main.py $MNIST_PERM --model ewc --lr 0.1 --n_memories 256 --memory_strength 3


echo GEM
$MY_PYTHON main.py $MNIST_ROTA --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5
$MY_PYTHON main.py $MNIST_PERM --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5

echo CUSTOM
for lr in 0.01 0.1
do
    for memgem in 0.1 0.5 1
    do
        for memewc in 1 3 10
        do
        # model "EWC+GEM 1"
        $MY_PYTHON main.py $MNIST_ROTA --model ewgem1 --lr ${lr} --n_memories 256 --memory_strength_gem ${memgem} --memory_strength_ewc ${memewc}
        $MY_PYTHON main.py $MNIST_PERM --model ewgem1 --lr ${lr} --n_memories 256 --memory_strength_gem ${memgem} --memory_strength_ewc ${memewc}

        # model "EWC+GEM 2"
        $MY_PYTHON main.py $MNIST_ROTA --model ewgem2 --lr ${lr} --n_memories 256 --memory_strength_gem ${memgem} --memory_strength_ewc ${memewc}
        $MY_PYTHON main.py $MNIST_PERM --model ewgem2 --lr ${lr} --n_memories 256 --memory_strength_gem ${memgem} --memory_strength_ewc ${memewc}
        done
    done
done