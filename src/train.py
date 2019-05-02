import os
import sys

import numpy as np
import chainer
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer.functions import binary_accuracy
from chainer.iterators import SerialIterator
from chainer.optimizer_hooks import WeightDecay

from modules.accuracy import joint_accuracy
from modules.dataloader import DataLoader
from modules.loss import mean_absolute_error_with_weight
from modules.network import GeneratingStage, RefinementStage

def main():
    # load datasets
    dataloader = DataLoader('./data/raw/hands/hand_position.json')
    dataloader.load_data()
    train, valid, test = dataloader.split()
    train_iter = SerialIterator(train, batch_size=4, repeat=True, shuffle=True)

    # load networks
    first_stage = GeneratingStage()
    second_stage = RefinementStage()
    third_stage = RefinementStage()
    networks = [first_stage, second_stage, third_stage]

    # set optimizer
    optimizers = []
    for i, stage in enumerate(networks):
        optimizers.append(chainer.optimizers.SGD(lr=0.001))
        optimizers[i].setup(stage)
        for param in stage.params():
            if param.name != 'b':
                param.update_rule.add_hook(WeightDecay(0.0001))

    # config
    n_batch = 2
    n_epoch = 5
    # gpu_id = 0

    # train
    results_train = {
        'stage1_loss': [],
        'stage2_loss': [],
        'stage3_loss': [],
    }
    results_valid = {
        'stage1_loss': [],
        'stage2_loss': [],
        'stage3_loss': [],
        'stage1_acc': [],
        'stage2_acc': [],
        'stage3_acc': [],
    }
    # net.to_gpu(gpu_id)

    for epoch in range(n_epoch):
        while True:
            train_batch = train_iter.next()
            x_train, t_train = concat_examples(train_batch) #, gpu_id)

            # train first stage
            output_1 = networks[0](x_train)
            loss = mean_absolute_error_with_weight(output_1, t_train.astype(np.float32))

            networks[0].cleargrads()
            loss.backward()
            optimizers[0].update()
            results_train['stage1_loss'].append(loss.array.mean())

            # train second and third stage
            present_output = output_1
            for i in range(len(networks) - 1):
                current_output = networks[i + 1](x_train, present_output)
                loss = mean_absolute_error_with_weight(current_output, t_train.astype(np.float32))

                networks[i + 1].cleargrads()
                loss.backward()
                optimizers[i + 1].update()
                results_train['stage{}_loss'.format(i + 1)].append(loss.array.mean())
                present_output = current_output

            # validation
            ## TODO: loss, acc to_cpu
            if train_iter.is_new_epoch:
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    x_valid, t_valid = chainer.dataset.concat_examples(valid) #, gpu_id)
                    output_1 = networks[0](x_valid)
                    stage1_loss = mean_absolute_error_with_weight(output_1, t_valid.astype(np.float32)).array.mean()
                    stage1_acc = joint_accuracy(output_1, t_valid)

                    output_2 = networks[1](x_valid, output_1)
                    stage2_loss = mean_absolute_error_with_weight(output_2, t_valid.astype(np.float32)).array.mean()
                    stage2_acc = joint_accuracy(output_2, t_valid)

                    output_3 = networks[2](x_valid, output_2)
                    stage3_loss = mean_absolute_error_with_weight(output_3, t_valid.astype(np.float32)).array.mean()
                    stage3_acc = joint_accuracy(output_3, t_valid)

                print('epoch {}: valid_stage3_loss: {:.4f}, valid_stage3_acc: {:.4f}'.format(
                    epoch, stage3_loss, stage3_acc))

                results_valid['stage1_loss'].append(stage1_loss)
                results_valid['stage2_loss'].append(stage2_loss)
                results_valid['stage3_loss'].append(stage3_loss)
                results_valid['stage1_acc'].append(stage1_acc)
                results_valid['stage2_acc'].append(stage2_acc)
                results_valid['stage3_acc'].append(stage3_acc)

                break

if __name__ == "__main__":
    main()