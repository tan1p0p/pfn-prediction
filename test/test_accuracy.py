import numpy as np
from chainer import Variable

from src.modules.accuracy import joint_accuracy

def test_joint_accuracy_max():
    y_np = np.zeros((2, 3, 4, 4))
    y_np[:, :, 0, 0] = 1
    y = Variable(y_np)

    t = np.zeros((2, 3, 4, 4))
    t[:, :, 3, 3] = 1

    assert joint_accuracy(y, t) == 1

def test_joint_accuracy_min():
    y_np = np.zeros((2, 3, 4, 4))
    y_np[:, :, 2, 2] = 1
    y = Variable(y_np)

    t = np.zeros((2, 3, 4, 4))
    t[:, :, 2, 2] = 1

    assert joint_accuracy(y, t) == 0
