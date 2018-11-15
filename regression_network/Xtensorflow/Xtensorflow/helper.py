import tensorflow as tf
import numpy as np

# ============================ACTIVATION FUNCTION=====================
def lrelu(x):
    #leaky relu
    return tf.maximum(x * 0.2, x)

def prelu(_x):
    # leaky relu
    alphas = tf.get_variable('alpha', _x.get_shape()[-1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0), collections=xnet_collections)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

# ============================WEIGHTS HELPER=========================
def write_biases(pf, param_dict, num ):
    nparam = 1
    for param in param_dict:

        wtmp = param.reshape([-1])
        for ntmp in wtmp:
            if nparam % num == 0:
                pf.write('\n')
            pf.write('%f, ' % ntmp)
            nparam += 1

def write_weights(pf, param_dict, num):
    nparam = 1
    for param in param_dict:
        k_w = param.shape[0]

        for o in range(k_w):
            wtmp = param[o]
            if nparam % num == 0:
                pf.write('\n')
            pf.write('%f, ' % wtmp)
            nparam += 1

def read_from_txt(txt):
    f = open(txt, 'r')

    weights = []

    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        line = [float(tmp) for tmp in line]

        weights += line

    weights = np.asarray(weights).astype(np.float32).reshape([-1])

    return weights



# ============================VARIABLE HELPER=========================
def get_variables_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.

    Examples
    ---------
    >>> vars = get_variable_with_name('dense', True, True)
    """

    print("  [*] geting variables with %s" % name)

    if train_only:
        t_vars = tf.trainable_variables()
    else:
        t_vars = tf.global_variables()


    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

    return d_vars