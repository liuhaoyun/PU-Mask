# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs

##################################################################################
# Back projection Blocks
##################################################################################
def PointShuffler(inputs, scale=2):
    #inputs: B x N x 1 X C
    #outputs: B x N*scale x 1 x C//scale
    outputs = tf.reshape(inputs,[tf.shape(inputs)[0],tf.shape(inputs)[1],1,tf.shape(inputs)[3]//scale,scale])
    outputs = tf.transpose(outputs,[0, 1, 4, 3, 2])

    outputs = tf.reshape(outputs,[tf.shape(inputs)[0],tf.shape(inputs)[1]*scale,1,tf.shape(inputs)[3]//scale])

    return outputs

from Common.model_utils import gen_1d_grid,gen_grid


def Chain_Residual_Block(input, output=128, block_num=4, scope='chain_residual_block',
                         is_training=True,  use_bn=False, use_ibn=False, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        identity = input
        sum_residual = identity  # 存储每个链的残差

        for i in range(block_num):
            if i == 0:
                feature, residual = Rssidual_Block(identity, output, scope='residual_block%d' % i, is_training=is_training,
                                             use_bn = use_bn, use_ibn = use_ibn, bn_decay=bn_decay)
                sum_residual = residual
            else:
                feature, residual = Rssidual_Block(feature, output, scope='residual_block%d' % i, is_training=is_training,
                                             use_bn = use_bn, use_ibn = use_ibn, bn_decay=bn_decay)
                sum_residual = tf.concat([sum_residual, residual], axis=-1)  # concat residuals  [n * 128]

        sum_residual = SE_NET(sum_residual, scope='se_net',is_training=is_training)  #  attention
        sum_residual = conv2d(sum_residual, output, [1, 1],
                              padding='VALID', scope='layer_compress', is_training=is_training, bn=use_bn, ibn=use_ibn,
                              bn_decay=bn_decay, activation_fn=None)  # [128]  No Relu
        out = identity + sum_residual
        out = tf.nn.relu(out)
    return out

def RCB_conv(input, k, output=256, block_num=3, layer=1, scope='CRB', is_training=True,
             use_bn=False, use_ibn=False, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        y, idx = get_edge_feature(input, k=k, idx=None)  # [B N K 2*C]
        for i in range(layer):
            y = Chain_Residual_Block(y, output, block_num, scope='chain_residual_block%d' % i,
                                     is_training=is_training, use_bn=use_bn, use_ibn=use_ibn, bn_decay=bn_decay)  # [B, N, 128]

        y = conv2d(y, output, [1, 1], padding='VALID', scope='Adjust_layer', activation_fn = None,
                   is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  #  Conv without Relu


        y = tf.reduce_max(y, axis=-2)  # [B, N, C]

        return y, idx

def feature_extraction_RCB(inputs, scope='feature_extraction2', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        use_bn = False
        use_ibn = False
        growth_rate = 24

        dense_n = 3
        knn = 15
        comp = growth_rate
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, growth_rate, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)

        # encoding layer
        l1_features, l1_idx = RCB_conv(l0_features, k=knn, output= growth_rate * 2, scope="layer1",
                                       is_training=is_training,bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # 96

        l2_features = conv1d(l1_features, comp*2, 1,  padding='VALID', scope='layer2_prep',
                             is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)

        l2_features, l2_idx  = RCB_conv(l2_features, k=knn, output= growth_rate * 4, scope="layer2",
                                       is_training=is_training,bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 224

        l3_features = conv1d(l2_features, comp*3, 1,  # 48
                                     padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l3_features, l3_idx = RCB_conv(l3_features, k=knn, output= growth_rate * 6, scope="layer3",
                                       is_training=is_training,bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 352

        l4_features = conv1d(l3_features, comp*3, 1,  # 48
                                     padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l4_features, l4_idx = RCB_conv(l4_features, k=knn, output= growth_rate * 6, scope="layer4",
                                       is_training=is_training,bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 480

        l4_features = tf.expand_dims(l4_features, axis=2)

    return l4_features

# 残差块
# input :  (B, N, 1, C)
# output : (B, N ,1, C)
def Rssidual_Block(input, C_OUT, scope='residual_block', is_training=True,
                   use_ibn=False, use_bn=False, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        gamma = 4  # bottleNeck ratio
        x = input
        # RB : Conv + Bn + Relu
        residual = conv2d(input, C_OUT//gamma, [1, 1],
                     padding='VALID', scope='bottle_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                     bn_decay=bn_decay)

        residual = conv2d(residual, C_OUT, [1, 1],
                          padding='VALID', scope='bottle_2', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay, activation_fn=None)  # Conv + bn

        y = x + residual
        y = tf.nn.relu(y)
    return y, residual


def Pre_upsampling(xyz, input, up_ratio, dim=256, scope='Pre-upsampling',
                               is_training=True,  use_bn=False, use_ibn=False, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        knn = 12
        # feature knn
        knn_xyz, idx = get_KNN_feature(xyz, k=knn)  # [B,N,K,3]
        central_xyz = tf.expand_dims(xyz, axis=2)  # [B,N,1,3]
        central_xyz = tf.tile(central_xyz, [1, 1, knn, 1])  # [B,N,K,3]
        mask = tf.concat([central_xyz - knn_xyz],axis = -1)
        mask = tf.reshape(mask, [tf.shape(xyz)[0], tf.shape(xyz)[1], 1, -1])
        mask = conv2d(mask, dim//2, [1, 1], padding='VALID', scope='prelayer_1', is_training=is_training,
                      bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # Conv + Relu
        features_global = tf.reduce_max(mask, axis=1, keep_dims=True, name='maxpool')
        mask = tf.concat([mask, tf.tile(features_global, [1, tf.shape(mask)[1], 1, 1])], axis=-1)
        mask = conv2d(mask, dim, [1, 1], padding='VALID', scope='prelayer_2',is_training=is_training,
                      bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # Conv + Relu

    return mask

def Mask_Feature_Expand(xyz, input, mask, up_ratio, output=512, scope='Masked_Transformer',
                               is_training=True,  use_bn=False, use_ibn=False, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        grid_dim = 2
        grid = gen_grid(up_ratio)
        grid = tf.tile(tf.expand_dims(grid, 0),[tf.shape(xyz)[0], 1, tf.shape(xyz)[1]])  # [batch_size, num_point*4, 2])
        grid = tf.reshape(grid, [tf.shape(xyz)[0], -1, 1, grid_dim])
        grid0 = grid[:, 0:tf.shape(xyz)[1], :, :]
        mask1 = grid[:, tf.shape(xyz)[1]:tf.shape(xyz)[1] * 2, :, :]
        net = tf.concat([input, grid0], axis = -1)
        mask_point = tf.concat([mask, mask1], axis=-1)
        # align feature
        net = conv2d(net, 256, [1, 1], padding='VALID', scope='Adjust_layer',
                         is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # Conv + Relu
        # Encoder
        net = attention_unit(net, scope='Encoder_L0_Head0',is_training=is_training)
        # Multi-head Transformer Decoder
        mask_point = conv2d(mask_point, 512, [1, 1], padding='VALID', scope='MLP_Layer1',
                            is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # Conv + Relu
        up = tf.concat([net, mask_point], axis = -1)
        L0H0 = attention_unit(up, scope='Decoder_L0_Head0', is_training=is_training)
        L0H1 = attention_unit(up, scope='Decoder_L0_Head1', is_training=is_training)
        L0 = tf.concat([L0H0,L0H1],axis = -1)
        L0 = conv2d(L0, output, [1, 1], padding='VALID', scope='MLP_Layer2',
                            is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # Conv + Relu
        L1H0 = attention_unit(L0, scope='Decoder_L1_Head0', is_training=is_training)
        L1H1 = attention_unit(L0, scope='Decoder_L1_Head1', is_training=is_training)
        L1 = tf.concat([L1H0, L1H1], axis=-1)
        L1 = conv2d(L1, output, [1, 1], padding='VALID', scope='MLP_Layer3',
                    is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # Conv + Relu
        L2H0 = attention_unit(L1, scope='Decoder_L2_Head0', is_training=is_training)
        L2H1 = attention_unit(L1, scope='Decoder_L2_Head1', is_training=is_training)
        L2 = tf.concat([L2H0, L2H1], axis=-1)
        L2 = conv2d(L2,output*2+512 , [1, 1], padding='VALID', scope='Linear',
                is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay, activation_fn=None)  # Conv , No Relu

    return L2

def Coordinate_Refine(xyz_up, features, out_dim = 128, layer = 1, knn = 26, scope = 'Coordinate_refine', is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # geometry knn
        xyz = tf.squeeze(xyz_up, [2])
        knn_xyz, idx = get_KNN_feature(xyz, k=knn)  # [B,N,K,3]
        central_xyz = tf.expand_dims(xyz, axis = 2) # [B,N,1,3]
        central_xyz = tf.tile(central_xyz, [1, 1, knn, 1]) # [B,N,K,3]
        edge_xyz = tf.concat([central_xyz, central_xyz-knn_xyz], axis = -1)
        edge_xyz = conv2d(edge_xyz, out_dim//4, [1, 1], padding='VALID', scope='MLP_1',is_training=is_training)  # 使用 Conv + Relu
        edge_xyz = conv2d(edge_xyz, out_dim//2, [1, 1], padding='VALID', scope='MLP_2', is_training=is_training,activation_fn=None)  # 使用 Conv + Relu
        for i in range(layer):
            edge_xyz, residual = Rssidual_Block(edge_xyz, out_dim//2, scope='residual_layer%d' % i)
        edge_xyz = conv2d(edge_xyz, out_dim, [1, 1], padding='VALID', scope='MLP_3', is_training=is_training,activation_fn=None)
        feature = conv2d(features, out_dim, [1, 1], padding='VALID', scope='MLP_4', is_training=is_training,activation_fn=None)
        feature =  tf.tile(feature, [1, 1, knn, 1]) # [B,N,K,3]
        res = edge_xyz - feature
        res = tf.reduce_max(res, axis=2, keep_dims=True)
        coord = conv2d(res, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer1', bn_decay=None)

        coord = conv2d(coord, 3, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer2', bn_decay=None,
                           activation_fn=None, weight_decay=0.0)

        return coord

# input ： (B, N, 1，C)
# output : (B, N ,1, C)
def SE_NET(input, scope='se_net', is_training=True, bn_decay=None, use_bn = False, use_ibn=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        B, N, K, C = [i.value for i in input.get_shape()]
        mean = tf.reduce_mean(input, axis=[1, 2])  # Global Average Pooling [B C]
        mean = tf.expand_dims(tf.expand_dims(mean, axis=1), axis=2) # [B 1 1 C]
        mean_knn = tf.tile(mean, [1,N,K,1]) # [B,N,K,C]
        res = input - mean_knn
        res = conv2d(res, C, [1, 1],padding='VALID', scope='SE_0',is_training=is_training,
                     bn=use_bn, ibn=use_ibn, bn_decay=bn_decay, activation_fn=None)
        res = tf.reduce_max(res, axis=[1, 2])  # Global Average Pooling [B C]
        res = tf.expand_dims(tf.expand_dims(res, axis=1), axis=2)  # [B 1 1 C]
        feature = mean + res
        feature = conv2d(feature, C//16, [1, 1],
                          padding='VALID', scope='SE_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay)
        feature = conv2d(feature, C, [1, 1],
                          padding='VALID', scope='SE_2', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay, activation_fn=None)

        scale = tf.sigmoid(feature)  # [B,1,1,C]
        out = input * scale
    return out


def attention_unit(inputs, scope='attention_unit',is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape()[-1].value
        layer = dim//4
        f = conv2d(inputs,layer, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv_f', bn_decay=None)

        g = conv2d(inputs, layer, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_g', bn_decay=None)

        h = conv2d(inputs, dim, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_h', bn_decay=None)

        # channel attention
        h = SE_NET(h, scope='GCSE', is_training=is_training)  # attention

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

    return x


##################################################################################
# Other function
##################################################################################
def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=tf.AUTO_REUSE):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias = True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs,num_outputs,
                                  use_bias=use_bias,kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

from tf_ops.grouping.tf_grouping import knn_point_2
def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx

def get_KNN_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)

    return point_cloud_neighbors, idx

def dense_conv(feature, n=3,growth_rate=64, k=16, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y,idx

def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keep_dims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keep_dims=True)), axis=1, keep_dims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)


def tf_covariance(data):
    ## x: [batch_size, num_point, k, 3]
    batch_size = data.get_shape()[0].value
    num_point = data.get_shape()[1].value

    mean_data = tf.reduce_mean(data, axis=2, keep_dims=True)  # (batch_size, num_point, 1, 3)
    mx = tf.matmul(tf.transpose(mean_data, perm=[0, 1, 3, 2]), mean_data)  # (batch_size, num_point, 3, 3)
    vx = tf.matmul(tf.transpose(data, perm=[0, 1, 3, 2]), data) / tf.cast(tf.shape(data)[0], tf.float32)  # (batch_size, num_point, 3, 3)
    data_cov = tf.reshape(vx - mx, shape=[batch_size, num_point, -1])

    return data_cov



def add_scalar_summary(name, value,collection='train_summary'):
    tf.summary.scalar(name, value, collections=[collection])
def add_hist_summary(name, value,collection='train_summary'):
    tf.summary.histogram(name, value, collections=[collection])

def add_train_scalar_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])

def add_train_hist_summary(name, value):
    tf.summary.histogram(name, value, collections=['train_summary'])

def add_train_image_summary(name, value):
    tf.summary.image(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update
