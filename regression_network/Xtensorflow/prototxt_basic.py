import sys

def data(txt_file):
  txt_file.write('name: "model"\n')
  txt_file.write('layer {\n')
  txt_file.write('  name: "data"\n')
  txt_file.write('  type: "Input"\n')
  txt_file.write('  top: "data"\n')
  txt_file.write('  input_param {\n')
  txt_file.write('    shape: { dim: 1 dim: 1 dim: 128 dim: 128 }\n') # TODO
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def Convolution(txt_file, info):
  if info['attrs']['no_bias'] == 'True':
    bias_term = 'false'
  else:
    bias_term = 'true'  
  txt_file.write('layer {\n')
  txt_file.write('	bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('	top: "%s"\n'          % info['top'])
  txt_file.write('	name: "%s"\n'         % info['top'])
  txt_file.write('	type: "Convolution"\n')
  txt_file.write('	convolution_param {\n')
  txt_file.write('		num_output: %s\n'   % info['attrs']['num_filter'])
  txt_file.write('		kernel_size: %s\n'  % info['attrs']['kernel']) # TODO

  # if info['attrs'].has_key('pad'):
  txt_file.write('		pad: %s\n'           % info['attrs']['pad']) # TODO

  # txt_file.write('		group: %s\n'        % info['attrs']['num_group'])
  txt_file.write('		stride: %s\n'       % info['attrs']['stride'])
  txt_file.write('		bias_term: %s\n'    % bias_term)
  txt_file.write('	}\n')
  # if 'share' in info.keys() and info['share']:
  #   txt_file.write('	param {\n')
  #   txt_file.write('	name: "%s"\n'     % info['params'][0])
  #   txt_file.write('	}\n')
  txt_file.write('}\n')
  txt_file.write('\n')


  #TODO: inplace activation
  if info['activation'] != 'ACTIVE_LINEAR':
      txt_file.write('layer {\n')
      txt_file.write('  bottom: "%s"\n'       % info['name'])
      txt_file.write('  top: "%s"\n'          % info['name'])
      txt_file.write('  name: "%s"\n'         % (info['name'] + '_activation'))

      txt_file.write('  type: "ReLU"\n')
      # if info['activation'] == 'ACTIVE_TANH':
      #     txt_file.write('  type: "TanH"\n')
      # elif info['activation'] == 'ACTIVE_RECTIFIED_LINEAR':
      #     txt_file.write('  type: "ReLU"\n')
      txt_file.write('}\n')
      txt_file.write('\n')

  # txt_file.write('layer {\n')
  # txt_file.write('  bottom: "%s"\n' % info['name'])
  # txt_file.write('  top: "%s"\n' % info['name'])
  # txt_file.write('  name: "%s"\n' % (info['name'] + '_activation'))
  # txt_file.write('  type: "ReLU"\n')
  # txt_file.write('}\n')
  # txt_file.write('\n')

def ChannelwiseConvolution(txt_file, info):
  Convolution(txt_file, info)
  
def BatchNorm(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "BatchNorm"\n')
  txt_file.write('  batch_norm_param {\n')
  txt_file.write('    use_global_stats: true\n')        # TODO
  txt_file.write('    moving_average_fraction: 0.9\n')  # TODO
  txt_file.write('    eps: 0.00002\n')                  # TODO
  txt_file.write('  }\n')
  txt_file.write('}\n')

  # if info['fix_gamma'] is "True":
  #   a = 0# TODO

  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'  % info['top'])
  txt_file.write('  top: "%s"\n'     % info['top'])
  txt_file.write('  name: "%s"\n'    % info['top'])
  txt_file.write('  type: "Scale"\n')
  txt_file.write('  scale_param { bias_term: true }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Scale(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
  txt_file.write('  top: "%s"\n'    % info['top'])
  txt_file.write('  name: "%s"\n'   % info['top'])
  txt_file.write('  type: "Scale"\n')
  txt_file.write('  scale_param { bias_term: true }\n')
  txt_file.write('}\n')
  txt_file.write('\n')


  #TODO: inplace activation
  if info['activation'] != 'ACTIVE_LINEAR':
      txt_file.write('layer {\n')
      txt_file.write('  bottom: "%s"\n'       % info['name'])
      txt_file.write('  top: "%s"\n'          % info['name'])
      txt_file.write('  name: "%s"\n'         % (info['name'] + '_activation'))

      txt_file.write('  type: "ReLU"\n')
      # if info['activation'] == 'ACTIVE_TANH':
      #     txt_file.write('  type: "TanH"\n')
      # elif info['activation'] == 'ACTIVE_RECTIFIED_LINEAR':
      #     txt_file.write('  type: "ReLU"\n')
      txt_file.write('}\n')
      txt_file.write('\n')


  pass

def Resize(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
  txt_file.write('  top: "%s"\n'    % info['top'])
  txt_file.write('  name: "%s"\n'   % info['top'])
  txt_file.write('  type: "Resize"\n')
  txt_file.write('  resize_param { resize: RS_NEAREST \n ')
  txt_file.write('                 upsample_ratio: 2 }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Activation(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['name'])
  txt_file.write('  type: "ReLU"\n')      # TODO
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Concat(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Concat"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('}\n')
  txt_file.write('\n')
  pass
  
def ElementWiseSum(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Eltwise"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Pooling(txt_file, info):
  pool_type = 'AVE' if info['attrs']['pool_type'] == 'avg' else 'MAX'
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Pooling"\n')
  txt_file.write('  pooling_param {\n')
  txt_file.write('    pool: %s\n'         % pool_type)       # TODO
  txt_file.write('    kernel_size: %s\n'  % info['attrs']['kernel'])
  txt_file.write('    stride: %s\n'       % info['attrs']['stride'])
  txt_file.write('    pad: %s\n'          % info['attrs']['pad'])
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass


def FullyConnected(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  txt_file.write('  type: "InnerProduct"\n')
  txt_file.write('  inner_product_param {\n')
  txt_file.write('    num_output: %s\n' % info['attrs']['num_hidden'])
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

  if info['activation'] != 'ACTIVE_LINEAR':
      txt_file.write('layer {\n')
      txt_file.write('  bottom: "%s"\n'       % info['name'])
      txt_file.write('  top: "%s"\n'          % info['name'])
      txt_file.write('  name: "%s"\n'         % (info['name'] + '_activation'))

      if info['activation'] == 'ACTIVE_TANH':
        txt_file.write('  type: "TanH"\n')
      elif info['activation'] == 'ACTIVE_RECTIFIED_LINEAR':
        txt_file.write('  type: "ReLU"\n')

      txt_file.write('}\n')
      txt_file.write('\n')
  pass

def Flatten(txt_file, info):
  pass
  
def SoftmaxOutput(txt_file, info):
  pass


# ----------------------------------------------------------------
def XnetLayerList2InfoList(layer_list):
    info_list = []
    for layer in layer_list:
        info = {}

        if layer['layer_type'] == 'LAYER_CONVOLUTIONAL':
            info['op'] = 'Convolution'
            info['name'] = layer['name']

            if layer['name'] == 'conv_10':
                a = 0

            if layer['input_index'] == 0:
                info['bottom'] = ['data']
            else:
                input = layer['input_index'] - 1 - layer['reduce_index']
                info['bottom'] = [layer_list[input]['name']]

            info['top'] = layer['name']
            info['attrs'] = {}
            info['attrs']['num_filter'] = str(layer['outpt_dim'])
            info['attrs']['kernel'] = str(layer['kernel_size'])
            info['attrs']['stride'] = str(layer['stride'])
            info['attrs']['pad'] = str(layer['padding'])
            info['attrs']['no_bias'] = not layer['bias']
            info['activation'] = layer['activation']
            info_list.append(info)

        if layer['layer_type'] == 'LAYER_NN_RESIZE':
            info['op'] = 'Resize'
            info['name'] = layer['name']

            if layer['input_index'] == 0:
                info['bottom'] = ['data']
            else:
                input = layer['input_index'] - 1
                info['bottom'] = [layer_list[input]['name']]

            info['top'] = layer['name']
            info['attrs'] = {}
            info['attrs']['ratio'] = '2'
            info['resize'] = 'NEAREST'
            info_list.append(info)

        if layer['layer_type'] == 'LAYER_CONCAT':
            info['op'] = 'Concat'
            info['name'] = layer['name']

            info['bottom'] = []

            for input_index in layer['input_index']:
                input = input_index - 1 - layer['reduce_index']
                info['bottom'].append(layer_list[input]['name'])

            info['top'] = layer['name']
            info['attrs'] = {}
            info['attrs']['ratio'] = '2'
            info['resize'] = 'NEAREST'
            info_list.append(info)

        # if layer['layer_type'] == 'LAYER_CONCAT':
        #     info['op'] = 'Concat'
        #     info['name'] = layer['name']
        #
        #     info['bottom'] = []
        #
        #     for input_index in layer['input_index']:
        #         input = input_index - 1
        #         info['bottom'].append(layer_list[input]['name'])
        #
        #     info['top'] = layer['name']
        #     info['attrs'] = {}
        #     info['attrs']['ratio'] = '2'
        #     info['resize'] = 'NEAREST'
        #     info_list.append(info)

    return info_list

        # elif layer['type'] == 'LAYER_CONCAT':

def XnetLayerDict2InfoList(layer_dict):
    info_list = []
    for i in range(1, len(layer_dict)):
        info = {}

        # if layer_dict.has_key(str(i)):
        #     layer = layer_dict[str(i)]

        if str(i) in layer_dict:
            layer = layer_dict[str(i)]


        if layer['layer_type'] == 'LAYER_CONVOLUTIONAL':
            info['op'] = 'Convolution'
            info['name'] = layer['name']

            if layer['input_index'] == 0:
                info['bottom'] = ['data']
            else:
                input = str(layer['input_index'])
                info['bottom'] = [layer_dict[input]['name']]

            info['top'] = layer['name']
            info['attrs'] = {}
            info['attrs']['num_filter'] = str(layer['output_dim'])
            info['attrs']['kernel'] = str(layer['kernel_size'])
            info['attrs']['stride'] = str(layer['stride'])
            info['attrs']['pad'] = str(layer['padding'])
            info['attrs']['no_bias'] = not layer['bias']
            info['activation'] = layer['activation']
            info_list.append(info)

        elif layer['layer_type'] == 'LAYER_SCALE':
            info['op'] = 'Scale'
            info['name'] = layer['name']

            if layer['input_index'] == 0:
                info['bottom'] = ['data']
            else:
                input = str(layer['input_index'])
                info['bottom'] = [layer_dict[input]['name']]

            info['top'] = layer['name']
            info['attrs'] = {}
            info['attrs']['num_hidden'] = layer['output_dim']
            info['attrs']['no_bias'] = not layer['bias']
            info['activation'] = layer['activation']
            info_list.append(info)

        elif layer['layer_type'] == 'LAYER_FULL_CONNECTION':
            info['op'] = 'FullyConnected'
            info['name'] = layer['name']

            if layer['input_index'] == 0:
                info['bottom'] = ['data']
            else:
                input = str(layer['input_index'])
                info['bottom'] = [layer_dict[input]['name']]

            info['top'] = layer['name']
            info['attrs'] = {}
            info['attrs']['num_hidden'] = layer['output_dim']
            info['attrs']['no_bias'] = not layer['bias']
            info['activation'] = layer['activation']
            info_list.append(info)

        elif layer['layer_type'] == 'LAYER_NN_RESIZE':
            info['op'] = 'Resize'
            info['name'] = layer['name']

            if layer['input_index'] == 0:
                info['bottom'] = ['data']
            else:
                input = str(layer['input_index'])
                info['bottom'] = [layer_dict[input]['name']]

            info['top'] = layer['name']
            info['attrs'] = {}
            info['attrs']['ratio'] = '2'
            info['resize'] = 'NEAREST'
            info_list.append(info)

        elif layer['layer_type'] == 'LAYER_CONCAT':
            info['op'] = 'Concat'
            info['name'] = layer['name']

            info['bottom'] = []

            for input_index in layer['input_index']:
                input = str(input_index)
                info['bottom'].append(layer_dict[input]['name'])

            info['top'] = layer['name']
            info_list.append(info)

        elif layer['layer_type'] == 'LAYER_MAX_POOLING':
            info['op'] = 'Pooling'
            info['name'] = layer['name']

            if layer['input_index'] == 0:
                info['bottom'] = ['data']
            else:
                input = str(layer['input_index'])
                info['bottom'] = [layer_dict[input]['name']]

            info['top'] = layer['name']
            info['attrs'] = {}
            info['attrs']['pool_type'] = 'MAX'
            info['attrs']['kernel'] = str(layer['kernel_size'])
            info['attrs']['stride'] = str(layer['stride'])
            info['attrs']['pad'] = str(layer['padding'])

            info_list.append(info)

        elif layer['layer_type'] == 'LAYER_AVE_POOLING':
            info['op'] = 'Pooling'
            info['name'] = layer['name']

            if layer['input_index'] == 0:
                info['bottom'] = ['data']
            else:
                input = str(layer['input_index'])
                info['bottom'] = [layer_dict[input]['name']]

            info['top'] = layer['name']
            info['attrs'] = {}
            info['attrs']['pool_type'] = 'avg'
            info['attrs']['kernel'] = str(layer['kernel_size'])
            info['attrs']['stride'] = str(layer['stride'])
            info['attrs']['pad'] = str(layer['padding'])

            info_list.append(info)

        elif layer['layer_type'] == 'LAYER_ELTWISE_SUM':
            info['op'] = 'elemwise_add'
            info['name'] = layer['name']

            info['bottom'] = []

            for input_index in layer['input_index']:
                input = str(input_index)
                info['bottom'].append(layer_dict[input]['name'])

            info['top'] = layer['name']
            info_list.append(info)

    return info_list

        # elif layer['type'] == 'LAYER_CONCAT':



def write_node(txt_file, info):

    # if 'label' in info['name']:
    #     return
    if info['op'] == 'null' and info['name'] == 'data':
        data(txt_file, info)
    elif info['op'] == 'Convolution':
        Convolution(txt_file, info)
    elif info['op'] == 'Resize':
        Resize(txt_file, info)
    elif info['op'] == 'ChannelwiseConvolution':
        ChannelwiseConvolution(txt_file, info)
    elif info['op'] == 'BatchNorm':
        BatchNorm(txt_file, info)
    elif info['op'] == 'Scale':
        Scale(txt_file, info)
    elif info['op'] == 'Activation':
        Activation(txt_file, info)
    elif info['op'] == 'elemwise_add':
        ElementWiseSum(txt_file, info)
    elif info['op'] == '_Plus':
        ElementWiseSum(txt_file, info)
    elif info['op'] == 'Concat':
        Concat(txt_file, info)
    elif info['op'] == 'Pooling':
        Pooling(txt_file, info)
    elif info['op'] == 'Flatten':
        Flatten(txt_file, info)
    elif info['op'] == 'FullyConnected':
        FullyConnected(txt_file, info)
    elif info['op'] == 'SoftmaxOutput':
        SoftmaxOutput(txt_file, info)
    else:
        print("Warning!  Unknown mxnet op:{}".format(info['op']))
        # sys.exit("Warning!  Unknown mxnet op:{}".format(info['op']))




