import sys
caffe_root = ''
sys.path.insert(0,caffe_root+'python')
import caffe
from caffe.proto.caffe_pb2 import NetParameter
import Queue
import google.protobuf as pb
import copy
import numpy as np

class Node():
    def __init__(self, node_id, layer_id, fl_layer_in = 32, fl_layer_out = 32, fl_params = 32):
        self.node_id = node_id
        self.layer_id = layer_id
        self.fl_layer_in = fl_layer_in
        self.fl_layer_out = fl_layer_out
        self.fl_params = fl_params

class Graph():
    def __init__(self, net_file):
        self.node_id = 0
        self.nodes_list = []
        self.bottom_nodes_dict = {}
        self.top_nodes_dict = {}
        self.layer_id_to_node_dict = {}
        self.in_node_num_dict_up = {}
        self.in_node_num_dict_down = {}
        self.channel_dict = {}
        with open(net_file, 'r') as fp:
            self.net = NetParameter()
            pb.text_format.Parse(fp.read(), self.net)

    def add_edge(self, from_node, to_node):
        if self.bottom_nodes_dict.has_key(to_node):
            self.bottom_nodes_dict[to_node].append(from_node)
        else:
            self.bottom_nodes_dict[to_node] = [from_node]
        if self.top_nodes_dict.has_key(from_node):
            self.top_nodes_dict[from_node].append(to_node)
        else:
            self.top_nodes_dict[from_node] = [to_node]
        self.in_node_num_dict_up[from_node] += 1
        self.in_node_num_dict_down[to_node] += 1

    def create_quantized_graph(self):
        layer_num = len(self.net.layer)
        for x in range(layer_num):
            layer = self.net.layer[x]
            if layer.type == 'ReLU':
                continue
            elif layer.type == 'Input':
                node = Node(self.node_id, x)
                self.channel_dict[node] = layer.input_param.shape[0].dim._values[1]
                self.in_node_num_dict_up[node] = self.in_node_num_dict_down[node] = 0
                self.layer_id_to_node_dict[x] = node
                self.nodes_list.append(node)
                self.node_id += 1
            elif layer.type in ['Concat', 'Eltwise']:
                node = Node(self.node_id, x)
                self.in_node_num_dict_up[node] = self.in_node_num_dict_down[node] = 0
                for bottom_name in self.net.layer[x].bottom:
                    for y in range(layer_num):
                        if self.net.layer[y].top[0] == bottom_name:
                            self.add_edge(self.layer_id_to_node_dict[y], node)
                            break
                self.layer_id_to_node_dict[x] = node
                self.nodes_list.append(node)
                self.node_id += 1
            elif layer.type in ['Pooling', 'ConvolutionRistretto', 'FcRistretto', 'DeconvolutionRistretto', 'ScaleRistretto']:
                if layer.type != 'Pooling':
                    fl_layer_in = layer.quantization_param.fl_layer_in
                    fl_layer_out = layer.quantization_param.fl_layer_out
                    fl_params = layer.quantization_param.fl_params
                    node = Node(self.node_id, x, fl_layer_in, fl_layer_out, fl_params)
                else:
                    node = Node(self.node_id, x)
                self.in_node_num_dict_up[node] = self.in_node_num_dict_down[node] = 0
                for y in range(layer_num):
                    if self.net.layer[y].top[0] == self.net.layer[x].bottom[0]:
                        self.add_edge(self.layer_id_to_node_dict[y], node)
                        break
                self.layer_id_to_node_dict[x] = node
                self.nodes_list.append(node)
                self.node_id += 1
            else:
                print('layer_type %s not supported!!!' %layer.type)
                pass


    def debug_graph(self):
        layers = self.net.layer
        for node in self.nodes_list:
            print (node.node_id, layers[node.layer_id].name, node.fl_layer_in, node.fl_layer_out, node.fl_params)
            line = 'bottom: '
            if self.bottom_nodes_dict.has_key(node):
                for bottom_node in self.bottom_nodes_dict[node]:
                    line += layers[bottom_node.layer_id].name + ' '
            print (line)
            line = 'top: '
            if self.top_nodes_dict.has_key(node):
                for top_node in self.top_nodes_dict[node]:
                    line += layers[top_node.layer_id].name + ' '
            print (line)
            print ('')

    def calculate_params(self, node, channel, layer):
        if layer.type == 'ConvolutionRistretto' or layer.type == 'DeconvolutionRistretto':
            if layer.convolution_param.HasField('group'):
                node.fl_params = min(node.fl_params, 27 - node.fl_layer_in)
            else:
                kernel_size = layer.convolution_param.kernel_size
                macs = kernel_size * kernel_size * channel
                if macs > 64:
                    node.fl_params = min(node.fl_params, 25 - node.fl_layer_in)
                else:
                    for log in range(31):
                        if (1 << log) >= macs:
                            break
                    node.fl_params = min(node.fl_params, 31 - log)
        elif layer.type == 'FcRistretto':
            node.fl_params = min(node.fl_params, 25 - node.fl_layer_in)
        elif layer.type == 'Pooling':
            if layer.pooling_param.pool == 0:
                node.fl_params = 15
        elif layer.type == 'ScaleRistretto':
            pass
        else:
            print('layer_type %s not supported!!!' % layer.type)
            pass

    # adjust fl_layer_in and fl_layer_out using topological_sorting
    def topological_sorting_down(self, flag):
        layers = self.net.layer
        queue = Queue.Queue()
        for node, in_cnt in self.in_node_num_dict_down:
            if in_cnt == 0:
                queue.put(node)
        in_node_num_dict = copy.deepcopy(self.in_node_num_dict_down)
        while not queue.empty():
            u_node = queue.get()
            for v_node in self.top_nodes_dict[u_node]:
                if not flag:
                    v_node.fl_layer_in = min(v_node.fl_layer_in, u_node.fl_layer_out)
                    if v_node.fl_layer_in == 32:
                        v_node.fl_layer_out = v_node.fl_layer_in
                else:
                    if v_node.fl_params != 32:
                        self.calculate_params(v_node, self.channel_dict[u_node], layers[v_node.layer_id])
                in_node_num_dict[v_node] -= 1
                if in_node_num_dict[v_node] == 0:
                    queue.put(v_node)


    def topological_sorting_up(self):
        layers = self.net.layer
        queue = Queue.Queue()
        for node, in_cnt in self.in_node_num_dict_up:
            if in_cnt == 0:
                queue.put(node)
        while not queue.empty():
            u_node = queue.get()
            for v_node in self.bottom_nodes_dict[u_node]:
                v_node.fl_layer_out = min(v_node.fl_layer_out, u_node.fl_layer_in)
                self.in_node_num_dict_up[v_node] -= 1
                if self.in_node_num_dict_up[v_node] == 0:
                    queue.put(v_node)


if __name__ == '__main__':
    net_file = ''
    zy = Graph(net_file)
    zy.create_quantized_graph()
    # zy.debug_graph(net_file)


        


