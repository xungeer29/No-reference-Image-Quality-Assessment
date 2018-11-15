import os
import math

def read_norm_list_correct_distribution(txt_path, num, upbound=0.7, lowbound=0.2):
    f = open(txt_path, 'r')
    lines = f.readlines()

    if num > 0:
        lines = lines[0:num]

    samples = []
    norm_list = []
    for line in lines:
        line = line.strip().split(' ')
        path = line[0]
        confidence = float(line[1])

        norm_list.append(confidence)
        samples.append([path, confidence])

    max_norm = max(norm_list)
    min_norm = min(norm_list)

    for i in range(len(samples)):
        samples[i][1] = (samples[i][1] - min_norm) / (max_norm - min_norm)

        if samples[i][1] > 1.0:
            samples[i][1] = 1.0
        if samples[i][1] < 0.0:
            samples[i][1] = 0.0

        if samples[i][1] > upbound:
            samples[i][1] = upbound
        if samples[i][1] < lowbound:
            samples[i][1] = lowbound

        samples[i][1] = (samples[i][1] - lowbound) / (upbound - lowbound)
    return samples, len(samples)

def read_norm_list_correct_distribution5label(txt_path, num, upbound=0.7, lowbound=0.2):
    f = open(txt_path, 'r')
    lines = f.readlines()

    if num > 0:
        lines = lines[0:num]

    samples = []
    norm_list = []
    for line in lines:
        line = line.strip().split(' ')
        path = line[0]
        confidence = float(line[1])

        norm_list.append(confidence)
        samples.append([path, confidence])

    max_norm = max(norm_list)
    min_norm = min(norm_list)

    for i in range(len(samples)):
        samples[i][1] = (samples[i][1] - min_norm) / (max_norm - min_norm)

        if samples[i][1] > 1.0:
            samples[i][1] = 1.0
        if samples[i][1] < 0.0:
            samples[i][1] = 0.0

        if samples[i][1] > upbound:
            samples[i][1] = upbound
        if samples[i][1] < lowbound:
            samples[i][1] = lowbound

        samples[i][1] = (samples[i][1] - lowbound) / (upbound - lowbound)

        if samples[i][1] > 1.0:
            samples[i][1] = 1.0
        if samples[i][1] < 0.0:
            samples[i][1] = 0.0

        cur_tmp = samples[i][1]
        label = 0.0

        if cur_tmp<0.2:
            label = 1.0
        elif cur_tmp<0.4:
            label = 2.0
        elif cur_tmp<0.6:
            label = 3.0
        elif cur_tmp<0.8:
            label = 4.0
        else:
            label =5.0

        samples[i][1] = label
        # print(label)
    return samples, len(samples)

def read_angle_list(txt_path, num):
    f = open(txt_path, 'r')
    lines = f.readlines()

    if num > 0:
        lines = lines[0:num]

    samples = []
    for line in lines:
        line = line.strip().split(' ')
        path = line[0]
        confidence = int(line[1])

        samples.append([path, confidence])


    return samples, len(samples)

def read_IQIYI_list(txt_path, num):
    f = open(txt_path, 'r')
    lines = f.readlines()

    if num > 0:
        lines = lines[0:num]

    samples = []
    for line in lines:
        line = line.strip()
        path = line

        name = os.path.split(path)[1]
        confidence = float(name.replace('.jpg','')) / 200.0
        samples.append([path, confidence])
        
    return samples, len(samples)


def read_hardangle_list():
    txt_path = r'/data2/shentao/DATA/FACE/FaceQualityIQIYI/hard_angle.txt'

    f = open(txt_path, 'r')
    lines = f.readlines()

    samples = []
    for line in lines:
        line = line.strip()
        path = line
        samples.append([path, 0.0])

    return samples, len(samples)
