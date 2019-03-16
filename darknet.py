#Import necessary library
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#Parsing the configuration files
def parse_cfg(cfgfile):
    """
    Takes a configuration files

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Bloc
    k is presented as dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                 #store the lines in a list
    lines = [x for x in lines if len(x) >0]         #get rid off the empty lines
    lines = [x for x in lines if x[0] != '#']       #get rid of the comments
    lines = [x.rstrip().lstrip() for x in lines]    #get rid of fringe whatespace

    #Loop over the resultant list to get blocks
    #print(lines)

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":                         #mark the start of a new block
            if len(block) != 0:                    #if block is not empty, implies it for storing values of previous blocks
                blocks.append(block)               #add it to the blocks list
                block = {}                         #re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

"""

Testing of above functional

t = "/home/sudhir/Desktop/YOLO_v3/cfg/yolov3.cfg"
x  = parse_cfg(t)
print(x[0])

"""

#Creating the building blocks
def create_modules(blocks):
    net_info = blocks[0]                                #Captures the information about the input and pre-processing
    module_list = nn.ModuleList()                       #This class is almost like a normal list containing nn.Module objects
    prev_filter = 3                                     #Keep track of number of filter in previous layer initially RGB = 3
    output_filters = []                                 #To keep track of number of filter in each layers
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of the blocks
        #Create the new module for the blocks
        #append to the module_list

        if(x["type"] == "convolutional"):
            #Get info about the layers
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])         #whenever there is no batch_normalization except will execute
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernal_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernal_size-1)//2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filter, filters, kernal_size, stride, pad)
            module.add_module("conv_{0}".format(index), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #Check the activation
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("batch_norm_{0}".format(index), activn)


        #If it is upsampling the layers
        #We use Binear2dUpsampling
        elif(x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        #if it is a route layer
        elif(x["type"] == "route"):
            x["layers"] = x["layers"].split(",")
            #Start of route
            start = int(x["layers"][0])
            #end, if there exists one
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = Emptylayer()
            module.add_module("route_{0}".format(index),route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        #shortcut corresponding to skip connection
        elif x["type"] == "shortcut":
            shortcut = Emptylayer()
            module.add_module("shortcut_{}".format(index), shortcut)

    return 0
