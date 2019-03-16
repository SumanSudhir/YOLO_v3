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
