import genprog.core as gp
import genprog.evolution as gpevo
import math
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import random
import pandas
import numpy as np
import base64
import games_genprog.utilities
import torch
import sys
import copy
import games_genprog.gamesgp as gamesgp

possibleTypes = ['float', 'vector18', 'tensor2x3x3', 'tensor64x2x2', 'vector64', 'tensor64x2x2x2',
                 'tensor8x64', 'vector8', 'tensor64x64x2x2', 'tensor8x2x2x2', 'tensor8x2x2',
                 'tensor8x8x2x2',
                 'vector4', 'vector2', 'vector1', 'tensor1x2x2', 'tensor2x3x3', 'tensor1x1x2x2',
                 'tensor1x2x2x2', 'tensor1x2x3x3']
class Interpreter(gp.Interpreter):
    def FunctionDefinition(self, functionName: str, argumentsList: List[Any]) -> Any:

        if functionName == 'conv2x3x3_64_2x2':
            if argumentsList[0].shape != (2, 3, 3):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (2, 3, 3)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (64, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (64, 2, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            if argumentsList[2].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2].shape = {} != (64,)".format(
                        functionName, argumentsList[2].shape))
            return torch.nn.functional.conv2d(input=torch.from_numpy(argumentsList[0]).unsqueeze(0),
                                              weight=torch.from_numpy(argumentsList[1]),
                                              bias=torch.from_numpy(argumentsList[2])).squeeze().numpy()
        elif functionName == 'conv2x3x3_8_2x2':
            if argumentsList[0].shape != (2, 3, 3):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (2, 3, 3)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (8, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (8, 2, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            if argumentsList[2].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2].shape = {} != (8,)".format(
                        functionName, argumentsList[2].shape))
            return torch.nn.functional.conv2d(input=torch.from_numpy(argumentsList[0]).unsqueeze(0),
                                              weight=torch.from_numpy(argumentsList[1]),
                                              bias=torch.from_numpy(argumentsList[2])).squeeze().numpy()

        elif functionName == 'maxpool64x2x2':
            if argumentsList[0].shape != (64, 2, 2):
                raise ValueError("tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return torch.nn.functional.max_pool2d(torch.from_numpy(argumentsList[0]), kernel_size=2).squeeze().numpy()

        elif functionName == 'maxpool8x2x2':
            if argumentsList[0].shape != (8, 2, 2):
                raise ValueError("tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return torch.nn.functional.max_pool2d(torch.from_numpy(argumentsList[0]), kernel_size=2).squeeze().numpy()

        elif functionName == 'relu64x2x2':
            if argumentsList[0].shape != (64, 2, 2):
                raise ValueError("tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return torch.nn.functional.relu(torch.from_numpy(argumentsList[0])).numpy()

        elif functionName == 'relu8x2x2':
            if argumentsList[0].shape != (8, 2, 2):
                raise ValueError("tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return torch.nn.functional.relu(torch.from_numpy(argumentsList[0])).numpy()

        elif functionName == 'linear64_8':
            if argumentsList[0].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (8, 64):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (8, 64)".format(
                        functionName, argumentsList[1].shape))
            if argumentsList[2].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2].shape = {} != (8,)".format(
                        functionName, argumentsList[2].shape))
            return torch.nn.functional.linear(torch.from_numpy(argumentsList[0]),
                                              torch.from_numpy(argumentsList[1]),
                                              torch.from_numpy(argumentsList[2])).numpy()

        elif functionName == 'linear8_1':
            if argumentsList[0].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (8,)".format(
                        functionName, argumentsList[1].shape))
            if argumentsList[2].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2].shape = {} != (1,)".format(
                        functionName, argumentsList[1].shape))
            return torch.nn.functional.linear(torch.from_numpy(argumentsList[0]),
                                              torch.from_numpy(argumentsList[1]).unsqueeze(0),
                                              torch.from_numpy(argumentsList[2])).item()  # Must return a float
        elif functionName == 'relu64':
            if argumentsList[0].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64,)".format(
                        functionName, argumentsList[0].shape))
            return torch.nn.functional.relu(torch.from_numpy(argumentsList[0])).numpy()
        elif functionName == 'relu8':
            if argumentsList[0].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8,)".format(
                        functionName, argumentsList[0].shape))
            return torch.nn.functional.relu(torch.from_numpy(argumentsList[0])).numpy()
        elif functionName == 'tunnel8x64':
            if argumentsList[0].shape != (8, 64):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8, 64)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0]
        elif functionName == 'tunnel2x3x3':
            if argumentsList[0].shape != (2, 3, 3):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (2, 3, 3)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0]
        elif functionName == 'tunnel64x2x2x2':
            if argumentsList[0].shape != (64, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64, 2, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0]

        elif functionName == 'average64x2x2x2':
            if argumentsList[0].shape != (64, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64, 2, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (64, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (64, 2, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'average64x2x2':
            if argumentsList[0].shape != (64, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (64, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (64, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'average64':
            if argumentsList[0].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (64,)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'average8':
            if argumentsList[0].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (8,)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'max64':
            if argumentsList[0].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (64,)".format(
                        functionName, argumentsList[1].shape))
            return np.maximum(argumentsList[0], argumentsList[1])
        elif functionName == 'min64':
            if argumentsList[0].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (64,)".format(
                        functionName, argumentsList[1].shape))
            return np.minimum(argumentsList[0], argumentsList[1])
        elif functionName == 'max8':
            if argumentsList[0].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (8,)".format(
                        functionName, argumentsList[1].shape))
            return np.maximum(argumentsList[0], argumentsList[1])
        elif functionName == 'min8':
            if argumentsList[0].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (8,)".format(
                        functionName, argumentsList[1].shape))
            return np.minimum(argumentsList[0], argumentsList[1])

        elif functionName == 'conv64x2x2_vector64':
            if argumentsList[0].shape != (64, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (64, 64, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (64, 64, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            if argumentsList[2].shape != (64,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2].shape = {} != (64,)".format(
                        functionName, argumentsList[2].shape))
            return torch.nn.functional.conv2d(input=torch.from_numpy(argumentsList[0]).unsqueeze(0),
                                              weight=torch.from_numpy(argumentsList[1]),
                                              bias=torch.from_numpy(argumentsList[2])).squeeze().numpy()
        elif functionName == 'conv8x2x2_vector8':
            if argumentsList[0].shape != (8, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (8, 8, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (8, 8, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            if argumentsList[2].shape != (8,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2].shape = {} != (8,)".format(
                        functionName, argumentsList[2].shape))
            return torch.nn.functional.conv2d(input=torch.from_numpy(argumentsList[0]).unsqueeze(0),
                                              weight=torch.from_numpy(argumentsList[1]),
                                              bias=torch.from_numpy(argumentsList[2])).squeeze().numpy()
        elif functionName == 'tunnel64x64x2x2':
            if argumentsList[0].shape != (64, 64, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64, 64, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0]
        elif functionName == 'tunnel8x2x2x2':
            if argumentsList[0].shape != (8, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8, 2, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0]
        elif functionName == 'tunnel8x8x2x2':
            if argumentsList[0].shape != (8, 8, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (8, 8, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0]

        elif functionName == 'concat4_8':
            if argumentsList[0].shape != (4,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (4,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (4,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (4,)".format(
                        functionName, argumentsList[1].shape))
            return np.concatenate((argumentsList[0], argumentsList[1]))
        elif functionName == 'concat2_4':
            if argumentsList[0].shape != (2,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (2,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (2,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (2,)".format(
                        functionName, argumentsList[1].shape))
            return np.concatenate((argumentsList[0], argumentsList[1]))
        elif functionName == 'concat1_2':
            if argumentsList[0].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (1,)".format(
                        functionName, argumentsList[1].shape))
            return np.concatenate((argumentsList[0], argumentsList[1]))
        elif functionName == 'relu1':
            if argumentsList[0].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1,)".format(
                        functionName, argumentsList[0].shape))
            return torch.nn.functional.relu(torch.from_numpy(argumentsList[0])).numpy()
        elif functionName == 'conv1x2x2_1_2x2':
            if argumentsList[0].shape != (1, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (1, 1, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (1, 1, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            if argumentsList[2].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2].shape = {} != (1,)".format(
                        functionName, argumentsList[2].shape))
            # .squeeze() will get rid of all dimensions...
            single_value = torch.nn.functional.conv2d(input=torch.from_numpy(argumentsList[0]).unsqueeze(0),
                                              weight=torch.from_numpy(argumentsList[1]),
                                              bias=torch.from_numpy(argumentsList[2])).squeeze().numpy().item()
            return np.array([single_value])
        elif functionName == 'conv2x3x3_1_3x3':
            if argumentsList[0].shape != (2, 3, 3):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (2, 3, 3)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (1, 2, 3, 3):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (1, 2, 3, 3)".format(
                        functionName, argumentsList[1].shape))
            if argumentsList[2].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2].shape = {} != (1,)".format(
                        functionName, argumentsList[2].shape))
            # .squeeze() will get rid of all dimensions...
            single_value = torch.nn.functional.conv2d(input=torch.from_numpy(argumentsList[0]).unsqueeze(0),
                                              weight=torch.from_numpy(argumentsList[1]),
                                              bias=torch.from_numpy(argumentsList[2])).squeeze().numpy().item()
            return np.array([single_value])
        elif functionName == 'relu1x2x2':
            if argumentsList[0].shape != (1, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return torch.nn.functional.relu(torch.from_numpy(argumentsList[0])).numpy()
        elif functionName == 'conv2x3x3_1_2x2':
            if argumentsList[0].shape != (2, 3, 3):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (2, 3, 3)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (1, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (1, 2, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            if argumentsList[2].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2].shape = {} != (1,)".format(
                        functionName, argumentsList[2].shape))
            a = torch.nn.functional.conv2d(input=torch.from_numpy(argumentsList[0]).unsqueeze(0),
                                              weight=torch.from_numpy(argumentsList[1]),
                                              bias=torch.from_numpy(argumentsList[2])).squeeze().numpy()
            return np.expand_dims(a, axis=0)
        elif functionName == 'tunnel1x1x2x2':
            if argumentsList[0].shape != (1, 1, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1, 1, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0]
        elif functionName == 'tunnel1x2x2x2':
            if argumentsList[0].shape != (1, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1, 2, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0]
        elif functionName == 'tunnel1x2x3x3':
            if argumentsList[0].shape != (1, 2, 3, 3):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1, 2, 3, 3)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0]
        elif functionName == 'vector1_to_float':
            if argumentsList[0].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1,)".format(
                        functionName, argumentsList[0].shape))
            return argumentsList[0][0]
        elif functionName == 'average4':
            if argumentsList[0].shape != (4,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (4,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (4,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (4,)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'average2':
            if argumentsList[0].shape != (2,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (2,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (2,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (2,)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'average1':
            if argumentsList[0].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1,)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (1,):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (1,)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'average1x2x2x2':
            if argumentsList[0].shape != (1, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1, 2, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (1, 2, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (1, 2, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'average1x1x2x2':
            if argumentsList[0].shape != (1, 1, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1, 1, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (1, 1, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (1, 1, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'average1x2x2':
            if argumentsList[0].shape != (1, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (1, 2, 2):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (1, 2, 2)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'average1x2x3x3':
            if argumentsList[0].shape != (1, 2, 3, 3):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (1, 2, 3, 3)".format(
                        functionName, argumentsList[0].shape))
            if argumentsList[1].shape != (1, 2, 3, 3):
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1].shape = {} != (1, 2, 3, 3)".format(
                        functionName, argumentsList[1].shape))
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'averagefloat':
            if type(argumentsList[0]) is not float:
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0] = {} is not a float".format(
                        functionName, argumentsList[0]))
            if type(argumentsList[1]) is not float:
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[1] = {} is not a float".format(
                        functionName, argumentsList[1]))
            return (argumentsList[0] + argumentsList[1])/2

        else:
            raise NotImplementedError("tictactoe.Interpreter.FunctionDefinition(): Not implemented function '{}'".format(functionName))

    def TypeConverter(self, type: str, value: str) -> Any:
        if type == 'float':
            return float(value)
        elif type == 'bool':
            if value.upper() == 'TRUE':
                return True
            elif value.upper == 'FALSE':
                return False
            else:
                raise NotImplementedError("tictactoe.Interpreter.TypeConverter(): Type = {}; Not implemented value '{}'".format(type, value))
        else:  # A vector or a tensor
            array1D = games_genprog.utilities.StringTo1DArray(value)
            if type.startswith('vector'):
                return array1D
            elif type == 'tensor2x3x3':
                return np.reshape(array1D, (2, 3, 3))
            elif type == 'tensor64x2x2':
                return np.reshape(array1D, (64, 2, 2))
            elif type == 'tensor64x2x2x2':
                return np.reshape(array1D, (64, 2, 2, 2))
            elif type == 'tensor8x64':
                return np.reshape(array1D, (8, 64))
            elif type == 'tensor64x64x2x2':
                return np.reshape(array1D, (64, 64, 2, 2))
            elif type == 'tensor8x2x2x2':
                return np.reshape(array1D, (8, 2, 2, 2))
            elif type == 'tensor8x2x2':
                return np.reshape(array1D, (8, 2, 2))
            elif type == 'tensor8x8x2x2':
                return np.reshape(array1D, (8, 8, 2, 2))
            elif type == 'tensor1x2x2':
                return np.reshape(array1D, (1, 2, 2))
            elif type == 'tensor1x2x2x2':
                return np.reshape(array1D, (1, 2, 2, 2))
            elif type == 'tensor1x1x2x2':
                return np.reshape(array1D, (1, 1, 2, 2))
            elif type == 'tensor1x2x3x3':
                return np.reshape(array1D, (1, 2, 3, 3))

            else:
                raise NotImplementedError("tictactoe.Interpreter.TypeConverter(): Not implemented type '{}'".format(type))





    def CreateConstant(self, returnType: str, parametersList: Optional[ List[Any] ] ):
        if returnType == 'float':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            return str(random.uniform(parametersList[0], parametersList[1]))
        elif returnType == 'vector18':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (18,))
            return games_genprog.utilities.ArrayToString(random_vector)
        elif returnType == 'tensor2x3x3':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_arr = np.random.uniform(parametersList[0], parametersList[1], (2, 3, 3))
            return games_genprog.utilities.ArrayToString(random_arr)
        elif returnType == 'tensor64x2x2x2':
            if len(parametersList) != 2:
                raise ValueError(
                    "tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(
                        returnType, len(parametersList)))
            random_arr = np.random.uniform(parametersList[0], parametersList[1], (64, 2, 2, 2))
            return games_genprog.utilities.ArrayToString(random_arr)
        elif returnType == 'tensor64x2x2':
            if len(parametersList) != 2:
                raise ValueError(
                    "tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(
                        returnType, len(parametersList)))
            random_arr = np.random.uniform(parametersList[0], parametersList[1], (64, 2, 2))
            return games_genprog.utilities.ArrayToString(random_arr)
        elif returnType == 'tensor64x64x2x2':
            if len(parametersList) != 2:
                raise ValueError(
                    "tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(
                        returnType, len(parametersList)))
            random_arr = np.random.uniform(parametersList[0], parametersList[1], (64, 64, 2, 2))
            return games_genprog.utilities.ArrayToString(random_arr)
        elif returnType == 'tensor8x64':
            if len(parametersList) != 2:
                raise ValueError(
                    "tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(
                        returnType, len(parametersList)))
            random_arr = np.random.uniform(parametersList[0], parametersList[1], (8, 64))
            return games_genprog.utilities.ArrayToString(random_arr)
        elif returnType == 'vector8':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (8,))
            return games_genprog.utilities.ArrayToString(random_vector)
        elif returnType == 'vector64':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (64,))
            return games_genprog.utilities.ArrayToString(random_vector)
        elif returnType == 'tensor8x2x2x2':
            if len(parametersList) != 2:
                raise ValueError(
                    "tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(
                        returnType, len(parametersList)))
            random_arr = np.random.uniform(parametersList[0], parametersList[1], (8, 2, 2, 2))
            return games_genprog.utilities.ArrayToString(random_arr)
        elif returnType == 'tensor8x2x2':
            if len(parametersList) != 2:
                raise ValueError(
                    "tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(
                        returnType, len(parametersList)))
            random_arr = np.random.uniform(parametersList[0], parametersList[1], (8, 2, 2))
            return games_genprog.utilities.ArrayToString(random_arr)
        elif returnType == 'tensor8x8x2x2':
            if len(parametersList) != 2:
                raise ValueError(
                    "tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(
                        returnType, len(parametersList)))
            random_arr = np.random.uniform(parametersList[0], parametersList[1], (8, 8, 2, 2))
            return games_genprog.utilities.ArrayToString(random_arr)
        elif returnType == 'vector1':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (1,))
            return games_genprog.utilities.ArrayToString(random_vector)
        elif returnType == 'vector2':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (2,))
            return games_genprog.utilities.ArrayToString(random_vector)
        elif returnType == 'vector4':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (4,))
            return games_genprog.utilities.ArrayToString(random_vector)
        elif returnType == 'tensor1x2x2':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (1, 2, 2))
            return games_genprog.utilities.ArrayToString(random_vector)
        elif returnType == 'tensor1x1x2x2':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (1, 1, 2, 2))
            return games_genprog.utilities.ArrayToString(random_vector)
        elif returnType == 'tensor1x2x2x2':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (1, 2, 2, 2))
            return games_genprog.utilities.ArrayToString(random_vector)
        elif returnType == 'tensor1x2x3x3':
            if len(parametersList) != 2:
                raise ValueError("tictactoe.Interpreter.CreateConstant(): returnType = {}; len(parametersList) = {} != 2".format(returnType, len(parametersList)))
            random_vector = np.random.uniform(parametersList[0], parametersList[1], (1, 2, 3, 3))
            return games_genprog.utilities.ArrayToString(random_vector)
        else:
            raise NotImplementedError("tictactoe.Interpreter.CreateConstant(): Not implemented return type {}".format(returnType))

    def PossibleTypes(self):
        return possibleTypes

class Population(gamesgp.PlayersPopulation):
    """def __init__(self):
        super().__init__()
    """

    """def EvaluateIndividualCosts(self,
                                inputOutputTuplesList: List[ Tuple[ Dict[str, Any], Any ] ],
                                variableNameToTypeDict: Dict[str, str],
                                interpreter: gp.Interpreter,
                                returnType: str,
                                weightForNumberOfElements: float) -> Dict[gp.Individual, float]:
        individual_to_sum = {individual: 0 for individual in self._individualsList}
        for player1Ndx in range(len(self._individualsList)):
            for player2Ndx in range(player1Ndx, len(self._individualsList)):
                player1 = self._individualsList[player1Ndx]
                player2 = self._individualsList[player2Ndx]
                winner_1_vs_2 = self.WinnerOf(player1, player2, interpreter)
                winner_2_vs_1 = self.WinnerOf(player2, player1, interpreter)
                if winner_1_vs_2 == 'player1':
                    individual_to_sum[player1] += 1
                    individual_to_sum[player2] -= 1
                elif winner_1_vs_2 == 'player2':
                    individual_to_sum[player2] += 1
                    individual_to_sum[player1] -= 1
                if winner_2_vs_1 == 'player1':
                    individual_to_sum[player1] += 1
                    individual_to_sum[player2] -= 1
                elif winner_2_vs_1 == 'player2':
                    individual_to_sum[player2] += 1
                    individual_to_sum[player1] -= 1
        individual_to_average = {individual: individual_to_sum[individual] / (2 * (len(self._individualsList) - 1))
                                 for individual in self._individualsList}
        return individual_to_average
    """

    """def WinnerOf(self, player1, player2, interpreter):
        positions_list, winner = self.Game(player1, player2, interpreter)
        if winner == 'player1':
            return player1
        elif winner == 'player2':
            return player2
        return winner

    def Game(self, player1, player2, interpreter):
        positions_list = []
        position = np.zeros((2, 3, 3), dtype=float)
        winner = None
        current_player = player1
        while winner is None:
            if current_player == player2:
                position = self.SwapPositions(position)
            print ("tictactoe.Population.Game(): position = {}".format(position))
            position, winner = self.ChooseNextPosition(current_player, position, interpreter)

            if current_player == player2:
                position = self.SwapPositions(position)
                if winner is not None and winner != 'draw':
                    if winner == 'player1':  # The current player won with this move. It must be swapped.
                        winner = 'player2'
                    else: # The current player lost with this move. It must me swapped.
                        winner = 'player1'

            if current_player == player1:
                current_player = player2
            else:
                current_player = player1
            positions_list.append(position)
        return positions_list, winner
    """

    """def ChooseNextPosition(self, player, position, interpreter):
        legalPositionsWinner_afterMove = self.LegalPositionsAndWinnerAfterMove(position)
        candidatePositionEvaluation_list = []
        for candidate_position, winner in legalPositionsWinner_afterMove:
            evaluation = None
            if winner == 'player1':
                evaluation = sys.float_info.max
            elif winner == 'player2':  # The other player wins: avoid that if possible
                evaluation = -sys.float_info.max
            else:
                evaluation = interpreter.Evaluate(player, {'position': 'tensor2x3x3'},
                                                  {'position': position}, 'float')
            candidatePositionEvaluation_list.append((candidate_position, evaluation))
        highest_evaluation = -sys.float_info.max
        best_positions = []
        for candidate_position, evaluation in candidatePositionEvaluation_list:
            if evaluation > highest_evaluation:
                highest_evaluation = evaluation
                best_positions = [candidate_position]
            elif evaluation == highest_evaluation:
                best_positions.append(candidate_position)
        if len(best_positions) == 0:
            raise ValueError("tictactoe.Population.ChooseNextPosition(): best_positions is empty...?!")
        position = random.choice(best_positions)
        corresponding_winner = None
        for candidate_position, candidate_winner in legalPositionsWinner_afterMove:
            if np.array_equal(candidate_position, position):
                corresponding_winner = candidate_winner
        return position, corresponding_winner

    def SwapPositions(self, position):
        return position[[1, 0], :, :]
    """

    def PositionShape(self):
        return (2, 3, 3)

    def StartingPosition(self):
        return np.zeros(self.PositionShape(), dtype=float)

    def SwapPosition(self, position):
        return position[[1, 0], :, :]

    def LegalPositionsAndWinnerAfterMove(self, position):
        legalPositionWinner_list = []
        for row in range(3):
            for col in range(3):
                if position[0, row, col] == 0 and position[1, row, col] == 0:
                    candidate_position = copy.deepcopy(position)
                    candidate_position[0, row, col] = 1
                    winner = self.PositionWinner(candidate_position)
                    legalPositionWinner_list.append((candidate_position, winner))
        return legalPositionWinner_list

    def PositionWinner(self, position):
        # Full row
        for row in range(3):
            number_of_Xs = 0
            number_of_Os = 0
            for col in range(3):
                if position[0, row, col] == 1:
                    number_of_Xs += 1
                elif position[1, row, col] == 1:
                    number_of_Os += 1
            if number_of_Xs == 3:
                return 'player1'
            elif number_of_Os == 3:
                return 'player2'
        # Full column
        for col in range(3):
            number_of_Xs = 0
            number_of_Os = 0
            for row in range(3):
                if position[0, row, col] == 1:
                    number_of_Xs += 1
                elif position[1, row, col] == 1:
                    number_of_Os += 1
            if number_of_Xs == 3:
                return 'player1'
            elif number_of_Os == 3:
                return 'player2'
        # Diagonals
        number_of_Xs = 0
        number_of_Os = 0
        for i in range(3):
            if position[0, i, i] == 1:
                number_of_Xs += 1
            elif position[1, i, i] == 1:
                number_of_Os += 1
        if number_of_Xs == 3:
            return 'player1'
        elif number_of_Os == 3:
            return 'player2'

        number_of_Xs = 0
        number_of_Os = 0
        for i in range(3):
            if position[0, 2 - i, i] == 1:
                number_of_Xs += 1
            elif position[1, 2 - i, i] == 1:
                number_of_Os += 1
        if number_of_Xs == 3:
            return 'player1'
        elif number_of_Os == 3:
            return 'player2'

        if np.count_nonzero(position) == 9:
            return 'draw'
        return None

def Display(position):
        if position.shape != (2, 3, 3):
            raise ValueError("tictactoe.Display(): The shape of position ({}) is not (2, 3, 3)".format(position.shape))
        for row in range(3):
            for column in range(3):
                #occupancy = None
                if position[0, row, column] == 1.0:
                    print (' X ', end='', flush=True)
                elif position[1, row, column] == 1.0:
                    print (' O ', end='', flush=True)
                else:
                    print ('   ', end='', flush=True)
                if column != 2:
                    print ('|', end='', flush=True)
                else:
                    print('') # new line
            if row != 2:
                print ('--- --- ---')