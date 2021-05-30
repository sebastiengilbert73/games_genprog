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

possibleTypes = ['float', 'vector18', 'tensor2x3x3', 'tensor64x2x2', 'vector64', 'tensor64x2x2x2',
                 'tensor8x64', 'vector8']
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

        elif functionName == 'maxpool64x2x2':
            if argumentsList[0].shape != (64, 2, 2):
                raise ValueError("tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64, 2, 2)".format(
                        functionName, argumentsList[0].shape))
            return torch.nn.functional.max_pool2d(torch.from_numpy(argumentsList[0]), kernel_size=2).squeeze().numpy()

        elif functionName == 'relu64x2x2':
            if argumentsList[0].shape != (64, 2, 2):
                raise ValueError("tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[0].shape = {} != (64, 2, 2)".format(
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
            if type(argumentsList[2]) is not float:
                raise ValueError(
                    "tictactoe.Interpreter.FunctionDefinition(): functionName = {}; argumentsList[2] = {} is not float".format(
                        functionName, argumentsList[2]))
            return torch.nn.functional.linear(torch.from_numpy(argumentsList[0]),
                                              torch.from_numpy(argumentsList[1]).unsqueeze(0),
                                              torch.tensor([argumentsList[2]])).item()  # Must return a float

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
                raise NotImplementedError("tictactoe_interpreter.TypeConverter(): Type = {}; Not implemented value '{}'".format(type, value))
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
        else:
            raise NotImplementedError("tictactoe.Interpreter.CreateConstant(): Not implemented return type {}".format(returnType))

    def PossibleTypes(self):
        return possibleTypes

class Population(gpevo.Population):
    def __init__(self):
        super().__init__()

    def EvaluateIndividualCosts(self,
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
                winner_1_vs_2 = self.WinnerOf(player1, player2)
                winner_2_vs_1 = self.WinnerOf(player2, player1)
                if winner_1_vs_2 == player1:
                    individual_to_sum[player1] += 1
                    individual_to_sum[player2] -= 1
                elif winner_1_vs_2 == player2:
                    individual_to_sum[player2] += 1
                    individual_to_sum[player1] -= 1
                if winner_2_vs_1 == player1:
                    individual_to_sum[player1] += 1
                    individual_to_sum[player2] -= 1
                elif winner_2_vs_1 == player2:
                    individual_to_sum[player2] += 1
                    individual_to_sum[player1] -= 1
        individual_to_average = {individual: individual_to_sum[individual] / (2 * (len(self._individualsList) - 1))
                                 for individual in self._individualsList}
        return individual_to_average

    def WinnerOf(self, player1, player2):
        return random.choice([player1, player2])