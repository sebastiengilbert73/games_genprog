import logging
from genprog import core as gp, evolution as gpevo
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import random
import math
import argparse
import xml.etree.ElementTree as ET
import ast
import sys
from games_genprog.games import tictactoe

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("create_tictactoe_population.py main()")
    tree_filepath = '../src/games_genprog/games/tictactoe.xml'
    domainFunctionsTree: ET.ElementTree = ET.parse(tree_filepath)
    interpreter: tictactoe.Interpreter = tictactoe.Interpreter(domainFunctionsTree)
    population: tictactoe.Population = tictactoe.Population()
    population.Generate(
        numberOfIndividuals=5,
        interpreter=interpreter,
        returnType='float',
        levelToFunctionProbabilityDict={0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        proportionOfConstants=0.5,
        constantCreationParametersList=[-1, 1],
        variableNameToTypeDict={'position': 'tensor2x3x3'},
        functionNameToWeightDict=None
    )
    population.SaveIndividuals("./outputs/indiv")

    individual_to_cost_dict = population.EvaluateIndividualCosts(
        inputOutputTuplesList=None  ,#: List[Tuple[Dict[str, Any], Any]],  Self-supervised learning: no input nor output
        variableNameToTypeDict={'position', 'tensor2x3x3'} ,#: Dict[str, str],
        interpreter=interpreter,
        returnType='float',
        weightForNumberOfElements=0.001
    )
    print ("individual_to_cost_dict = {}".format(individual_to_cost_dict))

if __name__ == '__main__':
    main()