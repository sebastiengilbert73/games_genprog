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
    numberOfIndividuals = 50
    levelToFunctionProbabilityDict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    proportionOfConstants = 0.5
    constantCreationParametersList = [-1, 1]
    variableNameToTypeDict = {'position': 'tensor2x3x3'}
    population.Generate(
        numberOfIndividuals=numberOfIndividuals,
        interpreter=interpreter,
        returnType='float',
        levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
        proportionOfConstants=proportionOfConstants,
        constantCreationParametersList=constantCreationParametersList,
        variableNameToTypeDict=variableNameToTypeDict,
        functionNameToWeightDict=None
    )
    population.SaveIndividuals("./outputs/indiv")



    individual_to_cost_dict = population.EvaluateIndividualCosts(
        inputOutputTuplesList=None  ,#: List[Tuple[Dict[str, Any], Any]],  Self-supervised learning: no input nor output
        variableNameToTypeDict=variableNameToTypeDict ,#: Dict[str, str],
        interpreter=interpreter,
        returnType='float',
        weightForNumberOfElements=0.001
    )
    print ("individual_to_cost_dict = {}".format(individual_to_cost_dict))

    for generationNdx in range(1, 10):
        individual_to_cost_dict = population.NewGenerationWithTournament(
                inputOutputTuplesList=None,
                variableNameToTypeDict=variableNameToTypeDict,
                interpreter=interpreter,
                returnType='float',
                numberOfTournamentParticipants=4,
                mutationProbability=0.1,
                currentIndividualToCostDict=individual_to_cost_dict,
                proportionOfConstants=proportionOfConstants,
                levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
                functionNameToWeightDict=None,
                constantCreationParametersList=constantCreationParametersList,
                proportionOfNewIndividuals=0.05,
                weightForNumberOfElements=0.001,
                maximumNumberOfMissedCreationTrials=1000
                )
        (champion, lowest_cost) = population.Champion(individual_to_cost_dict)
        logging.info("Generation {}: lowest cost = {}".format(generationNdx, lowest_cost))
        champion_filepath = "./outputs/champion_{}_{:.4f}.xml".format(generationNdx, lowest_cost)
        champion.Save(champion_filepath)
        #print("\nindividual_to_cost_dict = {}".format(individual_to_cost_dict))


if __name__ == '__main__':
    main()