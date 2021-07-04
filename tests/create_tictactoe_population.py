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
    tree_filepath = '../src/games_genprog/games/tictactoe_atomicKernels.xml'
    domainFunctionsTree: ET.ElementTree = ET.parse(tree_filepath)
    interpreter: tictactoe.Interpreter = tictactoe.Interpreter(domainFunctionsTree)

    """
    population = tictactoe.Population()
    population.LoadIndividuals("./outputs/toKeep/champion")
    position = population.StartingPosition()

    positions_list, winner = population.CommitteeGameAgainstARandomPlayer(interpreter, 'tensor2x3x3', random_player_starts=False)
    for p in positions_list:
        tictactoe.Display(p)
    print("winner = {}".format(winner))
    """
    numberOfIndividuals = 50
    levelToFunctionProbabilityDict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0.5, 8: 0.5, 9: 0.5}
    proportionOfConstants = 0.5
    constantCreationParametersList = [-1, 1]
    variableNameToTypeDict = {'position': 'tensor2x3x3'}
    numberOfTournamentParticipants = 2
    mutationProbability = 0.4
    proportionOfNewIndividuals = 0.1
    weightForNumberOfElements = 0.0001
    maximumNumberOfMissedCreationTrials = 1000
    number_of_games = 30
    number_of_generations = 100
    maximum_number_of_opponents = 20

    population: tictactoe.Population = tictactoe.Population(maximum_number_of_opponents)

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
    #population.SaveIndividuals("./outputs/indiv")



    individual_to_cost_dict = population.EvaluateIndividualCosts(
        inputOutputTuplesList=None  ,#: List[Tuple[Dict[str, Any], Any]],  Self-supervised learning: no input nor output
        variableNameToTypeDict=variableNameToTypeDict ,#: Dict[str, str],
        interpreter=interpreter,
        returnType='float',
        weightForNumberOfElements=weightForNumberOfElements,

    )
    print ("individual_to_cost_dict = {}".format(individual_to_cost_dict))

    for generationNdx in range(1, number_of_generations + 1):
        individual_to_cost_dict = population.NewGenerationWithTournament(
                inputOutputTuplesList=None,
                variableNameToTypeDict=variableNameToTypeDict,
                interpreter=interpreter,
                returnType='float',
                numberOfTournamentParticipants=numberOfTournamentParticipants,
                mutationProbability=mutationProbability,
                currentIndividualToCostDict=individual_to_cost_dict,
                proportionOfConstants=proportionOfConstants,
                levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
                functionNameToWeightDict=None,
                constantCreationParametersList=constantCreationParametersList,
                proportionOfNewIndividuals=proportionOfNewIndividuals,
                weightForNumberOfElements=weightForNumberOfElements,
                maximumNumberOfMissedCreationTrials=maximumNumberOfMissedCreationTrials
                )
        (champion, lowest_cost) = population.Champion(individual_to_cost_dict)
        logging.info("Generation {}: lowest cost = {}".format(generationNdx, lowest_cost))
        # Let the champion face a random player
        logging.info("The champion is playing against a random player...")
        number_of_player_wins, number_of_draws, number_of_player_losses = population.PlayMultipleGamesAgainstARandomPlayer(
            champion, interpreter, 'tensor2x3x3', number_of_games=number_of_games)
        logging.info("number_of_player_wins = {}; number_of_draws = {}; number_of_player_losses = {}".format(
            number_of_player_wins, number_of_draws, number_of_player_losses
        ))
        champion_filepath = "./outputs/champion_{}_{:.4f}_{}_{}_{}.xml".format(generationNdx, lowest_cost,
                                                                               str(number_of_player_wins),
                                                                               str(number_of_draws),
                                                                               str(number_of_player_losses))
        champion.Save(champion_filepath)
        #print("\nindividual_to_cost_dict = {}".format(individual_to_cost_dict))


if __name__ == '__main__':
    main()