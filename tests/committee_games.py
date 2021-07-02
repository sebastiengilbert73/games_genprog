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

parser = argparse.ArgumentParser()
parser.add_argument('--populationFilepathPrefix', help="The filepath prefix of the population. Default: './outputs/toKeep/champion'", default='./outputs/toKeep/champion')
parser.add_argument('--functionsDefinitionFilepath', help="The filepath to the tictactoe functions defintion. Default: './games_genprog/games/tictactoe.xml'", default='./games_genprog/games/tictactoe.xml')
parser.add_argument('--numberOfGames', help="The number of games played. Default: 30", type=int, default=30)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')


def main():
    logging.info("committee_games.py main()")

    domainFunctionsTree: ET.ElementTree = ET.parse(args.functionsDefinitionFilepath)
    interpreter: tictactoe.Interpreter = tictactoe.Interpreter(domainFunctionsTree)
    population: tictactoe.Population = tictactoe.Population()
    population.LoadIndividuals(args.populationFilepathPrefix)

    positions_list, winner = population.CommitteeGameAgainstARandomPlayer(interpreter, position_type='tensor2x3x3', random_player_starts=True)
    for position in positions_list:
        tictactoe.Display(position)
        print()
    print ("winner = {}".format(winner))

    """position = population.StartingPosition()
    position[0, 1, 1] = 1
    position[1, 0, 0] = 1
    position[1, 2, 1] = 1
    positionWinnerCount_list = population.PositionWinnerCounts(position, interpreter, position_type='tensor2x3x3')
    print (positionWinnerCount_list)
    """
    """
    number_of_committee_wins, number_of_draws, number_of_committee_losses = population.PlayMultipleCommitteeGamesAgainstARandomPlayer(interpreter, position_type='tensor2x3x3', number_of_games=args.numberOfGames)
    logging.info ("Results: {} - {} - {}".format(number_of_committee_wins, number_of_draws, number_of_committee_losses))
    """

if __name__ == '__main__':
    main()