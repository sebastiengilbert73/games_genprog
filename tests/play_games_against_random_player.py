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
    logging.info("play_games_against_random_player.py main()")
    tree_filepath = '../src/games_genprog/games/tictactoe.xml'
    domainFunctionsTree: ET.ElementTree = ET.parse(tree_filepath)
    interpreter: tictactoe.Interpreter = tictactoe.Interpreter(domainFunctionsTree)
    population: tictactoe.Population = tictactoe.Population()

    player = gp.LoadIndividual('./outputs/champion_9_0.2398.xml')
    number_of_player_wins, number_of_draws, number_of_player_losses = population.PlayMultipleGamesAgainstARandomPlayer(
        player, interpreter, 'tensor2x3x3', number_of_games=30)
    logging.info("number_of_player_wins = {}; number_of_draws = {}; number_of_player_losses = {}".format(
        number_of_player_wins, number_of_draws, number_of_player_losses
    ))


if __name__ == '__main__':
    main()