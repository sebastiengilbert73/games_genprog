import genprog.evolution as gpevo
import abc
from typing import Dict, List, Any, Set, Optional, Union, Tuple
import genprog.core as gp
import genprog.evolution as gpevo
import sys
import random
import numpy as np

class PlayersPopulation(gpevo.Population):

    @abc.abstractmethod
    def PositionShape(self):
        pass

    @abc.abstractmethod
    def StartingPosition(self):
        pass

    @abc.abstractmethod
    def SwapPosition(self, position):
        pass

    @abc.abstractmethod
    def LegalPositionsAndWinnerAfterMove(self, position):
        pass

    def EvaluateIndividualCosts(self,
                                inputOutputTuplesList: List[ Tuple[ Dict[str, Any], Any ] ],  # None
                                variableNameToTypeDict: Dict[str, str],
                                interpreter: gp.Interpreter,
                                returnType: str,  # 'float'
                                weightForNumberOfElements: float) -> Dict[gp.Individual, float]:
        individual_to_sum = {individual: 0 for individual in self._individualsList}
        if 'position' not in variableNameToTypeDict:
            raise ValueError("gamesgp.PlayersPopulation.EvaluateIndividualCosts(): 'position' is not a key in variableNameToTypeDict")
        position_type = variableNameToTypeDict['position']
        for player1Ndx in range(len(self._individualsList)):
            for player2Ndx in range(player1Ndx, len(self._individualsList)):
                player1 = self._individualsList[player1Ndx]
                player2 = self._individualsList[player2Ndx]
                winner_1_vs_2 = self.WinnerOf(player1, player2, interpreter, position_type)
                winner_2_vs_1 = self.WinnerOf(player2, player1, interpreter, position_type)
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

    def WinnerOf(self, player1, player2, interpreter, position_type):
        positions_list, winner = self.Game(player1, player2, interpreter, position_type)
        if winner == 'player1':
            return player1
        elif winner == 'player2':
            return player2
        return winner

    def Game(self, player1, player2, interpreter, position_type):
        positions_list = []
        position = self.StartingPosition()
        winner = None
        current_player = player1
        while winner is None:
            if current_player == player2:
                position = self.SwapPosition(position)
            #print ("gamesgp.PlayersPopulation..Game(): position = {}".format(position))
            position, winner = self.ChooseNextPosition(current_player, position, interpreter, position_type)

            if current_player == player2:
                position = self.SwapPosition(position)
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

    def ChooseNextPosition(self, player, position, interpreter, position_type):
        legalPositionsWinner_afterMove = self.LegalPositionsAndWinnerAfterMove(position)
        candidatePositionEvaluation_list = []
        for candidate_position, winner in legalPositionsWinner_afterMove:
            evaluation = None
            if winner == 'player1':
                evaluation = sys.float_info.max
            elif winner == 'player2':  # The other player wins: avoid that if possible
                evaluation = -sys.float_info.max
            else:
                evaluation = interpreter.Evaluate(player, {'position': position_type},
                                                  {'position': candidate_position}, 'float')
            candidatePositionEvaluation_list.append((candidate_position, evaluation))
            print ("gamesgp.ChooseNextPosition(): evaluation = {}".format(evaluation))

        highest_evaluation = -sys.float_info.max
        best_positions = []
        for candidate_position, evaluation in candidatePositionEvaluation_list:
            if evaluation > highest_evaluation:
                highest_evaluation = evaluation
                best_positions = [candidate_position]
            elif evaluation == highest_evaluation:
                best_positions.append(candidate_position)
        if len(best_positions) == 0:
            raise ValueError("gamesgp.PlayersPopulation.ChooseNextPosition(): best_positions is empty...?!")
        chosen_position = random.choice(best_positions)
        corresponding_winner = None
        for candidate_position, candidate_winner in legalPositionsWinner_afterMove:
            if np.array_equal(candidate_position, chosen_position):
                corresponding_winner = candidate_winner
        return chosen_position, corresponding_winner



