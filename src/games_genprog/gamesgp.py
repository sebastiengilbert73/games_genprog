import genprog.evolution as gpevo
import abc
from typing import Dict, List, Any, Set, Optional, Union, Tuple
import genprog.core as gp
import genprog.evolution as gpevo
import sys
import random
import numpy as np
import logging

class PlayersPopulation(gpevo.Population):

    def __init__(self, maximum_number_of_opponents=None):
        super().__init__()
        self.maximum_number_of_opponents = maximum_number_of_opponents

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
        if self.maximum_number_of_opponents is None:
            number_of_opponents = len(self._individualsList) - 1
        else:
            number_of_opponents = min(len(self._individualsList) - 1, self.maximum_number_of_opponents)
        for playerNdx in range(len(self._individualsList)):
            opponentIndexList = [index for index in range(len(self._individualsList)) if index != playerNdx]
            if len(opponentIndexList) > number_of_opponents:
                opponentIndexList = random.sample(opponentIndexList, number_of_opponents)
            for opponentNdx in opponentIndexList:
                player = self._individualsList[playerNdx]
                opponent = self._individualsList[opponentNdx]
                winner_player_vs_opponent = self.WinnerOf(player, opponent, interpreter, position_type)
                winner_opponent_vs_player = self.WinnerOf(opponent, player, interpreter, position_type)
                if winner_player_vs_opponent == 'player1':
                    individual_to_sum[player] += 1
                elif winner_player_vs_opponent == 'player2':
                    individual_to_sum[player] -= 1
                if winner_opponent_vs_player == 'player1':
                    individual_to_sum[player] -= 1
                elif winner_opponent_vs_player == 'player2':
                    individual_to_sum[player] += 1

        individual_to_average = {individual: individual_to_sum[individual] / (2 * number_of_opponents)
                                 for individual in self._individualsList}
        individual_to_cost = { individual: (1 - individual_to_average[individual])/2
                               for individual in self._individualsList}
        if weightForNumberOfElements != 0:
            for individual in self._individualsList:
                individual_to_cost[individual] += weightForNumberOfElements * len(individual.Elements())

        return individual_to_cost

    def WinnerOf(self, player1, player2, interpreter, position_type):
        positions_list, winner = self.Game(player1, player2, interpreter, position_type)
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

    def ChooseNextPosition(self, player, position, interpreter, position_type, legalPositionsWinner_afterMove=None):
        if legalPositionsWinner_afterMove is None:
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


    def GameAgainstARandomPlayer(self, player, interpreter, position_type, random_player_starts=False):
        positions_list = []
        position = self.StartingPosition()
        winner = None
        if random_player_starts:
            current_player = None
        else:
            current_player = player
        while winner is None:
            if current_player is None:  # Random player
                if not random_player_starts:
                    position = self.SwapPosition(position)
                legalPositionWinner_list = self.LegalPositionsAndWinnerAfterMove(position)
                we_found_a_winning_move = False
                for candidate_position, candidate_winner in legalPositionWinner_list:
                    if candidate_winner == 'player1':
                        position = candidate_position
                        winner = candidate_winner
                        we_found_a_winning_move = True
                if not we_found_a_winning_move:
                    position, winner = random.choice(legalPositionWinner_list)
                if not random_player_starts:
                    position = self.SwapPosition(position)
                    if winner is not None and winner != 'draw':
                        if winner == 'player1':  # The current player won with this move. It must be swapped.
                            winner = 'player2'
                        else:  # The current player lost with this move. It must me swapped.
                            winner = 'player1'
            else:  # The player is an individual
                if random_player_starts:
                    position = self.SwapPosition(position)
                position, winner = self.ChooseNextPosition(current_player, position, interpreter, position_type)

                if random_player_starts:
                    position = self.SwapPosition(position)
                    if winner is not None and winner != 'draw':
                        if winner == 'player1':  # The current player won with this move. It must be swapped.
                            winner = 'player2'
                        else: # The current player lost with this move. It must me swapped.
                            winner = 'player1'

            if current_player == player:
                current_player = None
            else:
                current_player = player
            positions_list.append(position)
        return positions_list, winner

    def PlayMultipleGamesAgainstARandomPlayer(self, player, interpreter, position_type, number_of_games):
        number_of_player_wins = 0
        number_of_player_losses = 0
        number_of_draws = 0
        for gameNdx in range(number_of_games):
            if gameNdx %2 == 0:  # player starts
                positions_list, winner = self.GameAgainstARandomPlayer(player, interpreter, position_type, random_player_starts=False)
                if winner == 'player1':
                    number_of_player_wins += 1
                elif winner == 'player2':
                    number_of_player_losses += 1
                elif winner == 'draw':
                    number_of_draws += 1
                else:
                    raise NotImplementedError("PlayersPopulation.PlayMultipleGamesAgainstARandomPlayer(): winner = {}".format(winner))
            else:  # Random player starts
                positions_list, winner = self.GameAgainstARandomPlayer(player, interpreter, position_type, random_player_starts=True)
                if winner == 'player2':
                    number_of_player_wins += 1
                elif winner == 'player1':
                    number_of_player_losses += 1
                elif winner == 'draw':
                    number_of_draws += 1
                else:
                    raise NotImplementedError("PlayersPopulation.PlayMultipleGamesAgainstARandomPlayer(): winner = {}".format(winner))
        return (number_of_player_wins, number_of_draws, number_of_player_losses)

    def ChooseNextPositionByCommittee(self, position, interpreter, position_type):
        positionWinnerCounts = self.PositionWinnerCounts(position, interpreter, position_type)

        # Find the most popular choice
        highest_count = -1
        most_popular_positionWinner_list = None
        for ((nextPosition, winner), count) in positionWinnerCounts:
            if count > highest_count:
                highest_count = count
                most_popular_positionWinner_list = [(nextPosition, winner)]
            elif count == highest_count:
                most_popular_positionWinner_list.append((nextPosition, winner))
        if most_popular_positionWinner_list is None:
            raise ValueError("PlayersPopulation.ChooseNextPositionByCommittee(): most_popular_positionWinner_list is None... (?)")
        return random.choice(most_popular_positionWinner_list)

    def PositionWinnerCounts(self, position, interpreter, position_type):
        legalPositionsWinner_afterMove = self.LegalPositionsAndWinnerAfterMove(position)
        corresponding_counts = [0] * len(legalPositionsWinner_afterMove)
        for individual in self._individualsList:
            chosenPositionWinner = self.ChooseNextPosition(individual, position, interpreter, position_type,
                                                           legalPositionsWinner_afterMove)
            for positionWinnerNdx in range(len(legalPositionsWinner_afterMove)):
                (p, w) = legalPositionsWinner_afterMove[positionWinnerNdx]
                if np.array_equal(chosenPositionWinner[0], p):
                    corresponding_counts[positionWinnerNdx] += 1
                    break
        positionWinnerCount_list = list(zip(legalPositionsWinner_afterMove, corresponding_counts))
        return positionWinnerCount_list

    def CommitteeGameAgainstARandomPlayer(self, interpreter, position_type, random_player_starts=False):
        positions_list = []
        position = self.StartingPosition()
        winner = None
        if random_player_starts:
            current_player = None
        else:
            current_player = 'committee'
        while winner is None:
            if current_player is None:  # Random player
                if not random_player_starts:
                    position = self.SwapPosition(position)
                legalPositionWinner_list = self.LegalPositionsAndWinnerAfterMove(position)
                we_found_a_winning_move = False
                for candidate_position, candidate_winner in legalPositionWinner_list:
                    if candidate_winner == 'player1':
                        position = candidate_position
                        winner = candidate_winner
                        we_found_a_winning_move = True
                if not we_found_a_winning_move:
                    position, winner = random.choice(legalPositionWinner_list)
                if not random_player_starts:
                    position = self.SwapPosition(position)
                    if winner is not None and winner != 'draw':
                        if winner == 'player1':  # The current player won with this move. It must be swapped.
                            winner = 'player2'
                        else:  # The current player lost with this move. It must me swapped.
                            winner = 'player1'
            else:  # The player is the committee
                if random_player_starts:
                    position = self.SwapPosition(position)
                position, winner = self.ChooseNextPositionByCommittee(position, interpreter, position_type)

                if random_player_starts:
                    position = self.SwapPosition(position)
                    if winner is not None and winner != 'draw':
                        if winner == 'player1':  # The current player won with this move. It must be swapped.
                            winner = 'player2'
                        else: # The current player lost with this move. It must me swapped.
                            winner = 'player1'

            if current_player == 'committee':
                current_player = None
            else:
                current_player = 'committee'
            positions_list.append(position)
        return positions_list, winner

    def PlayMultipleCommitteeGamesAgainstARandomPlayer(self, interpreter, position_type, number_of_games):
        number_of_committee_wins = 0
        number_of_committee_losses = 0
        number_of_draws = 0
        for gameNdx in range(number_of_games):
            if gameNdx %2 == 0:  # committee starts
                positions_list, winner = self.CommitteeGameAgainstARandomPlayer(interpreter, position_type, random_player_starts=False)
                if winner == 'player1':
                    number_of_committee_wins += 1
                elif winner == 'player2':
                    number_of_committee_losses += 1
                elif winner == 'draw':
                    number_of_draws += 1
                else:
                    raise NotImplementedError("PlayersPopulation.PlayMultipleCommitteeGamesAgainstARandomPlayer(): winner = {}".format(winner))
            else:  # Random player starts
                positions_list, winner = self.CommitteeGameAgainstARandomPlayer(interpreter, position_type, random_player_starts=True)
                if winner == 'player2':
                    number_of_committee_wins += 1
                elif winner == 'player1':
                    number_of_committee_losses += 1
                elif winner == 'draw':
                    number_of_draws += 1
                else:
                    raise NotImplementedError("PlayersPopulation.PlayMultipleCommitteeGamesAgainstARandomPlayer(): winner = {}".format(winner))
        return (number_of_committee_wins, number_of_draws, number_of_committee_losses)