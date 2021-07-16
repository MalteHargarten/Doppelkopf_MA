import numpy as np
from typing import List
from doppelkopf.game.Card import Card
from doppelkopf.game.Player import Player
from doppelkopf.utils.Console import Console
from doppelkopf.game.CardFeedback import CardFeedback

class Doppelkopf():
    TOTAL_POINTS = 240
    HALF_POINTS = 120
    MAX_PLAYERS_IN_GAME = 4
    MAX_CARDS_PER_TRICK = 4
    MAX_STATES_PER_TRICK = 5 # Every trick consists of up to 5 states (starting with the empty state)
    MAX_POINTS_PER_TRICK = MAX_PLAYERS_IN_GAME * Card.HIGHEST_CARD_VALUE
    MAX_CARDS_PER_PLAYER = 12
    MAX_STATES_PER_GAME = MAX_CARDS_PER_PLAYER * (MAX_PLAYERS_IN_GAME + 1) # +1 because of the "empty" set (no cards played yet) within each trick

    @staticmethod
    def ResetBeforeGame(players: List[Player], teamRe, teamKontra):
        teamRe.Reset()
        teamKontra.Reset()
        for player in players:
            player.Reset()

    @staticmethod 
    def dealCards(players, teamRe, teamKontra, cardDeck: np.ndarray):
        num_of_cards = int(len(cardDeck) / len(players))
        valid_card_distribution = False
        while not valid_card_distribution: # Reshuffle and re-deal until the dealt cards are valid
            Doppelkopf.ResetBeforeGame(players, teamRe, teamKontra) # reset() all player- and team-objects
            valid_card_distribution = True # Assume distribution to be valid (until someone is dealt two Queens of Clubs)
            np.random.shuffle(cardDeck) # Shuffle for good measure
            indices = np.arange(len(cardDeck))
            np.random.shuffle(indices)
            for i in range(len(players)):
                start = num_of_cards * i
                end = num_of_cards*(i+1)
                #Console.WriteWarning("start: %d end: %d" % (start, end), "Doppelkopf.dealCards()")
                players_cards = cardDeck[indices[start:end]].tolist()
                num_of_queens_of_clubs = players[i].handCards(players_cards, teamRe, teamKontra) # Hand the cards to the player and get the number of Queens of CLubs (0-2)
                if num_of_queens_of_clubs == 2:
                    valid_card_distribution = False # Declare this distribution invalid (and re-deal)
                    break # Break out of the inner loop (and cause the outer loop to run again)
        return num_of_cards

    @staticmethod
    def findBestCard(trickStack):
        leadingCard = trickStack[0]
        leadingSuit = leadingCard.suit # The first card played determines this trick's suit
        bestCard = leadingCard # The first card's rank is automatically the best, unless one of the others is higher ranking or trumping
        bestCardIndex = 0 # The winner is, by default, the first player, unless someone else played a higher ranking (or trumping) card
        for i in range(1, len(trickStack)): # Start at 1, since the first card has already been examined
            card = trickStack[i]
            if Doppelkopf.a_or_b(bestCard, card, leadingSuit) == card: #  if 'card' is better than 'best_card'
                bestCard = card
                bestCardIndex = i
        return bestCardIndex

    @staticmethod
    def a_or_b(a: Card, b: Card, leadingSuit):
        if a.isTrump and b.isTrump: # If both are trumps
            return b if (b.rank < a.rank) else a
        elif not a.isTrump and not b.isTrump: # If both are plain cards
            if a.suit == leadingSuit and b.suit == leadingSuit: # If both are leading suits
                return b if (b.rank < a.rank) else a
            elif a.suit == leadingSuit:
                return a
            elif b.suits == leadingSuit:
                return b
        elif a.isTrump and not b.isTrump:
            return a
        elif not a.isTrump and b.isTrump:
            return b
        Console.WriteError("Technically, neither card (%s or %s) won! Returning 'a' by default!" % (a, b), "Doppelkopf.a_or_b()")
        return a # Be default, a wins (this only happens if none of the above criteria applies here, which should never happen in the first place)

    @staticmethod
    def GetTrickValue(trickStack):
        score = 0
        card: Card
        for card in trickStack:
            score += card.value
        return score

    @staticmethod
    def GetWinner(teamRe, teamKontra):
        # Calculate score for each team
        teamRe.CountTeamScore()
        teamKontra.CountTeamScore()
        if teamRe.Score() + teamKontra.Score() != Doppelkopf.TOTAL_POINTS:
            Console.WriteError("Illegal state, points do not sum up (%d/%d)!" % (teamRe.Score() + teamKontra.Score(), Doppelkopf.TOTAL_POINTS), "Doppelkopf.GetWinner()")        
        if teamRe.Score() > Doppelkopf.HALF_POINTS: # If team 'Re' has more than half of the total score
            return teamRe
        elif teamKontra.Score() >= Doppelkopf.HALF_POINTS: # If team 'Kontra' has half the points or more
            return teamKontra
        else:
            Console.WriteError("Illegal state, neither team has enough points to win!", "Doppelkopf.GetWinner()")
            return None

    @staticmethod
    def getLegalPlayableMask(leadingCard: Card, player: Player) -> np.ndarray:
        mask = np.zeros(len(Card.CARDTYPES), dtype=np.float32)
        for card in Card.CARDTYPES:
            feedback = player.getFeedback(card.cardType, leadingCard)
            if feedback == CardFeedback.OK or feedback == CardFeedback.OK_COULD_NOT_FOLLOW_SUIT:
                mask[card.cardType] = 1
        return mask