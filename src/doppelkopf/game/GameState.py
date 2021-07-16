import numpy as np
from typing import List
from doppelkopf.utils.Console import Console
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.game.Card import Card
from doppelkopf.game.Team import Team
#from doppelkopf.game.Player import Player

class GameState():
    SIZE_PLAYERDECK = Doppelkopf.MAX_CARDS_PER_PLAYER * Card.NUM_CARDTYPES # 12 * 24 = 288 elements
    SIZE_CURRENTSTACK = Doppelkopf.MAX_PLAYERS_IN_GAME * Card.NUM_CARDTYPES # 4 * 24 = 96 elements
    SIZE_CURRENTPLAYER = Doppelkopf.MAX_PLAYERS_IN_GAME # 4 elements (one-hot encoding of player index)
    SIZE_CARDSUPPLY = Card.NUM_CARDTYPES # 24 elements (each element indicating the number of cards left of this type)
    SIZE_TEAM_AFFILIATIONS = Doppelkopf.MAX_PLAYERS_IN_GAME # 4 elements (one for each player, in the order of the cards on the table)
    SIZE_STATE = SIZE_PLAYERDECK + SIZE_CURRENTSTACK + SIZE_CURRENTPLAYER + SIZE_CARDSUPPLY + SIZE_TEAM_AFFILIATIONS # The sum of all components (should be 416)
    RANGE_START = {
        "playerDeck": 0,
        "currentStack": SIZE_PLAYERDECK,
        "currentPlayer": SIZE_PLAYERDECK + SIZE_CURRENTSTACK,
        "cardSupply": SIZE_PLAYERDECK + SIZE_CURRENTSTACK + SIZE_CURRENTPLAYER,
        "teamAffiliations": SIZE_PLAYERDECK + SIZE_CURRENTSTACK + SIZE_CURRENTPLAYER + SIZE_CARDSUPPLY,
    }
    RANGE_END = {
        "playerDeck": SIZE_PLAYERDECK,
        "currentStack": SIZE_PLAYERDECK + SIZE_CURRENTSTACK,
        "currentPlayer": SIZE_PLAYERDECK + SIZE_CURRENTSTACK + SIZE_CURRENTPLAYER,
        "cardSupply": SIZE_PLAYERDECK + SIZE_CURRENTSTACK + SIZE_CURRENTPLAYER + SIZE_CARDSUPPLY,
        "teamAffiliations": SIZE_PLAYERDECK + SIZE_CURRENTSTACK + SIZE_CURRENTPLAYER + SIZE_CARDSUPPLY + SIZE_TEAM_AFFILIATIONS,
    }

    def __init__(self, index: int, playerDeck: List[Card], currentStack: List[Card], cardSupply: np.ndarray, currentPlayerIndex: int, teamAffiliations, isTerminal: bool, flat=None):
        self.index = index # All states are 'numbered' for better tracking
        self.playerDeck = playerDeck.copy() # Keep copies of the original list
        self.currentStack = currentStack.copy() # Keep copies of the original list
        self.cardSupply = cardSupply.copy() # Keep a copy of the card supply at this time during the game
        self.currentPlayerIndex = currentPlayerIndex
        self.teamAffiliations = teamAffiliations
        self.isTerminal = isTerminal
        self.flat = flat if flat is not None else self.flatten()
        if not self.IsValid():
            Console.WriteWarning("This state is not valid: %s" % self)

    @staticmethod
    def FromFlat(flat: np.ndarray, index):
        # # # # # # # # # # # # # # # Slice out the part with the player cards # # # # # # # # # # # # # # #
        start = GameState.RANGE_START["playerDeck"]
        end = GameState.RANGE_END["playerDeck"]
        playerCardsSlice = flat[start:end]
        # # # # # # # # # # # # # # # Slice out the part with the current stack # # # # # # # # # # # # # # #
        start = GameState.RANGE_START["currentStack"]
        end = GameState.RANGE_END["currentStack"]
        currentStackSlice = flat[start:end]
        # # # # # # # # # # # # # # # Slice out the part with the current player index # # # # # # # # # # # # # # #
        start = GameState.RANGE_START["currentPlayer"]
        end = GameState.RANGE_END["currentPlayer"]
        currentPlayerIndex = np.argmax(flat[start:end]) # Use argmax to get the player index
        # # # # # # # # # # # # # # # Slice out the part with the card supply # # # # # # # # # # # # # # #
        start = GameState.RANGE_START["cardSupply"]
        end = GameState.RANGE_END["cardSupply"]
        cardSupplySlice = flat[start:end]
        # # # # # # # # # # # # # # # Slice out the part with the team affiliations # # # # # # # # # # # # # # #
        start = GameState.RANGE_START["teamAffiliations"]
        end = GameState.RANGE_END["teamAffiliations"]
        teamAffiliationsSlice = flat[start:end]
        # # # # # # # # # # # # # # # Determine whether the state is terminal or not # # # # # # # # # # # # # # #
        isTerminal = np.sum(currentStackSlice) >= Doppelkopf.MAX_PLAYERS_IN_GAME # If there are as many cards on the table as there possibly can be, then the state is terminal
        # # # # # # # # # # # # # # # Make player deck from slice # # # # # # # # # # # # # # #
        playerDeck = []
        for i in range(Doppelkopf.MAX_CARDS_PER_PLAYER):
            card = Card.FromFlat(playerCardsSlice[i * Card.NUM_CARDTYPES:(i+1) * Card.NUM_CARDTYPES])
            if card is not None:
                playerDeck.append(card)
        # # # # # # # # # # # # # # # Make current stack from slice # # # # # # # # # # # # # # #
        stack = []
        for i in range(Doppelkopf.MAX_PLAYERS_IN_GAME):
            card = Card.FromFlat(currentStackSlice[i * Card.NUM_CARDTYPES:(i+1) * Card.NUM_CARDTYPES])
            if card is not None:
                stack.append(card)
        # # # # # # # # # # # # # # # Make GameState object from gathered info # # # # # # # # # # # # # # #
        return GameState(index, playerDeck, stack, cardSupplySlice, currentPlayerIndex, teamAffiliationsSlice, isTerminal, flat)

    def flatten(self) -> np.ndarray:
        result = []
        # # # # # # # # # # # # # # # Flatten player deck # # # # # # # # # # # # # # #
        for card in self.playerDeck:
            result.extend(card.Flat())
        for _ in range(Doppelkopf.MAX_CARDS_PER_PLAYER - len(self.playerDeck)):
            result.extend(Card.DUMMY_FLAT)
        # # # # # # # # # # # # # # # Flatten current stack # # # # # # # # # # # # # # #
        for card in self.currentStack:
            result.extend(card.Flat())
        for _ in range(Doppelkopf.MAX_PLAYERS_IN_GAME - len(self.currentStack)):
            result.extend(Card.DUMMY_FLAT)
        # # # # # # # # # # # # # # # Flatten current player # # # # # # # # # # # # # # #
        result = np.array(result, dtype=np.float32)
        flatPlayerIndex = np.zeros(shape=(Doppelkopf.MAX_PLAYERS_IN_GAME), dtype=np.float32)
        flatPlayerIndex[self.currentPlayerIndex] = 1
        result = np.concatenate((result, flatPlayerIndex))
        # # # # # # # # # # # # # # # Flatten cardSupply # # # # # # # # # # # # # # #
        result = np.concatenate((result, self.cardSupply), axis=0)
        # # # # # # # # # # # # # # # Flatten team affiliations # # # # # # # # # # # # # # #
        result = np.concatenate((result, self.teamAffiliations), axis=0)
        return result

    def Flat(self) -> np.ndarray:
        return self.flat

    @staticmethod
    def Random():
        stateIndex = np.random.randint(0, Doppelkopf.MAX_STATES_PER_GAME) # Any number between 0 and 60
        completedTricks = int(stateIndex / Doppelkopf.MAX_STATES_PER_TRICK) # integer division
        stackSize = np.random.randint(0, Doppelkopf.MAX_PLAYERS_IN_GAME) # Any number between 0 and 3
        currentPlayerIndex = np.random.randint(0, Doppelkopf.MAX_PLAYERS_IN_GAME) # Any player between 0 and 3 may start the trick
        cardDeck = Card.createDeck().tolist()
        cardsInHand = Doppelkopf.MAX_CARDS_PER_PLAYER - completedTricks # 12 - the number of completed tricks
        # # # # # # # # # # # # # # # Construct player hand # # # # # # # # # # # # # # #
        playerCards = []
        for _ in range(cardsInHand):
            cardIndex = int(np.random.randint(0, len(cardDeck)))
            card = cardDeck[cardIndex]
            cardDeck.remove(card)
            playerCards.append(card)
        # # # # # # # # # # # # # # # Construct cardSupply # # # # # # # # # # # # # # #
        cardSupply = np.zeros(shape=(Card.NUM_CARDTYPES), dtype=np.int)
        cardSupply.fill(2) # Fill the supply array with 2's to indicate that all cards are available
        cardsAlreadyPlayed = completedTricks * Doppelkopf.MAX_CARDS_PER_TRICK + stackSize # number of tricks * 4 + the size of the current stack
        cardTypes = np.array([card.cardType for card in cardDeck]) # All cards that haven't been assigned to the player yet
        np.random.shuffle(cardTypes)
        cardsPlayedTypes = cardTypes[:cardsAlreadyPlayed] # The first 'cardsAlreadyPlayed' indices of the remaining indices
        for cardPlayedType in cardsPlayedTypes:
            cardSupply[cardPlayedType] -= 1 # Any card, that has been played before, is decreased in supply by one
        # # # # # # # # # # # # # # # Construct current stack # # # # # # # # # # # # # # #
        currentStack = []
        for _ in range(stackSize):
            cardIndex = int(np.random.randint(0, len(cardDeck)))
            card = cardDeck[cardIndex]
            cardDeck.remove(card)
            currentStack.append(card)
            cardSupply[card.cardType] -= 1 # Any card, that has been played in this trick, is also decreased in supply by one
        # # # # # # # # # # # # # # # Determine if state is terminal # # # # # # # # # # # # # # #
        isTerminal = stackSize >= Doppelkopf.MAX_CARDS_PER_TRICK
        # # # # # # # # # # # # # # # Construct team affiliations # # # # # # # # # # # # # # #
        teamAffiliations = np.zeros(shape=(Doppelkopf.MAX_PLAYERS_IN_GAME), dtype=np.float32)
        indices = np.arange(0, Doppelkopf.MAX_PLAYERS_IN_GAME)
        np.random.shuffle(indices)
        teamAffiliations[indices[:2]] = 1 # The first two random indices are marked as team 1, the others remain team 0
        return GameState(stateIndex, playerCards, currentStack, cardSupply, currentPlayerIndex, teamAffiliations, isTerminal)

    @staticmethod
    def RandomList(howMany):
        l = []
        for i in range(howMany):
            l.append(GameState.Random())
        return l

    def IsValid(self) -> bool:
        counter = np.zeros(Card.NUM_CARDTYPES)
        for card in self.playerDeck:
            counter[card.cardType] += 1
            if counter[card.cardType] > 2:
                return False
        for card in self.currentStack:
            counter[card.cardType] += 1
            if counter[card.cardType] > 2:
                return False
        return True

    @staticmethod
    def AreIdentcial(a, b) -> bool:
        if len(a.player_deck) != len(b.player_deck):
            Console.WriteError("a and b have differently sized player decks: %d (a) and %d (b)" % (len(a.player_deck), len(b.player_deck)))
            return False
        elif len(a.currentStack) != len(b.currentStack):
            Console.WriteError("a and b have differently sized stacks: %d (a) and %d (b)" % (len(a.currentStack), len(b.currentStack)))
            return False
        for card_a, card_b in zip(a.player_deck, b.player_deck):
            if not Card.AreIdentcial(card_a, card_b):
                Console.WriteError("a and b have cards in player deck: %d (a) and %d (b)" % (card_a, card_b))
                return False
        for card_a, card_b in zip(a.currentStack, b.currentStack):
            if not Card.AreIdentcial(card_a, card_b):
                return False
        return True

    def __str__(self):
        return str(self.playerDeck) + "," + str(self.currentStack)

    def __repr__(self):
        return self.__str__()