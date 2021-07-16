import numpy as np
from typing import List
from doppelkopf.game.Card import Card
from doppelkopf.utils.Console import Console
from doppelkopf.game.CardFeedback import CardFeedback

class Player():
    NAMES = ["Steve", "Mary", "Todd", "Jane"]

    def __init__(self, index: int):
        self.index = index
        self.name = Player.NAMES[index]
        self.nextPlayer = None
        self.previousPlayer = None
        self.cards: List[Card] = []
        self.team = None
        self.perceivedTeams = {}

    @staticmethod
    def createPlayers():
        players = []
        for i in range(len(Player.NAMES)):
            players.append(Player(i))
        for i in range(len(players)):
            if i > 0:
                Player.AssignNeighbours(players[i - 1], players[i])
            else:
                Player.AssignNeighbours(players[-1], players[i]) # The last's players 'next' is the first player (i == 0)
        return players

    def Reset(self):
        self.cards.clear()
        self.team = None
        for player in self.perceivedTeams.keys():
            self.perceivedTeams[player] = None

    def AssignNext(self, nextPlayer):
        self.nextPlayer = nextPlayer

    def AssignPrevious(self, previousPlayer):
        self.previousPlayer = previousPlayer

    @staticmethod
    def AssignNeighbours(previousPlayer, nextPlayer):
        previousPlayer.AssignNext(nextPlayer)
        nextPlayer.AssignPrevious(previousPlayer)

    def perceiveOtherPlayers(self, playerList):
        self.perceivedTeams.clear()
        for player in playerList:
            # # # # # # # # # # # # # # # Filter out 'this' player # # # # # # # # # # # # # # #
            if player.name != self.name and player.index != self.index:
                self.perceivedTeams[player] = None # Add this player to the dictionary with no known team
                #Console.WriteInfo("I (index %d) am now aware of other player %s (index %d)" % (self.index, player.name, player.index), self.name)

    def perceiveTeam(self, otherPlayer, otherTeam):
        self.perceivedTeams[otherPlayer] = otherTeam # Once a player's team is known, assign this team to the player's entry in the dictionary
        
    def tryFindTeammate(self, teamRe, teamKontra):
        reCounter = 0
        kontraCounter = 0
        unknown = []
        for other, team in self.perceivedTeams.items():
            if team is None:
                unknown.append(other)
            elif team is teamRe:
                reCounter += 1
            elif team is teamKontra:
                kontraCounter += 1
        if len(unknown) == 1: # If there is exactly one player whose team affiliation we do not yet know
            if reCounter == 2: # And we know that the other two are in team 'Re'
                self.perceivedTeams[unknown[0]] = teamKontra # Assign the remaining player to team Kontra (our teammate)
            elif kontraCounter == 2: # Else, if we know that the other two are in team 'Kontra'
                self.perceivedTeams[unknown[0]] = teamRe # Assign the remaining player to team Re (our teammate)

    def isTeammate(self, otherPlayer) -> bool: # otherPlayer should be an object of type "Player" or an int of the player's index
        for player, team in self.perceivedTeams.items():
            if isinstance(otherPlayer, Player): # If we were passed a Player object
                if player.index == otherPlayer.index:
                    return team is not None and team == self.team # Team must not be None
            elif isinstance(otherPlayer, int): # If we were passed an int instead of a Player object
                if player.index == otherPlayer:
                    return team is not None and team == self.team
        if isinstance(otherPlayer, Player): # If we were passed a Player object
            Console.WriteError("I (index %d) have not perceived player %s (index %d) before! Don't know who that is!" % (self.index, otherPlayer.name, otherPlayer.index), "%s (Player %d)" % (self.name, self.index))
        elif isinstance(otherPlayer, int): # If we were passed an int instead of a Player object
            Console.WriteError("I (index %d) have not perceived player %d before! Don't know who that is!" % (self.index, otherPlayer), "%s (Player %d)" % (self.name, self.index))
        else:
            raise ValueError("'otherPlayer' was neither a Player nor an int!")
        return False

    def handCards(self, cards, teamRe, teamKontra) -> int:
        self.cards = cards
        queenCounter = 0
        for card in self.cards:
            if card.cardType == Card.QUEEN_OF_CLUBS_TYPE:
                queenCounter += 1
        hasQueenOfClubs = queenCounter > 0 # see if player has at least one queen of clubs
        if hasQueenOfClubs: # If player has at least one Queen of Clubs
            teamRe.AddMember(self)
            self.team = teamRe
        else: # If player has no Queen of Clubs
            teamKontra.AddMember(self)
            self.team = teamKontra
        for otherPlayer in self.perceivedTeams:
            self.perceivedTeams[otherPlayer] = teamKontra if hasQueenOfClubs else teamRe # Assume that all others players are on the opposing teams until proven otherwise
        return queenCounter

    def removeFirstOccurrence(self, cardType: int):
        toDelete = None
        for card in self.cards:
            if card.cardType == cardType:
                toDelete = card
                break
        if toDelete is not None:
            self.cards.remove(toDelete)
            return True
        return False

    def canFollowCard(self, leadingCard: Card):
        if leadingCard is None:
            return True
        for card in self.cards:
            if leadingCard.isTrump: # If leading_card is a trump
                if card.isTrump:
                    return True
            elif not card.isTrump: # If neither cards are trumps
                if card.suit == leadingCard.suit:
                    return True
        return False

    def hasCardType(self, cardType: int):
        for card in self.cards:
            if card.cardType == cardType:
                return True
        return False

    def tryGetCardFromHand(self, cardType) -> Card:
        for card in self.cards:
            if card.cardType == cardType:
                return card
        return Card.CARDTYPES[cardType].clone()

    def getFeedback(self, cardType: int, leadingCard: Card):
        if not self.hasCardType(cardType):
            return CardFeedback.NOT_IN_HAND
        card = self.tryGetCardFromHand(cardType)
        if leadingCard is None: # we are leading the trick, we can play anything
            return CardFeedback.OK
        canFollow = self.canFollowCard(leadingCard)
        if not canFollow: # If we cannot follow suit, we can play anything
            return CardFeedback.OK_COULD_NOT_FOLLOW_SUIT
        else: # If we are able to follow along with leading_card
            if leadingCard.isTrump: # If leading_card is trump
                return CardFeedback.OK if card.isTrump else CardFeedback.NOT_ALLOWED # Only allow other trumps
            else: # If leading_card is NOT a trump
                return CardFeedback.OK if not card.isTrump and card.suit == leadingCard.suit else CardFeedback.NOT_ALLOWED # Only allow non-trump cards of the same suit

    def pickCardFromHand(self) -> Card: # Pick a random card from the current player hand
        return self.cards[np.random.randint(0, len(self.cards))]

    def tryFollowSuit(self, leadingCard: Card) -> Card:
        if leadingCard is None or not self.canFollowCard(leadingCard):
            return self.pickCardFromHand()
        for card in self.cards:
            if self.getFeedback(card.cardType, leadingCard) == CardFeedback.OK:
                return card

    def __str__(self):
        return "%s (%s) is holding %d cards: %s" % (self.name, self.team.name if self.team != None else "No Team", len(self.cards), self.cards)

    def __repr__(self):
        return self.__str__()