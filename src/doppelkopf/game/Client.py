import traceback # For debugging
import socket
import threading
import numpy as np
from typing import List
from doppelkopf.game.Card import Card
from doppelkopf.game.Team import Team
from doppelkopf.utils.File import File
from doppelkopf.game.Player import Player
from doppelkopf.utils.Console import Console
from doppelkopf.game.GameState import GameState
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.utils.SimpleMessaging import Message
from doppelkopf.game.CardFeedback import CardFeedback
from doppelkopf.utils.SimpleMessaging import MessageType
from doppelkopf.utils.SimpleMessaging import MessageStatus
from doppelkopf.utils.SimpleMessaging import SimpleMessaging

class Client():
    def __init__(self, onCardRequested, onPlayerReceived=None, onPlayerListReceived=None, onGameStarted=None, onCardWasOk=None, onCardWasNotOk=None, onStateReceived=None, onTrickCompleted=None, onGameCompleted=None):
        self.mySocket = None
        self.name = ""
        self.myPlayer: Player = None
        self.players: List[Player] = None
        self.teamRe: Team = None
        self.teamKontra: Team = None
        if onCardRequested is None:
            raise RuntimeError("onCardRequested must not be None!")
        self.onCardRequested = onCardRequested # Callback function when the client needs to pick a card. Can't be None
        self.onPlayerReceived = onPlayerReceived if onPlayerReceived is not None else self.noop # Callback function when a player was received from the Server
        self.onPlayerListReceived = onPlayerListReceived if onPlayerListReceived is not None else self.noop # Callback function when a list of all players was received from the Server
        self.onGameStarted = onGameStarted if onGameStarted is not None else self.noop # Callback function when the client starts a new game
        self.onCardWasOk = onCardWasOk if onCardWasOk is not None else self.noop # Callback function when the chosen card was OK
        self.onCardWasNotOk = onCardWasNotOk if onCardWasNotOk is not None else self.noop # Callback function when the chosen card was not OK
        self.onStateReceived = onStateReceived if onStateReceived is not None else self.noop # Callback function when a game state is received
        self.onTrickCompleted = onTrickCompleted if onTrickCompleted is not None else self.noop # Callback function when a trick is completed
        self.onGameCompleted = onGameCompleted if onGameCompleted is not None else self.noop # Callback function when a game is completed
        # # # # # # # # # # # # # # # Counter # # # # # # # # # # # # # # #
        self.okCardCounter = 0
        self.notInHandCounter = 0
        self.notAllowedCounter = 0
        self.cardsPickedCounter = 0
        self.gamesCompletedCounter = 0
        self.gamesWonCounter = 0
        self.gamesLostCounter = 0
        # # # # # # # # # # # # # # # Threading # # # # # # # # # # # # # # #
        self.stopFlag = False
        self.lockStopFlag = threading.RLock()
        Console.WriteDebug("Created", "Client")

    def Connect(self, host, port):
        # # # # # # # # # # # # # # # Connect to Server # # # # # # # # # # # # # # #
        self.mySocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.mySocket.setblocking(True)
        self.mySocket.connect((host, port))
        # # # # # # # # # # # # # # # Receive 'this' player object # # # # # # # # # # # # # # #
        self.myPlayer = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.PLAYER_OBJECT) # Recv() player object
        self.name = "%s (Client %d)" % (self.myPlayer.name, self.myPlayer.index)
        Console.WriteDebug("This client is %s" % (self.myPlayer), self.name)
        self.onPlayerReceived(player=self.myPlayer) # Inform callback of player object
        # # # # # # # # # # # # # # # Receive list of other players # # # # # # # # # # # # # # #
        self.players = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.PLAYERLIST) # Recv() player list
        Console.WriteDebug("Received players list from server: %s" %(self.players), self.name)
        #self.myPlayer.perceiveOtherPlayers(self.players) # Perceive list of other players
        self.onPlayerListReceived(playerList=self.players) # Inform callback of player list
        # # # # # # # # # # # # # # # Receive list of other players # # # # # # # # # # # # # # #
        self.teamRe, self.teamKontra = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.TEAMS) # Recv() and unpack teams-tuple
        Console.WriteDebug("Received teams from server: %s and %s" % (self.teamRe, self.teamKontra), self.name)
        Console.WriteSuccess("Connected to server as player %d (%s)" % (self.myPlayer.index, self.myPlayer.name), self.name)

    def Stop(self):
        with self.lockStopFlag: # Acquire lock before touching the stop flag
            self.stopFlag = True
            return True
            
    def PlayGames(self, numOfGames=None):
        self.okCardCounter = 0
        self.notInHandCounter = 0
        self.notAllowedCounter = 0
        self.cardsPickedCounter = 0
        self.gamesCompletedCounter = 0
        self.gamesWonCounter = 0
        self.gamesLostCounter = 0
        if numOfGames is not None and numOfGames <= 0: # If we are given an invalid number of games (zero or negative numbers)
            return False
        gameCounter = 1
        abort = False
        while True:
            with self.lockStopFlag:
                if self.stopFlag:
                    break # Break out of loop
            if numOfGames is not None and gameCounter > numOfGames: # If we have played as many games as requested
                break # Break out of loop
            Console.WriteSuccess("Waiting for game %d to start" % (gameCounter), self.name)
            if not self.PlayGame(gameCounter): # If client disconnected or client crashed during game
                abort = True
                break # Break out of loop
            gameCounter += 1
        # # # # # # # # # # # # # # # After having played the requested number of games # # # # # # # # # # # # # # #
        if self.gamesCompletedCounter > 0:
            if numOfGames is not None:
                Console.WriteSuccess("%d/%d games were completed successfully!" % (self.gamesCompletedCounter, numOfGames), self.name)
            else:
                Console.WriteSuccess("%d games were completed successfully!" % (self.gamesCompletedCounter), self.name)
            Console.WriteSuccess("%d/%d (%f %%) games were won!" % (self.gamesWonCounter, self.gamesCompletedCounter, (self.gamesWonCounter / self.gamesCompletedCounter)), self.name)
            Console.WriteSuccess("%d/%d cards were OK. %d were NOT_IN_HAND and %d were NOT_ALLOWED" % (self.okCardCounter, self.cardsPickedCounter, self.notInHandCounter, self.notAllowedCounter), self.name)
        if self.cardsPickedCounter > 0:
            percentage_ok = 100 * self.okCardCounter / float(self.cardsPickedCounter)
            percentage_not_in_hand = 100 * self.notInHandCounter / float(self.cardsPickedCounter)
            percentage_not_allowed = 100 * self.notAllowedCounter / float(self.cardsPickedCounter)
            Console.WriteSuccess("\n%f %% Cards OK\n%f %% Cards NOT_IN_HAND\n%f %% Cards NOT_ALLOWED" % (percentage_ok, percentage_not_in_hand, percentage_not_allowed), self.name)
        return not abort # True if not aborted, False if aborted

    def PlayGame(self, g):
        try:
            Doppelkopf.ResetBeforeGame(self.players, self.teamRe, self.teamKontra) # Before the game starts, reset() all player- and team-objects
            ready = False
            disconnect = False
            selfShutdown = False
            while not ready and not disconnect and not selfShutdown:
                with self.lockStopFlag:
                    if self.stopFlag:
                        selfShutdown = True
                SimpleMessaging.SendMessage(self.mySocket, Message.READY_TO_PLAY)
                response = SimpleMessaging.ReceiveMessage(self.mySocket) # Wait for Server to respond
                if response.messageType == MessageType.NOT_YET_READY:
                    ready = False
                elif response.messageType == MessageType.READY_TO_PLAY:
                    ready = True
                elif response.messageType == MessageType.DISCONNECT:
                    disconnect = True
                else:
                    Console.WriteError("Unexpected message from Server: %s" % response, self.name)
            if disconnect:
                Console.WriteDebug("The Server is about to disconnect! Exiting game loop", self.name)
                return False # Break out of the game loop
            if selfShutdown:
                Console.WriteDebug("I was told to stop, so I stop!", self.name)
                return False
            Console.WriteDebug("The Server is ready to play another round", self.name)
            Console.WriteDebug("Starting game %d" % (g), self.name)
            self.onGameStarted() # Callback
            gameState = None
            # # # # # # # # # # # # # # # Receive cards from Server # # # # # # # # # # # # # # #
            cards = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.PLAYER_HAND) # Recv() player hand
            Console.WriteDebug("Received cards from server", self.name)
            self.myPlayer.handCards(cards, self.teamRe, self.teamKontra)
            for trickIndex in range(Doppelkopf.MAX_CARDS_PER_PLAYER): # Loop over the number of tricks
                Console.WriteDebug("Starting trick %d" % (trickIndex + 1), self.name)
                for _ in range(4): # Iterate over all 4 player moves within a trick
                    # # # # # # # # # # # # # # # Receive current game state from Server # # # # # # # # # # # # # # #
                    gameState: GameState = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.GAME_STATE) # Recv() Game State
                    Console.WriteDebug("Received GameState from server", self.name)
                    self.onStateReceived(state=gameState, isFinalState=False) # Callback
                    if gameState.currentPlayerIndex == self.myPlayer.index: # If it's this client's turn
                        chosenCard = None
                        Console.WriteDebug("It seems to be my turn to pick a card", "Client #%d" % (self.myPlayer.index))
                        wrongCardTypes = []
                        while True:
                            chosenCard = self.onCardRequested(state=gameState, wrongCardTypes=wrongCardTypes) # Callback
                            if chosenCard.cardType in wrongCardTypes:
                                Console.WriteError("The Agent picked a card (%s - %d) even though it was clearly forbidden by wrongCardTypes! Check your agent!" % (chosenCard, chosenCard.cardType), self.name)
                                Console.WriteInfo("All cardTypes that are forbidden: ", self.name)
                                for cardType in wrongCardTypes:
                                    Console.WriteInfo("Forbidden cardType %d" % (cardType), self.name)
                            # # # # # # # # # # # # # # # Send chosen card # # # # # # # # # # # # # # #
                            SimpleMessaging.SendMessage(self.mySocket, Message(MessageType.CHOSEN_CARD, chosenCard.cardType)) # Send() the chosen card type
                            Console.WriteDebug("Sent the chosen card %s to the Server!" %(chosenCard), self.name)
                            self.cardsPickedCounter += 1
                            # # # # # # # # # # # # # # # Receive card feedback from Server # # # # # # # # # # # # # # #
                            feedback = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.CARDFEEDBACK) # Recv() Card Feedback
                            Console.WriteDebug("Received feedback from Server", self.name)
                            if feedback == CardFeedback.NOT_ALLOWED or feedback == CardFeedback.NOT_IN_HAND:
                                self.onCardWasNotOk(state=gameState, card=chosenCard, feedback=feedback, trickIndex=trickIndex) # Callback
                                Console.WriteDebug("Card %s was not OK: %s. Adding it to the wrongCardTypes list" % (chosenCard, feedback), self.name)
                                if feedback == CardFeedback.NOT_IN_HAND:
                                    self.notInHandCounter += 1
                                elif feedback == CardFeedback.NOT_ALLOWED:
                                    self.notAllowedCounter += 1
                                wrongCardTypes.append(chosenCard.cardType)
                            else:
                                self.onCardWasOk(state=gameState, card=chosenCard, feedback=feedback) # Callback
                                Console.WriteDebug("Card %s was %s" % (chosenCard, feedback), self.name)
                                self.okCardCounter += 1
                                if not self.myPlayer.removeFirstOccurrence(chosenCard.cardType):
                                    Console.WriteError("Failed to remove card from hand", self.name)
                                break # Break out of endless loop
                        #gameState.currentStack.append(chosenCard) # Append the accepted card to stack
                    else: # If it is NOT our turn
                        wasQueen, queenCounter = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.WAS_QUEEN)
                        if wasQueen: # If the current player actually played a Queen of Clubs
                            self.myPlayer.perceiveTeam(self.players[gameState.currentPlayerIndex], self.teamRe) # Inform myPlayer that currentPlayer is team Re
                            if queenCounter == 2 and self.myPlayer.team is self.teamKontra: # If both Queens have been played and iterPlayer is in team Kontra
                                self.myPlayer.tryFindTeammate(self.teamRe, self.teamKontra)
                # # # # # # # # # # # # # # # Receive final trick state # # # # # # # # # # # # # # #
                finalTrickState: GameState = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.GAME_STATE) # Recv() final game state
                Console.WriteDebug("Received finalTrickState from server", self.name)
                self.onStateReceived(state=finalTrickState, isFinalState=True)
                # # # # # # # # # # # # # # # Receive winner of this trick # # # # # # # # # # # # # # #
                isTrickWinner: bool
                isTeamMateTrickWinner: bool
                isTrickWinner, isTeamMateTrickWinner, trickValue = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.TRICK_COMPLETED) # Recv() index of winner of this trick
                Console.WriteDebug("isTrickWinner: %s. isTeamMateTrickWinner: %s. TrickValue: %d" % (isTrickWinner, isTeamMateTrickWinner, trickValue), self.name)
                if isTrickWinner:
                    Console.WriteDebug("I won this trick!", self.name)
                elif isTeamMateTrickWinner:
                    Console.WriteDebug("My teammate won this trick!")
                self.onTrickCompleted(isTrickWinner=isTrickWinner, isTeamMateTrickWinner=isTeamMateTrickWinner, trickValue=trickValue)
            isGameWinner: bool
            isGameWinner, score = SimpleMessaging.ReceiveMessageData(self.mySocket, MessageType.GAME_COMPLETED) # Recv() name of winner team of this game, along with the score of said team
            if isGameWinner:
                Console.WriteDebug("I'm part of the winning team!", self.name)
                self.gamesWonCounter += 1
            else:
                Console.WriteDebug("Lost this one", self.name)
                self.gamesLostCounter += 1
            self.onGameCompleted(isGameWinner=isGameWinner, score=score)
            self.gamesCompletedCounter += 1
            return True
        except: # If an error is raised anywhere along the game:
            Console.WriteError(traceback.format_exc(), self.name)
            return False
        
    def Disconnect(self):
        SimpleMessaging.SendMessage(self.mySocket, Message.DISCONNECT) # Inform server of our intention to disconnect
        self.mySocket.close()
        Console.WriteSuccess("Disconnected from Server", self.name)
        self.myPlayer = None
        self.players = None
        self.teamRe = None
        self.teamKontra = None

    def noop(self, **kwargs): # A function that does nothing and acts as a default for any callback function that was not set to a real value
        pass