import traceback # For debugging
import socket
import threading
import numpy as np
from typing import List
from doppelkopf.utils.Console import Console
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.game.Team import Team
from doppelkopf.game.Card import Card
from doppelkopf.utils.SimpleMessaging import Message
from doppelkopf.utils.SimpleMessaging import MessageType
from doppelkopf.utils.SimpleMessaging import MessageStatus
from doppelkopf.game.CardFeedback import CardFeedback
from doppelkopf.utils.SimpleMessaging import SimpleMessaging
from doppelkopf.game.Player import Player
from doppelkopf.game.GameState import GameState

class Server():
    THREAD_NAME_WAITING = "Thread_Waiting_For_Clients"
    THREAD_NAME_PLAY = "Thread_Play_Games"

    def __init__(self, name, host, port, numOfClientsRequired):
        self.name = name
        self.host = host
        self.port = port
        self.numOfClientsRequired = numOfClientsRequired
        self.serverSocket = None
        self.players: List[Player] = Player.createPlayers()
        Console.WriteDebug("%s players instantiated" % (len(self.players)), self.name)
        for player in self.players:
            player.perceiveOtherPlayers(self.players) # Make each player aware of the others
        self.playerSocketPairing = {}
        for player in self.players:
            self.playerSocketPairing[player] = None
        self.teamRe, self.teamKontra = Team.createTeams()
        self.cardDeck = Card.createDeck()
        # # # # # # # # # # # # # # # Threading # # # # # # # # # # # # # # #
        self.threadRunGames = None
        self.threadWaitForClients = None
        self.stopFlag = False
        self.lockStartMethod = threading.Lock() # No RLock here because this lock shall only be acquired one at a time, not multiple times (not even by the same thread)
        self.lockStopMethod = threading.Lock()  # No RLock here because this lock shall only be acquired one at a time, not multiple times (not even by the same thread)
        self.lockStopFlag = threading.RLock()
        self.lockPlayerSocketPairing = threading.RLock()
        self.lockThreadRunGames = threading.RLock()
        self.lockThreadWaitForClients = threading.RLock()
        Console.WriteSuccess("Created", self.name)

    def Start(self):
        with self.lockStartMethod: # Acquire lock so that only one thread can execute the Start() method at a time
            if self.isRunningThreadWait() or self.isRunningThreadPlay():
                return False # If either of the two threads are already running, do nothing
            with self.lockStopFlag: # Acquire lock before accessing the stop flag
                self.stopFlag = False # Set stop flag to False
            # # # # # # # # # # # # # # # Start up socket # # # # # # # # # # # # # # #
            Console.WriteDebug("Trying to bind socket to host '%s' on port %d" % (self.host, self.port), self.name)
            self.serverSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.serverSocket.bind((self.host, self.port))
            self.serverSocket.listen(self.numOfClientsRequired)
            Console.WriteDebug("Socket bound to host '%s' on port %d" % (self.host, self.port), self.name)
            # # # # # # # # # # # # # # # Start Thread that waits for clients to connect # # # # # # # # # # # # # # #
            self.threadWaitForClients = threading.Thread(target=self.waitForClients, name=Server.THREAD_NAME_WAITING)
            self.threadWaitForClients.start()
            # # # # # # # # # # # # # # # Start Thread that plays games as long as enough clients are connected # # # # # # # # # # # # # # #
            self.threadRunGames = threading.Thread(target=self.PlayGames, name=Server.THREAD_NAME_PLAY)
            self.threadRunGames.start()
            Console.WriteSuccess("Started", self.name)
            return True

    def Stop(self):
        with self.lockStopMethod: # Acquire lock so that only one thread can execute the Stop() method at a time
            if not self.isRunningThreadWait() and not self.isRunningThreadPlay():
                return False # If neither of the two threads are running, do nothing
            with self.lockStopFlag: # Acquire lock before accessing the stop flag
                self.stopFlag = True # Set stop flag to True
            with self.lockThreadWaitForClients: # Acquire lock before accessing the 'wait for clients' thread object
                if self.isRunningThreadWait(): # If the 'wait for clients' thread is running
                    if threading.currentThread().name != self.threadWaitForClients.name: # If the calling thread's name is not the name of the 'wait for clients' thread
                        selfSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
                        selfSocket.setblocking(True)
                        selfSocket.connect((self.host, self.port)) # Connect to thyself
                        disconnectMsg = SimpleMessaging.ReceiveMessage(selfSocket)
                        if disconnectMsg.messageType != MessageType.DISCONNECT:
                            Console.WriteError("Expected DISCONNECT message, got %s instead" % disconnectMsg, self.name)
                        #Console.WriteSuccess("Got the expected disconnect message", self.name)
                        selfSocket.close() # Close connection to thyself
                        self.threadWaitForClients.join() # Wait for thread to complete
                        self.threadWaitForClients = None # Set thread variable to None to signify that this variable can be reused
            with self.lockThreadRunGames: # Acquire lock before accessing the thread object
                if self.isRunningThreadPlay(): # If the 'play games' thread is running
                    if threading.currentThread().name != self.threadRunGames.name: # If the calling thread's name is not the name of the 'play games' thread
                        self.threadRunGames.join() # Wait for thread to complete
                        self.threadRunGames = None # Set thread variable to None to signify that this variable can be reused
            # # # # # # # # # # # # # # # Clean up (close sockets) # # # # # # # # # # # # # # #
            with self.lockPlayerSocketPairing:
                for clientSocket in self.playerSocketPairing.values():
                    if clientSocket is not None:
                        clientSocket.close()
            self.serverSocket.close() # Close server socket
            Console.WriteSuccess("Stopped", self.name)
            return True

    def isRunningThreadWait(self):
        with self.lockThreadWaitForClients:
            return self.threadWaitForClients is not None

    def isRunningThreadPlay(self):
        with self.lockThreadRunGames:
            return self.threadRunGames is not None

    def disconnect_client_socket(self, client_socket):
        SimpleMessaging.SendMessage(client_socket, Message.DISCONNECT) # Inform client to disconnect this connection
        client_socket.close() # Close the connection immediately

    def getFirstAvailablePlayer(self) -> Player:
        with self.lockPlayerSocketPairing:
            for player, socket in self.playerSocketPairing.items():
                if socket is None:
                    return player
        return None
    
    def getNumOfConnectedClients(self) -> int:
        with self.lockPlayerSocketPairing: # Acquire lock before accessing the list of client sockets            
            counter = 0
            for socket in self.playerSocketPairing.values():
                if socket is not None:
                    counter += 1
            return counter

    def sendToAll(self, message: Message):
        with self.lockPlayerSocketPairing:
            for iter_client_socket in self.playerSocketPairing.values(): # Wait for all clients to confirm whether or not they wish to begin playing
                if iter_client_socket is not None:
                    SimpleMessaging.SendMessage(iter_client_socket, message)

    def waitForClients(self):
        try:
            while True:
                with self.lockStopFlag: # Acquire lock before accessing the stop flag
                    if self.stopFlag: # If stop flag is set to True
                        return # Exit and terminate this thread
                Console.WriteDebug("Now waiting for a Client", self.name)
                newClientSocket, address = self.serverSocket.accept() # accept() is blocking until a connection is established
                availablePlayer = self.getFirstAvailablePlayer()
                with self.lockStopFlag: # Acquire lock before accessing the stop flag
                    if self.stopFlag:
                        self.disconnect_client_socket(newClientSocket)
                    elif availablePlayer is None: # If there is no available player slot
                        self.disconnect_client_socket(newClientSocket)
                    else: # If there is an available player slot
                        Console.WriteDebug("New client connected from %s" % str(address), self.name)
                        newClientSocket.setblocking(True)
                        SimpleMessaging.SendMessage(newClientSocket, Message(MessageType.PLAYER_OBJECT, availablePlayer)) # Send() player object to client
                        with self.lockPlayerSocketPairing: # Acquire lock before accessing the list of client sockets
                            # # # # # # # # # # # # # # # Store new client socket as associated value of available player key # # # # # # # # # # # # # # #
                            self.playerSocketPairing[availablePlayer] = newClientSocket # Associate this new socket to the player object
                            # # # # # # # # # # # # # # # Send player and team info to new client socket # # # # # # # # # # # # # # #
                            SimpleMessaging.SendMessage(newClientSocket, Message(MessageType.PLAYERLIST, self.players)) # Send() player list to all clients
                            SimpleMessaging.SendMessage(newClientSocket, Message(MessageType.TEAMS, (self.teamRe, self.teamKontra))) # Send() teams to all clients
                        Console.WriteSuccess("%s has joined the Server" % availablePlayer.name, self.name)
        except:
            Console.WriteError(traceback.format_exc(), self.name)
            self.Stop() # The Server performs a Self-Stop if any error is raised while accepting new clients
            self.threadWaitForClients = None # Set thread variable to None to signify that this variable can be reused
            return # Exit and terminate this thread

    def PlayGames(self):
        try:
            gameCounter = 1
            gameStateIndex = 0
            cardSupply = np.zeros(shape=(Card.NUM_CARDTYPES), dtype=np.int) # An entry for every cardType between 0 and 2 (inclusive)
            while True:
                with self.lockStopFlag: # Acquire lock before accessing the stop flag
                    if self.stopFlag: # If stop flag is set to True
                        return # Exit and terminate this thread
                with self.lockPlayerSocketPairing:
                    # # # # # # # # # # # # # # # Check if all clients want to start playing # # # # # # # # # # # # # # #
                    allconnected = True
                    allready = True
                    for iterPlayer, iterClientSocket in self.playerSocketPairing.items(): # Wait for all clients to confirm whether or not they wish to begin playing
                        if iterClientSocket is not None:
                            #SimpleMessaging.SendMessage(iter_client_socket, Message.READY_TO_PLAY) # Inform this client that the Server is ready to play
                            Console.WriteDebug("Waiting for continue_msg from %s" % iterPlayer.name, self.name)
                            continueMsg = SimpleMessaging.ReceiveMessage(iterClientSocket) # Wait for the Client to respond
                            if continueMsg.messageType == MessageType.DISCONNECT: # If the Client wishes to disconnect, ...
                                allready = False
                                iterClientSocket.close() # ... close the socket
                                self.playerSocketPairing[iterPlayer] = None # Mark this player's socket as 'None' to indicate that this player is available again
                                Console.WriteLine("Client Socket %d (%s) has disconnected" % (iterPlayer.index, iterPlayer.name), self.name)
                            else:
                                Console.WriteDebug("Message received: %s" % continueMsg, self.name)
                        else:
                            allconnected = False
                    self.sendToAll(Message.READY_TO_PLAY if allready and allconnected else Message.NOT_YET_READY)
                    if allconnected and allready: # Proceed only if there are enough players and all players signaled that they are ready                
                        # # # # # # # # # # # # # # # play games # # # # # # # # # # # # # # #
                        Console.WriteSuccess("Starting game %d" % (gameCounter), self.name)
                        gameStateIndex = 0
                        queenCounter = 0 # Counter for the number of Queen of Clubs, that have been played
                        cardSupply.fill(2) # Before each game, set the list of cards, that have already been played, to all two's because all card types are in the deck twice
                        Doppelkopf.ResetBeforeGame(self.players, self.teamRe, self.teamKontra) # Before the game starts, reset() all player- and team-objects
                        numOfCards = Doppelkopf.dealCards(self.players, self.teamRe, self.teamKontra, self.cardDeck)
                        for iterPlayer, iterClientSocket in self.playerSocketPairing.items():
                            SimpleMessaging.SendMessage(iterClientSocket, Message(MessageType.PLAYER_HAND, iterPlayer.cards)) # Send() player hand to client
                        Console.WriteDebug("Dealt cards to players: %s" % (self.players), self.name)
                        currentPlayer = self.players[int(np.random.randint(0, len(self.players)))] # Pick a random player to start the first trick (after that, the winner of each trick starts the next one)
                        currentClientSocket = self.playerSocketPairing[currentPlayer]
                        for trickIndex in range(numOfCards): # Play as many tricks as each player holds cards
                            Console.WriteDebug("Starting trick %d. Starting player: %s" % (trickIndex + 1, currentPlayer.name), self.name)
                            stack: List[Card] = []
                            playOrder: List[Player] = []
                            Console.WriteDebug("Constructing play order", self.name)
                            temp: Player = currentPlayer
                            for _ in range(len(self.players)):                                
                                playOrder.append(temp)
                                temp = temp.nextPlayer
                                Console.WriteDebug("added %s to play order" % playOrder[-1].name, self.name)
                            for _ in range(len(self.players)): # Do one move for each player
                                for iterPlayer, iterClientSocket in self.playerSocketPairing.items(): # In each move, inform all players of the current game state and receive a response from the one player, whose turn it currently is
                                    # Create game state this player
                                    teamAffiliations = Server.constructTeamAffiliations(iterPlayer, playOrder)
                                    gameState = GameState(gameStateIndex, iterPlayer.cards, stack, cardSupply, currentPlayer.index, teamAffiliations, False)
                                    # Send game states to this player
                                    SimpleMessaging.SendMessage(iterClientSocket, Message(MessageType.GAME_STATE, gameState)) # Send() game state
                                    Console.WriteDebug("Game state sent to client socket %d (%s)" % (iterPlayer.index, iterPlayer.name), self.name)
                                gameStateIndex += 1
                                chosenCardType = None
                                feedback = None
                                while True:
                                    Console.WriteDebug("Waiting for chosen card from %s (client socket %d)" % (currentPlayer.name, currentPlayer.index), self.name)
                                    chosenCardType = SimpleMessaging.ReceiveMessageData(currentClientSocket, MessageType.CHOSEN_CARD) # Recv() chosen card
                                    Console.WriteDebug("Received chosen card type %d from %s (client socket %d)" % (chosenCardType, currentPlayer.name, currentPlayer.index), self.name)
                                    feedback = currentPlayer.getFeedback(chosenCardType, stack[0] if len(stack) > 0 else None)
                                    Console.WriteDebug("Card type %s was %s" % (chosenCardType, feedback), self.name)
                                    SimpleMessaging.SendMessage(currentClientSocket, Message(MessageType.CARDFEEDBACK, feedback)) # Send() feedback to client
                                    Console.WriteDebug("Feedback sent to client socket %d (%s)" % (currentPlayer.index, currentPlayer.name), self.name)
                                    if feedback == CardFeedback.OK or feedback == CardFeedback.OK_COULD_NOT_FOLLOW_SUIT:
                                        break # Break out of loop
                                    #elif feedback == CardFeedback.NOT_IN_HAND or feedback == CardFeedback.NOT_ALLOWED: # If player does not have this card (or card is not allowed)
                                        #Console.WriteWarning("Card type %d was not OK (%s). Leading card was %s. Client must pick a new card" % (chosenCardType, feedback, stack[0] if len(stack) > 0 else "None"), self.name)
                                card = currentPlayer.tryGetCardFromHand(chosenCardType) # Try and get the card from the current player's hand (if current_player has it)
                                if currentPlayer.hasCardType(chosenCardType): # If current player has the card, remove it from his hand
                                    currentPlayer.removeFirstOccurrence(chosenCardType)
                                stack.append(card) # Append card to stack
                                cardSupply[chosenCardType] -= 1 # Decrease the card counter for this cardType to indicate that this card has been played
                                wasQueen = card.cardType == Card.QUEEN_OF_CLUBS_TYPE
                                if wasQueen:
                                    queenCounter += 1
                                for iterPlayer, iterClientSocket in self.playerSocketPairing.items():
                                    if iterPlayer is not currentPlayer: # ... which aren't the current player
                                        if wasQueen: # If a Queen of Clubs was played
                                            iterPlayer.perceiveTeam(currentPlayer, self.teamRe) # Inform player that currentPlayer is in team Re
                                            Console.WriteDebug("%s (%s) is now aware that %s is in team Re!" % (iterPlayer.name, iterPlayer.team.name, currentPlayer.name), self.name)
                                            if queenCounter == 2 and iterPlayer.team is self.teamKontra: # If both Queens have been played and iterPlayer is in team Kontra
                                                iterPlayer.tryFindTeammate(self.teamRe, self.teamKontra)
                                        SimpleMessaging.SendMessage(iterClientSocket, Message(MessageType.WAS_QUEEN, (wasQueen, queenCounter)))
                                # After sending the game state to all players (and receiving a response from the one whose turn it is), increment to the next player
                                currentPlayer = currentPlayer.nextPlayer # Get the next player
                                currentClientSocket = self.playerSocketPairing[currentPlayer]
                            for iterPlayer, iterClientSocket in self.playerSocketPairing.items(): # Iterate over all players and send them the final trick state before resetting for the next trick
                                teamAffiliations = Server.constructTeamAffiliations(iterPlayer, playOrder)
                                finalGameState = GameState(gameStateIndex, iterPlayer.cards, stack, cardSupply, currentPlayer.index, teamAffiliations, True)
                                SimpleMessaging.SendMessage(iterClientSocket, Message(MessageType.GAME_STATE, finalGameState)) # Send() final game state
                                Console.WriteDebug("Final trick state sent to client socket %d (%s)" % (iterPlayer.index, iterPlayer.name), self.name)                                
                            gameStateIndex += 1
                            # Now that all players have played a card, determine the winner of this trick
                            bestCardIndex = Doppelkopf.findBestCard(stack) # Returns the index of the best card within the stack
                            trickWinningPlayer: Player = playOrder[bestCardIndex] # Winner is the player's index who played the best card within the stack
                            trickWinningPlayer.team.AddTrick(stack)
                            trickValue = Doppelkopf.GetTrickValue(stack)
                            iterPlayer: Player
                            for iterPlayer, iterClientSocket in self.playerSocketPairing.items():
                                isTrickWinner = iterPlayer.index == trickWinningPlayer.index # If the trick was won by iterPlayer
                                isTeamMateTrickWinner = iterPlayer.isTeammate(trickWinningPlayer) if not isTrickWinner else False # If the trick was won by their (known!) teammate. Does not count as a win if the player is not aware that the winner is their teammate!
                                SimpleMessaging.SendMessage(iterClientSocket, Message(MessageType.TRICK_COMPLETED, (isTrickWinner, isTeamMateTrickWinner, trickValue)))
                            Console.WriteDebug("Trick %d completed! Winner is: %s" % (trickIndex + 1, trickWinningPlayer.name), self.name)
                            currentPlayer = trickWinningPlayer # The winner of this trick may start the next trick
                            currentClientSocket = self.playerSocketPairing[currentPlayer] # The socket associated with this player is the next current socket
                        winningTeam = Doppelkopf.GetWinner(self.teamRe, self.teamKontra)
                        Console.WriteDebug("Game %d was won by team %s with %d points" % (gameCounter, winningTeam, winningTeam.Score()), self.name)
                        #self.sendToAll(Message(MessageType.GAME_WINNING_TEAM, (winningTeam.name, score)))
                        iterPlayer: Player
                        for iterPlayer, iterClientSocket in self.playerSocketPairing.items():
                            isGameWinner = iterPlayer.team == winningTeam
                            SimpleMessaging.SendMessage(iterClientSocket, Message(MessageType.GAME_COMPLETED, (isGameWinner, iterPlayer.team.Score())))
                        gameCounter += 1 # Increment game counter
        except:
            Console.WriteError(traceback.format_exc(), self.name)
            self.Stop() # The Server performs a Self-Stop if any error is raised during game play
            self.threadRunGames = None # Set thread variable to None to signify that this variable can be reused
            return # Exit and terminate this thread

    @staticmethod
    def constructTeamAffiliations(player: Player, playOrder: List[Player]) -> np.ndarray:
        teamAffiliations = np.zeros(shape=(Doppelkopf.MAX_PLAYERS_IN_GAME), dtype=np.float32)
        teamAffiliations[player.index] = 1 # 1 indicates thyself
        for i, orderedPlayer in enumerate(playOrder):
            if orderedPlayer.index == player.index or player.isTeammate(orderedPlayer): # if 'orderedPlayer' is 'player' OR if 'orderedPlayer' is a teammate of 'player'
                teamAffiliations[i] = 1 # 1 indicates either thyself or a teammate
            else:
                teamAffiliations[i] = 0 # 0 indicates an opponent (or someone who's team affiliation is not yet known)
        return teamAffiliations