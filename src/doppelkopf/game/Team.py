import numpy as np
from doppelkopf.game.Player import Player
from doppelkopf.game.Doppelkopf import Doppelkopf

class Team():
    def __init__(self, name):
        self.name = name
        self.members = []
        self.wonTricks = []
        self.score = 0

    @staticmethod
    def createTeams():
        teamRe = Team("Re")
        teamKontra = Team("Kontra")
        return teamRe, teamKontra

    def Reset(self):
        self.members.clear()
        self.wonTricks.clear()
        self.score = 0

    def AddMember(self, player: Player):
        self.members.append(player)

    def RemoveMember(self, player: Player):
        self.members.remove(player)

    def AddTrick(self, trickStack):
        self.wonTricks.append(trickStack)

    def CountTeamScore(self):
        self.score = 0
        for trick in self.wonTricks:
            self.score += Doppelkopf.GetTrickValue(trick)
        return self.score

    def Score(self):
        return self.score

    def __str__(self):
        result = self.name + " ("
        for i in range(len(self.members)):
            if i > 0:
                result += ", "
            result += self.members[i].name
        return result + ")"