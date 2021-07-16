import sys
from doppelkopf.utils.Console import Console
#Console.CurrentLevel = Console.LEVEL_OMIT_INFO
from doppelkopf.programs.Program import Program
from doppelkopf.agents.RulebasedPlayer import RulebasedPlayer
from doppelkopf.programs.OptionalArgument import OptionalArgument

class RunRulebasedPlayer(Program):
    def __init__(self):
        optionals = [
            OptionalArgument("host", expectedType=str, defaultValue='localhost'),
            OptionalArgument("port", expectedType=int, defaultValue=8088),
            OptionalArgument("numOfGames", expectedType=int, defaultValue=None),
            OptionalArgument(name="logFile", expectedType=str, defaultValue=None),
        ]
        super(RunRulebasedPlayer, self).__init__([], optionals)

    def onRun(self):
        # # # # # # # # # # # # # # # Get values from parameters # # # # # # # # # # # # # # #
        host = self.GetArgumentByName("host")
        port = self.GetArgumentByName("port")
        numOfGames = self.GetArgumentByName("numOfGames")
        logFile = self.GetArgumentByName("logFile")
        # # # # # # # # # # # # # # # Check special conditions # # # # # # # # # # # # # # #
        if numOfGames is not None and numOfGames <= 0:
            exit("numOfGames was set to 0 (zero). Exiting...")
        # # # # # # # # # # # # # # # Play games # # # # # # # # # # # # # # #
        player = RulebasedPlayer()
        player.ConnectToServer(host, port)
        player.PlayGames(numOfGames)
        if logFile is not None:
            player.LogReport(logFile, numOfGames)
        player.DisconnectFromServer()

def main(args):
    program = RunRulebasedPlayer()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)