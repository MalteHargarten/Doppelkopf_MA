import sys
from doppelkopf.utils.Console import Console
#Console.CurrentLevel = Console.LEVEL_OMIT_INFO
from doppelkopf.game.Server import Server
from doppelkopf.programs.Program import Program
from doppelkopf.programs.OptionalArgument import OptionalArgument

class RunServer(Program):
    def __init__(self):
        optionals = [
            OptionalArgument(name="host", expectedType=str, defaultValue="localhost"),     
            OptionalArgument(name="port", expectedType=int, defaultValue=8088),
        ]
        super(RunServer, self).__init__([], optionals) # Call base constuctor

    def onRun(self):
        # # # # # # # # # # # # # # # Get values from parameters # # # # # # # # # # # # # # #
        host = self.GetArgumentByName("host")
        port = self.GetArgumentByName("port")
        # # # # # # # # # # # # # # # Run Server # # # # # # # # # # # # # # #
        server = Server("Server", host, port, numOfClientsRequired=4)
        server.Start()
        Console.ReadInput("Press Enter to Stop the Server\n")
        server.Stop()

def main(args):
    program = RunServer()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)