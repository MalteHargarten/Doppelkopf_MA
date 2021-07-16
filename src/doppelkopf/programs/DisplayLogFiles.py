import sys
from doppelkopf.utils.File import File
from doppelkopf.utils.Console import Console
from doppelkopf.programs.Program import Program
from doppelkopf.utils.Directory import Directory
from doppelkopf.programs.OptionalArgument import OptionalArgument

class DisplayLogFiles(Program):
    def __init__(self):
        optionals = [
            OptionalArgument(name="directory", expectedType=str, defaultValue=None),
            OptionalArgument(name="logFile", expectedType=str, defaultValue=None),
            OptionalArgument(name="startsWith", expectedType=str, defaultValue=None),
        ]
        super(DisplayLogFiles, self).__init__([], optionals)

    def onRun(self):
        directory: str = self.GetArgumentByName("directory")
        logFile = self.GetArgumentByName("logFile")
        startsWith = self.GetArgumentByName("startsWith")
        logFiles = []
        if directory is not None:
            if not directory.endswith("/"):
                directory += "/"
            if startsWith is not None:
                logFiles = Directory.GetFilesThatStartWith(directory, startsWith)
            else:
                logFiles = Directory.GetFiles(directory)
        if logFile is not None:
            logFiles.append(logFile)
        if len(logFiles) == 0:
            Console.WriteWarning("Nothing to display here. Move along!")
        for l in logFiles:
            Console.WriteSuccess("# # # # # # # # # # # # # # # %s # # # # # # # # # # # # # # #" % (l))
            reports = File.ReadAll(l)
            for report in reports:
                Console.WriteInfo("# # # # # # # # # # # # # # # %s # # # # # # # # # # # # # # #" % (type(report)))
                Console.WriteInfo(report)

def main(args):
    displayProgram = DisplayLogFiles()
    displayProgram.Run(args)

if __name__ == '__main__':
    main(sys.argv)