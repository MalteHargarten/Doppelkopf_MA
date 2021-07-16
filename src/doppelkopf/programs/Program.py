from sys import argv
from typing import List, Tuple
from abc import ABC, abstractmethod
from doppelkopf.utils.Console import Console
from doppelkopf.programs.Argument import Argument
from doppelkopf.programs.OptionalArgument import OptionalArgument

class Program(ABC):
    def __init__(self, requiredArguments: List[Argument], optionalArguments: List[OptionalArgument]):
        self.requiredArguments = requiredArguments
        self.optionalArguments = optionalArguments
        self.optionalArguments.append(OptionalArgument(name="DEBUG", expectedType=bool, defaultValue=False))

    def tryGetArgumentIndex(self, argNames: List[str], arg: str):
        try:
            return argNames.index(arg)
        except ValueError:
            return None

    def isRequiredArg(self, requiredArgNames: List[str], argName: str) -> Tuple[bool, int]:
        index = self.tryGetArgumentIndex(requiredArgNames, argName)
        return index is not None, index

    def isOptionalArg(self, optionalArgNames: List[str], argName: str) -> Tuple[bool, int]:
        index = self.tryGetArgumentIndex(optionalArgNames, argName)
        return index is not None, index

    def tryProcessAsRequiredArg(self, index: int, argValue: str) -> bool:
        return self.requiredArguments[index].TryAssign(argValue)

    def tryProcessAsOptionalArg(self, index: int, argValue: str) ->bool:
        return self.optionalArguments[index].TryAssign(argValue)

    def checkArguments(self, args: List[str]):
        problems = []
        requiredNames = [required.name for required in self.requiredArguments]
        optionalNames = [optional.name for optional in self.optionalArguments]
        for i in range(1, len(args), 2): # Start at 1
            try:
                argName = args[i]
                argValue = args[i + 1]
                isRequired, index = self.isRequiredArg(requiredNames, argName)
                if isRequired:
                    if not self.tryProcessAsRequiredArg(index, argValue):
                        problems.append("Failed to assign '%s' to required argument '%s'" % (argValue, argName))
                else:
                    isOptional, index = self.isOptionalArg(optionalNames, argName)
                    if isOptional:
                        self.tryProcessAsOptionalArg(index, argValue)
                    else:
                        problems.append("Found an unexpected named argument ('%s')!" % (argName))
            except IndexError:
                problems.append("Found a named argument ('%s'), but no corresponding value!" % (argName))
        # # # # # # # # # # # # # # # Any required argument that was not assigned a value causes a problem # # # # # # # # # # # # # # #
        for required in self.requiredArguments:
            if not required.IsAssigned():
                problems.append("Missing a required argument: '%s'" % (required.name))
        # # # # # # # # # # # # # # # Any optional argument that was not assigned a value uses its default # # # # # # # # # # # # # # #
        for optional in self.optionalArguments:
            if not optional.IsAssigned():
                optional.UseDefault()
        # # # # # # # # # # # # # # # If there was at least one problem, return False # # # # # # # # # # # # # # #
        if len(problems) > 0:
            Console.WriteError(problems)
            return False
        return True

    def GetArgumentByName(self, name: str):
        for positional in self.requiredArguments:
            if name == positional.name:
                return positional.givenValue
        for optional in self.optionalArguments:
            if name == optional.name:
                return optional.givenValue
        raise IndexError("No argument by this name: %s" % name)

    def PreRun(self, args) -> bool:
        if not self.checkArguments(args):
            return False
        Console.DEBUGGING = self.GetArgumentByName("DEBUG")
        return True

    @abstractmethod
    def onRun(self):
        pass

    def Run(self, args):
        if not self.PreRun(args):
            return
        self.onRun()