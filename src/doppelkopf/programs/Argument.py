import ast
from abc import ABC
from doppelkopf.utils.Console import Console

class Argument(ABC):
    def __init__(self, name, expectedType):
        self.name = name
        self.expectedType = expectedType
        self.givenValue = None
        self.isAssigned = False

    def TryAssign(self, arg: str) ->bool:
        try:
            if self.expectedType is bool:
                if arg == "True":
                    self.givenValue = True
                elif arg == "False":
                    self.givenValue = False
                else:
                    raise ValueError("Illegal argument")
            elif self.expectedType is list:
                self.givenValue = ast.literal_eval(arg)
            else:
                self.givenValue = self.expectedType(arg)
            self.isAssigned = True
            return True
        except:
            Console.WriteError("Parameter '%s' could not be cast to the expected type %s (got '%s')" % (self.name, str(self.expectedType), arg))
            return False

    def IsAssigned(self):
        return self.isAssigned