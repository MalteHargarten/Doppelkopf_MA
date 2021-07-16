import ast
from doppelkopf.utils.Console import Console
from doppelkopf.programs.Argument import Argument

class OptionalArgument(Argument):
    def __init__(self, name, expectedType, defaultValue):
        super(OptionalArgument, self).__init__(name, expectedType)
        self.defaultValue = defaultValue

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
        except:
            self.UseDefault()
            Console.WriteError("Parameter '%s' could not be cast to the expected type %s (got '%s'). Using default %s instead" % (self.name, str(self.expectedType), arg, str(self.defaultValue)))
        self.isAssigned = True
        return True
        
    def UseDefault(self):
        self.givenValue = self.defaultValue