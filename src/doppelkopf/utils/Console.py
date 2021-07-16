import sys
import threading
from colorama import Fore
from colorama import Style
import colorama
from doppelkopf.utils.Helper import Helper

colorama.init() # Vital, as colorama may not work otherwise (at least on Windows?)

class Console():
    # Colour Code wrapper
    RED = Fore.RED
    YELLOW = Fore.YELLOW
    GREEN = Fore.GREEN
    WHITE = Fore.WHITE

    # Cursor navigation
    UPONELINE = "\033[A"
    PREVIOUSLINESTART = "\033[F"

    lock = threading.RLock()
    DEBUGGING=True
    LEVEL_ALL = 0
    LEVEL_OMIT_INFO = 1
    LEVEL_OMIT_WARNING = 2
    LEVEL_OMIT_SUCCESS = 3
    LEVEL_OMIT_ERROR = 4

    CurrentLevel = LEVEL_ALL

    @staticmethod
    def wrapTextInColour(text: str, colourCode: int):
        return colourCode + text + Style.RESET_ALL

    @staticmethod
    def Write(text, name="", prefix="", colourCode=None, printTime=True):
        if not isinstance(text, str):
            text = str(text)
        text = "%s%s%s%s" % (prefix, (Helper.TimeNowToString() + " ") if printTime else "", ("(" + name + "): " if name is not None and name != "" else ""), text)
        if colourCode is not None:
            text = Console.wrapTextInColour(text, colourCode)
        with Console.lock:
            #print("Trying to print %d characters" % len(text))
            #print(text)
            sys.stdout.write(text)
            sys.stdout.flush()

    @staticmethod
    def WriteLine(text, name="", prefix="", colourCode=None, printTime=True):
        if not isinstance(text, str):
            text = str(text)
        Console.Write(text + "\n", name, prefix, colourCode, printTime=printTime)

    @staticmethod
    def WriteDebug(debugText, name="", prefix="", colourCode=Fore.WHITE, printTime=True):
        if Console.DEBUGGING:
            Console.WriteLine(text=("DEBUG%s: %s" % ((" (" + name + ")") if name is not None and name != "" else "", debugText)), prefix=prefix, colourCode=colourCode, printTime=printTime)

    @staticmethod
    def WriteInfo(infoText, name="", prefix="", colourCode=Fore.WHITE, printTime=True):
        if Console.CurrentLevel < Console.LEVEL_OMIT_INFO:
            Console.WriteLine(text=("INFO%s: %s" % ((" (" + name + ")") if name is not None and name != "" else "", infoText)), prefix=prefix, colourCode=colourCode, printTime=printTime)

    @staticmethod
    def WriteError(errorText, name="", prefix="", colourCode=Fore.RED, printTime=True):
        if Console.CurrentLevel < Console.LEVEL_OMIT_ERROR:
            Console.WriteLine(text=("ERROR%s: %s" % ((" (" + name + ")") if name is not None and name != "" else "", errorText)), prefix=prefix, colourCode=colourCode, printTime=printTime)

    @staticmethod
    def WriteWarning(warningText, name="", prefix="", colourCode=Fore.YELLOW, printTime=True):
        if Console.CurrentLevel < Console.LEVEL_OMIT_WARNING:
            Console.WriteLine(text=("WARNING%s: %s" % ((" (" + name + ")") if name is not None and name != "" else "", warningText)), prefix=prefix, colourCode=colourCode, printTime=printTime)

    @staticmethod
    def WriteSuccess(successText, name="", prefix="", colourCode=Fore.GREEN, printTime=True):
        if Console.CurrentLevel < Console.LEVEL_OMIT_SUCCESS:
            Console.WriteLine(text=("SUCCESS%s: %s" % ((" (" + name + ")") if name is not None and name != "" else "", successText)), prefix=prefix, colourCode=colourCode, printTime=printTime)

    @staticmethod
    def WriteOverPreviousLine(text, name="", prefix="", colourCode=Fore.WHITE, printTime=True):
        Console.WriteLine(text=("%s%s" % ("(" + name + "): " if name is not None and name != "" else "", text)), prefix=("\r" + Console.UPONELINE + prefix), colourCode=colourCode, printTime=printTime)

    @staticmethod
    def ReadInput(prompt: str) -> str:
        Console.WriteLine(prompt)
        return input()