import os
from typing import List

class Directory():
    @staticmethod
    def GetDirectories(directoryPath: str) -> List[str]:
        return [directoryPath + i + "/" for i in os.listdir(directoryPath) if not os.path.isfile(os.path.join(directoryPath,i))]

    @staticmethod
    def GetFiles(directoryPath: str) -> List[str]:
        return [directoryPath + i for i in os.listdir(directoryPath) if os.path.isfile(os.path.join(directoryPath,i))]

    @staticmethod
    def GetFilesThatStartWith(directoryPath: str, startsWith: str) -> List[str]:
        return [directoryPath + i for i in os.listdir(directoryPath) if os.path.isfile(os.path.join(directoryPath,i)) and i.startswith(startsWith)]

    @staticmethod
    def GetFilesThatContain(directoryPath: str, contains: str) -> List[str]:
        return [directoryPath + i for i in os.listdir(directoryPath) if os.path.isfile(os.path.join(directoryPath,i)) and contains in i]
