import os
import pickle
from doppelkopf.utils.Console import Console

class File():
    @staticmethod
    def Write(obj, filepath):
        try:
            with open(filepath, 'wb') as fileHandle:
                pickle.dump(obj, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
                return True
        except FileNotFoundError:
            Console.WriteError("File could not be found!", "File.Write()")
        except Exception as error:
            Console.WriteError("File cannot be opened due to unspecified error: %s" % error, "File.Write()")
        return False

    @staticmethod
    def Append(obj, filepath):
        try:
            with open(filepath, 'ab') as fileHandle:
                pickle.dump(obj, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
                return True
        except FileNotFoundError:
            Console.WriteError("File could not be found!", "File.Append()")
        except Exception as error:
            Console.WriteError("File cannot be opened due to unspecified error: %s" % error, "File.Append()")
        return False

    @staticmethod
    def Read(filepath):
        try:
            with open(filepath, 'rb') as fileHandle:
                return pickle.load(fileHandle)
        except FileNotFoundError:
            Console.WriteError("File could not be found!", "File.Read()")
        except Exception as error:
            Console.WriteError("File cannot be opened due to unspecified error: %s" % error, "File.Read()")
        return None

    @staticmethod
    def Delete(filepath):
        try:
            os.remove(filepath)
        except FileNotFoundError: # Catch and ignore the FileNotFoundError
            pass

    '''
    Taken from : https://stackoverflow.com/a/28745948, a superb answer, which returns a generator rather than a list
    '''
    @staticmethod
    def ReadAllAsGenerator(filepath):
        with open(filepath, 'rb') as fileHandle:
            while True:
                try:
                    yield pickle.load(fileHandle)
                except EOFError:
                    break

    @staticmethod
    def ReadAll(filepath):
        with open(filepath, 'rb') as fileHandle:
            objs = []
            while True:
                try:
                    objs.append(pickle.load(fileHandle))
                except EOFError:
                    return objs

    @staticmethod
    def WriteText(text: str, filepath):
        try:
            with open(filepath, 'w') as f: # 'a' will create the file if not exists and overwrite its content if it does
                f.write(text)
        except FileNotFoundError:
            Console.WriteError("File could not be found!", "File.Append()")
        except Exception as error:
            Console.WriteError("File cannot be opened due to unspecified error: %s" % error, "File.WriteText()")

    @staticmethod
    def AppendText(text: str, filepath):
        try:
            with open(filepath, 'a') as f: # 'a' will create the file if not exists and append to it if it does
                f.write(text)
        except FileNotFoundError:
            Console.WriteError("File could not be found!", "File.AppendText()")
        except Exception as error:
            Console.WriteError("File cannot be opened due to unspecified error: %s" % error, "File.AppendText()")

    @staticmethod
    def WriteTextLine(text: str, filepath):
        File.WriteText("%s\n" % text, filepath)