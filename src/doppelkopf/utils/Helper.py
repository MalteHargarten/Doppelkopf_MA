import gc
import sys
import datetime

class Helper():
    '''
    This function is taken from the following SO Link:
    https://stackoverflow.com/a/53705610
    '''
    @staticmethod
    def SizeOf(obj):
        marked = {id(obj)} # Objects (or their references) that we have already processed
        obj_queue = [obj] # Queue (or rather list) of objects we need to process
        total_size = 0 # total size in Bytes
        while obj_queue:
            total_size += sum(map(sys.getsizeof, obj_queue))
            # Lookup all the object referred to by the object in obj_q.
            # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
            all_refr = ((id(o), o) for o in gc.get_referents(*obj_queue))
            # Filter object that are already marked.
            # Using dict notation will prevent repeated objects.
            new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}
            # The new obj_q will be the ones that were not marked,
            # and we will update marked with their ids so we will
            # not traverse them again.
            obj_queue = new_refr.values()
            marked.update(new_refr.keys())
        return total_size

    @staticmethod
    def TimeNowToString():
        now = datetime.datetime.now().time()
        return "[" + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + "]"

    @staticmethod
    def DateTimeNowToString():
        now = datetime.datetime.now()
        date = now.date()
        time = now.time()
        return "%d.%d.%d [%d:%d:%d]"  % (date.day, date.month, date.year, time.hour, time.minute, time.second)