# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

# Adapted from this stack overflow answer https://stackoverflow.com/questions/7323664/python-generator-pre-fetch
# by Winston Ewert

# Usage: If you have a generator function (i.e. one that ends with yield <something>, you can wrap that
# function in the BackgroundGenerator.
#
# For example:
#
# for batch in BackgroundGenerator(iterate_minibatches(data)):
#    do something to batch
#


import threading
import queue

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator,max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
             raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self