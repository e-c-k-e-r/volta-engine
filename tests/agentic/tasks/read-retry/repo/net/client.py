import time
from .backoff import next_delay


def call(fn, attempts=6):
    for i in range(attempts):
        try:
            return fn()
        except IOError:
            time.sleep(next_delay(i))
    raise IOError("gave up")
