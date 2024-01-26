from pathlib import Path
import time
import hashlib
import random

USER_NAME = "Jane doe"
USER_EMAIL = "jane-doe@uiowa.edu"

CACHE_DIR = Path.home() / ".imagiq"

UID_LEN = 32


def uid_generator():
    string = (str(time.time()) + USER_EMAIL + str(random.random())).encode("utf-8")
    return hashlib.md5(string).hexdigest()
