import os
from .. import env


class RequirementsSpec(object):
    '''
    Reads depedencies from a requirements.txt file
    and returns an Environment object from it.
    '''
    msg = None

    def __init__(self, filename=None, name=None, **kwargs):
        self.filename = filename
        self.name = name
        self.msg = None

    def can_handle(self):
        return self._valid_file() and self._valid_name()

    def _valid_file(self):
        if os.path.exists(self.filename):
            return True
        else:
            self.msg = "There is no requirements.txt"
            return False

    def _valid_name(self):
        if self.name is None:
            self.msg = "Environment with requierements.txt file needs a name"
            return False
        else:
            return True

    @property
    def environment(self):
        dependencies = []
        with open(self.filename) as reqfile:
            for line in reqfile:
                dependencies.append(line)
        return env.Environment(
            name=self.name,
            dependencies=dependencies
        )
