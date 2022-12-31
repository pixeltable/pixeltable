class Error(Exception):
    pass


class DuplicateNameError(Exception):
    pass


class UnknownEntityError(Exception):
    pass


class BadFormatError(Exception):
    pass


class DirectoryNotEmptyError(Exception):
    pass


class InsertError(Exception):
    pass


class OperationalError(Exception):
    pass
