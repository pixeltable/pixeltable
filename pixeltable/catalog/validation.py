def is_valid_identifier(name: str) -> bool:
    return name.isidentifier() and not name.startswith('_')

def is_valid_path(path: str, empty_is_valid : bool) -> bool:
    if path == '':
        return empty_is_valid
        
    for part in path.split('.'):
        if not is_valid_identifier(part):
            return False
    return True
