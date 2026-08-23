"""Loading Python files that are not importable by name, such as an application file given on the command line."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import inspect
import re
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

# the modules loaded by load_file(), keyed by resolved path; _lock guards both this dict and the exec that
# populates it, and is reentrant because executing a file loads the siblings it imports
_loaded_modules: dict[Path, ModuleType] = {}
_lock = threading.RLock()

# the synthetic packages standing for the directories of loaded files, keyed by directory
_dir_modules: dict[Path, ModuleType] = {}

_NON_IDENTIFIER_CHARS = re.compile(r'\W')


def _file_stem(path: Path) -> str:
    """The stem of path as an identifier, with anything that cannot appear in one replaced by an underscore.

    A file name is under no obligation to be a module name: it can start with a digit, which an identifier
    cannot, and validate_symbol_path() rejects a symbol path built from one.
    """
    stem = _NON_IDENTIFIER_CHARS.sub('_', path.stem)
    return stem if stem.isidentifier() else f'_{stem}'


def resolve_file_symbol(file: str, symbol_path: str) -> object | None:
    """The symbol that symbol_path names in file, or None if the file does not define it.

    The first element of symbol_path is the name of the file itself, which is there for error messages; the
    elements after it are the qualname of the symbol.
    """
    module = load_file(file)
    obj: object = module
    for el in symbol_path.split('.')[1:]:
        if not hasattr(obj, el):
            return None
        obj = getattr(obj, el)
    return obj


def file_symbol_path(fn: Callable) -> tuple[str | None, str | None]:
    """The file and symbol path of a function defined in a file we loaded earlier, or (None, None)."""
    file = get_loaded_file_path(fn)
    if file is None:
        return None, None
    return file, f'{_file_stem(Path(file))}.{fn.__qualname__}'


def _module_name(path: Path) -> str:
    """The name _load() gives the module it loads from path, derived from the resolved path."""
    path = path.resolve()
    return f'pxt_app_{_file_stem(path)}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}'


def load_file(file: str | Path, *, reload: bool = False) -> ModuleType:
    """Load the module in file, under a name derived from its path.

    Returns the module already loaded from that path, unless reload is True, which discards what was loaded
    from the file's directory and executes the file again.

    Imports in the file resolve against installed modules first and against the file's own directory second,
    so a file can import its neighbors without its directory being on sys.path.
    """
    path = Path(file).resolve()
    with _lock:
        if reload:
            _evict(path.parent)
        module = _loaded_modules.get(path)
        if module is not None:
            return module
        return _load(path)


def get_loaded_file_path(fn: Callable) -> str | None:
    """The path of fn's file if load_file() loaded it, and None for a function from anywhere else."""
    if len(_loaded_modules) == 0:
        return None  # _load() records a file before executing it, so nothing loaded means fn is from elsewhere
    try:
        path = Path(inspect.getfile(fn)).resolve()
    except TypeError:
        return None  # a builtin has no file
    with _lock:
        return str(path) if path in _loaded_modules else None


def _evict(dir: Path) -> None:
    """Discard every module loaded from dir or from a directory under it, such as a package next to a file."""
    for path in [p for p in _loaded_modules if p.is_relative_to(dir)]:
        # a submodule reached by a dotted import was registered under its package's name by the standard
        # machinery, so it is absent from _loaded_modules and would serve its old code after the reload
        _pop_module_tree(_module_name(path))
        del _loaded_modules[path]
    for module_dir in [d for d in _dir_modules if d.is_relative_to(dir)]:
        _pop_module_tree(_dir_modules[module_dir].__name__)
        del _dir_modules[module_dir]


def _pop_module_tree(name: str) -> None:
    """Remove the module called name from sys.modules, along with every module below it."""
    for loaded_name in [n for n in sys.modules if n == name or n.startswith(f'{name}.')]:
        del sys.modules[loaded_name]


def _load(path: Path, *, is_package: bool = False) -> ModuleType:
    """Execute the file at path, register the module it defines in _loaded_modules, and return the module.

    Callers hold _lock and pass a resolved path, which becomes its key in _loaded_modules. is_package loads
    the file as a package's __init__.py, so that imports of its submodules resolve. A file that fails to
    execute is not recorded.
    """
    name = _module_name(path)
    search_locations = [str(path.parent)] if is_package else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=search_locations)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {path}', path=str(path))
    module = importlib.util.module_from_spec(spec)
    # exec_module() supplies __builtins__ only when the module has none, so presetting it is what routes the
    # imports of the file, and of the functions it defines, through _make_import()
    module.__dict__['__builtins__'] = {**vars(builtins), '__import__': _make_import(path.parent)}
    sys.modules[name] = module
    # registered before exec_module(), because file_of() has to answer for this path while the file runs
    _loaded_modules[path] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del _loaded_modules[path]
        sys.modules.pop(name, None)
        raise
    return module


def _make_import(dir: Path) -> Callable[..., ModuleType]:
    """An __import__() for a file in dir, which resolves a name against dir when nothing installed provides it."""

    def _import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if level == 0:
            top = name.split('.', maxsplit=1)[0]
            try:
                return builtins.__import__(name, globals, locals, fromlist, level)
            except ModuleNotFoundError as e:
                if e.name != top:
                    raise  # something the named module imports is missing, not the module itself
                sibling = _load_from_dir(dir, top)
                if sibling is None:
                    raise
                return _submodule(sibling, name[len(top) + 1 :], fromlist)
        if level > 1:
            raise ImportError(f'{dir}: an import above the directory of a loaded file is not supported', name=name)

        if name == '':
            # a relative 'from . import' names its modules in fromlist and reads them off a package
            return _dir_module(dir, fromlist or ())
        neighbor = _load_from_dir(dir, name.split('.', maxsplit=1)[0])
        if neighbor is None:
            raise ImportError(f'{dir} holds no module named {name!r}', name=name)
        return _submodule(neighbor, name.split('.', 1)[1] if '.' in name else '', fromlist)

    return _import


def _load_from_dir(dir: Path, name: str) -> ModuleType | None:
    """The module or package named by name in dir, loaded if it is not loaded already."""
    with _lock:
        file_path = dir / f'{name}.py'
        if file_path.is_file():
            path = file_path.resolve()
            return _loaded_modules.get(path) or _load(path)
        init_path = dir / name / '__init__.py'
        if init_path.is_file():
            path = init_path.resolve()
            return _loaded_modules.get(path) or _load(path, is_package=True)
        return None


def _submodule(module: ModuleType, submodule_path: str, fromlist: tuple[str, ...] | None) -> ModuleType:
    """The module that an import of submodule_path under module denotes.

    A from-import denotes the submodule, a plain import the module the statement binds, which is what
    __import__() returns for each form. An empty submodule_path denotes module itself.
    """
    if submodule_path == '':
        for member in fromlist or ():
            if not hasattr(module, member):
                # a fromlist member can name a module of the package, which __import__() imports on demand;
                # the import statement fetches anything else as an attribute
                try:
                    _load_submodule(module, member)
                except ImportError:
                    pass
        return module
    submodule = _load_submodule(module, submodule_path)
    return submodule if fromlist else module


def _load_submodule(package: ModuleType, submodule_path: str) -> ModuleType:
    """Load the module that submodule_path names under package, and attach each step to its parent.

    Loaded here rather than by the standard machinery, so that a decorator running in the module sees its own
    file in _loaded_modules: a udf whose file is unknown is taken to be importable by name, and the name of a
    module loaded from a path is importable in no other process.
    """
    parent = package
    for name in submodule_path.split('.'):
        search_locations = getattr(parent, '__path__', None)
        if search_locations is None:
            raise ImportError(f'{parent.__name__} is not a package', name=name)
        submodule = _load_from_dir(Path(search_locations[0]), name)
        if submodule is None:
            raise ImportError(f'{search_locations[0]} holds no module named {name!r}', name=name)
        sys.modules[f'{parent.__name__}.{name}'] = submodule
        setattr(parent, name, submodule)
        parent = submodule
    return parent


def _dir_module(dir: Path, fromlist: tuple[str, ...]) -> ModuleType:
    """Create a module that represents dir as a package, and attach the modules named in fromlist to it.

    A relative import such as 'from . import helpers' fetches helpers as an attribute of the importing file's
    package, and this module is that package. Only the first call for a directory creates it; a later call
    attaches to the module already created and returns that.
    """
    with _lock:
        module = _dir_modules.get(dir)
        if module is None:
            name = f'pxt_dir_{hashlib.sha256(str(dir).encode()).hexdigest()[:8]}'
            module = ModuleType(name)
            module.__path__ = [str(dir)]
            sys.modules[name] = module
            _dir_modules[dir] = module
        for member in fromlist:
            if not hasattr(module, member):
                neighbor = _load_from_dir(dir, member)
                if neighbor is not None:
                    setattr(module, member, neighbor)
        return module
