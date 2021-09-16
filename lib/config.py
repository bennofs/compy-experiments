import pathlib

import appdirs


def get_project_dir() -> pathlib.Path:
    """Return the path to the root directory of this project (the path which contains the lib/ and bin/ folder)."""
    return pathlib.Path(__file__).resolve().parent.parent


def get_tracing_helpers_path() -> pathlib.Path:
    return get_project_dir() / "bin" / "tracing-helpers"


def get_cache_dir(subdir=None) -> pathlib.Path:
    path = pathlib.Path(appdirs.user_cache_dir('compy-experiments'))
    if subdir is not None:
        path = path / subdir
    path.mkdir(exist_ok=True, parents=True)
    return path


def is_ipython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True