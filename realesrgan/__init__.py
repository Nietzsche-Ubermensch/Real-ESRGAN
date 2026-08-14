# flake8: noqa
try:
    from .version import *
except Exception:
    __version__ = 'unknown'

try:
    from .archs import *
    from .data import *
    from .models import *
    from .utils import *
except Exception:
    # Allow lightweight submodule imports (e.g., batch_processing) when optional
    # training/inference dependencies are unavailable in the runtime environment.
    pass
