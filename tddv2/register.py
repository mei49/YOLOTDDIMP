"""Make TDD-YOLO modules usable from yaml with stock pip ultralytics (no fork).

ultralytics.nn.tasks.parse_model resolves yaml module names with globals() and only knows
how to compute channels for its own classes. We re-compile parse_model from its source with
two edits and install it back into ultralytics.nn.tasks:

  1. channel-changing classes are added to `base_modules`  -> called as m(c1, c2_scaled, *args)
  2. a new branch for channel-preserving classes             -> called as m(ch[f], *args), c2 = ch[f]

If either regex fails to match (ultralytics changed parse_model), we raise instead of
silently building a wrong model. Tested with the ultralytics version pinned in requirements.txt.
"""

import inspect
import re
import textwrap

import ultralytics
import ultralytics.nn.tasks as tasks

from .modules import CAFE2, CSAF2, DySample
from .orig.aver_fem import Averge_FEM, BackAverge_FEM
from .orig.cbam_cor import CBAM_cor
from .orig.cor_fem import co_FEM

TESTED_ULTRALYTICS = "8.4.213"

# m(c1, c2, *args): c2 = make_divisible(min(args[0], max_channels) * width, 8)
CHANNEL_CHANGING = (co_FEM, Averge_FEM, BackAverge_FEM, CAFE2)
# m(ch[f], *args): output channels == input channels
CHANNEL_PRESERVING = (CBAM_cor, CSAF2, DySample)

_BASE_RE = re.compile(r"base_modules\s*=\s*frozenset\(\s*\{")
_AIFI_RE = re.compile(r"^(?P<indent>[ \t]+)elif m is AIFI:", re.M)

_patched = False


def register(verbose=True):
    """Patch ultralytics.nn.tasks.parse_model once. Safe to call repeatedly."""
    global _patched
    if _patched:
        return
    if ultralytics.__version__ != TESTED_ULTRALYTICS and verbose:
        print(f"tddv2.register: WARNING ultralytics {ultralytics.__version__} != tested {TESTED_ULTRALYTICS}")

    src = textwrap.dedent(inspect.getsource(tasks.parse_model))

    if len(_BASE_RE.findall(src)) != 1:
        raise RuntimeError(
            "tddv2.register: could not find `base_modules = frozenset({` in ultralytics parse_model "
            f"(ultralytics {ultralytics.__version__}). Install the version pinned in requirements.txt."
        )
    src = _BASE_RE.sub("base_modules = frozenset({*_TDD_CHANNEL_CHANGING, ", src, count=1)

    m = _AIFI_RE.search(src)
    if m is None:
        raise RuntimeError(
            "tddv2.register: could not find `elif m is AIFI:` in ultralytics parse_model "
            f"(ultralytics {ultralytics.__version__}). Install the version pinned in requirements.txt."
        )
    ind = m.group("indent")
    branch = f"{ind}elif m in _TDD_CHANNEL_PRESERVING:\n{ind}    c2 = ch[f]\n{ind}    args = [c2, *args]\n"
    src = src[: m.start()] + branch + src[m.start() :]

    ns = tasks.__dict__
    for cls in CHANNEL_CHANGING + CHANNEL_PRESERVING:
        ns[cls.__name__] = cls
    ns["_TDD_CHANNEL_CHANGING"] = frozenset(CHANNEL_CHANGING)
    ns["_TDD_CHANNEL_PRESERVING"] = frozenset(CHANNEL_PRESERVING)

    code = compile(src, inspect.getsourcefile(tasks.parse_model), "exec")
    original = tasks.parse_model
    exec(code, ns)  # redefines tasks.parse_model in the tasks module namespace
    if tasks.parse_model is original:
        raise RuntimeError("tddv2.register: parse_model was not replaced")
    tasks.parse_model._tddv2 = True
    _patched = True
    if verbose:
        names = ", ".join(c.__name__ for c in CHANNEL_CHANGING + CHANNEL_PRESERVING)
        print(f"tddv2.register: patched ultralytics {ultralytics.__version__} parse_model ({names})")
