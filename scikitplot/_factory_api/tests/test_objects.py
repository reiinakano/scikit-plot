from .. import objects

from ..._core.plot import Plot
from ..._core.moves import Move
from ..._core.scales import Scale
from ..._marks.base import Mark
from ...stats.base import Stat


def test_objects_namespace():

    for name in dir(objects):
        if not name.startswith("__"):
            obj = getattr(objects, name)
            assert issubclass(obj, (Plot, Mark, Stat, Move, Scale))
