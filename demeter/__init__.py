# Deepnote's preinstalled copy of geopandas doesn't come with PyGEOS or Rtree,
# so spatial indexing doesn't work. It decides which library to use at import
# time, and since geopandas is somehow baked into Deepnote runtime it insists
# PyGEOS or Rtree are missing even after they're installed.
#
# HACK: "unimport" geopandas to force it to reload and find PyGEOS or Rtree.
import os

if "DEEPNOTE_PROJECT_ID" in os.environ:
    import sys

    geopandas_module_names = [
        name for name in sys.modules if name.startswith("geopandas")
    ]
    for name in geopandas_module_names:
        del sys.modules[name]
