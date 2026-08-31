# FlatOut 2 Track (.w32) import/export add-on.
# Bundles the W32 importer, exporter, and the TGA/DDS texture converters
# (tga2dds, dds2tga) so texture handling works out of the box.

bl_info = {
    "name": "FlatOut 2 Track (W32)",
    "author": "ravenDS",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "File > Import/Export > FlatOut 2 Track (.w32)",
    "description": "Import and export FlatOut 2 track geometry (.w32) and "
                   "companion files (BVH, plants, vertex colors)",
    "category": "Import-Export",
}

from . import io_import_fo2_w32 as _w32_import
from . import io_export_fo2_w32 as _w32_export


def register():
    _w32_import.register()
    _w32_export.register()


def unregister():
    _w32_export.unregister()
    _w32_import.unregister()
