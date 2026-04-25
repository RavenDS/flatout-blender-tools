# FlatOut Blender Tools
A collection of Blender Add-Ons &amp; Python scripts for **FlatOut 1/2/Ultimate Carnage** on PC/PS2/XBOX/PSP.  

## Blender Plugins/Addons
* **fo2_bgm_import:** Import **FO1/FO2/FOUC** car _body.bgm_ + additional files (_crash.dat_, _body.ini_, _camera.ini_)
* **fo2_bgm_export:** Export **FO1/FO2/FOUC** car _body.bgm_ + additional files (_crash.dat_, _body.ini_, _camera.ini_)
* **fo2_bgm_hierarchy:** Reorganize a blender scene into the fo2_bgm_export structure
* **fo2_trackai_import:** Import FO2 _trackai.bin_ + additional files (_splines.ai_, _startpoints.bed_, _splitpoints.bed_)
* **fo2_trackai_export:** Export FO2 _trackai.bin_ + additional files (_splines.ai_, _startpoints.bed_, _splitpoints.bed_)
* **fo2_collision_io:** **WIP!** Import/Export collision data _(.cdb2)_ & _shadowmap_w2.dat_

## Standalone Scripts
* **[bgm_tool](scripts/bgm_tool.py):** Convert + Optimize **FO1/FO2/UC** .bgm files for **PC/PS2/XBOX/PSP**
* **[dds2tga](scripts/dds2tga.py):** Convert .dds to .tga 32-bit, preserve alpha channel
* **[dds_normal](scripts/dds_normal.py)**: Convert **FOUC RXGB** (DXT5nm) to RGBA
* **[dds_resize](scripts/dds_resize.py)**: Quick resize .dds files for console ports
* **[tga2dds](scripts/tga2dds.py):** Convert .tga to .dds (DXT1, DXT3, DXT5)
* **[png2dds](scripts/png2dds.py):** Convert .png to .dds (DXT1, DXT3, DXT5)

## Standalone Tools
* **[PS2TexTool](tools/PS2TexTool.exe):** GUI to easily import **.dds** into PS2 **.tm2** and PSP **.tex** skin files

## Legacy Tools
_These tools are provided for convenience and won't be updated_
* **[FlatOutFSBtool](tools/FlatOutFSBTool.exe):** Convert audio to FlatOut 2 .FSB format (PS2)
* **[bfstool-gui](tools/bfstool-gui.exe):** GUI for [bfstool](https://github.com/xNyaDev/bfstool)


## Credits &amp; Notes
- [Chloe](https://github.com/gaycoderprincess) for her work on the model (bgm) format ([FlatOutW32BGMTool](https://github.com/gaycoderprincess/FlatOutW32BGMTool))
- [mrwonko](https://github.com/mrwonko/) for his work on the collision (cdb2) format ([flatout-open-level-editor](https://github.com/mrwonko/flatout-open-level-editor))
