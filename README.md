# FlatOut Blender Tools
A collection of Blender Add-Ons &amp; Python scripts for various formats from the FlatOut game series.  

Currently a **WIP**, please report all bugs/discoveries in issues. 

## Blender Plugins/Addons
* **fo2_bgm_import:** Import car _body.bgm_ & additional files (_crash.dat_, _body.ini_, _camera.ini_)
* **fo2_bgm_export:** **WIP!** Export car _body.bgm_ & additional files (_crash.dat_, _body.ini_, _camera.ini_)
* **fo2_bgm_hierarchy:** **WIP!** Reorganize a blender scene made with other converters into the expected fo2_bgm_export structure
* **fo2_trackai_import:** Import _trackai.bin_ & additional files (_splines.ai_, _startpoints.bed_, _splitpoints.bed_)
* **fo2_trackai_export:** **WIP!** Export _trackai.bin_ & additional files (_splines.ai_, _startpoints.bed_, _splitpoints.bed_)
* **fo2_collision_io:** **WIP!** Import/Export collision data _(.cdb2)_ & _shadowmap_w2.dat_

## Standalone Scripts
* **dds2tga:** Convert .dds to .tga 32-bit, preserve alpha channel
* **tga2dds:** Convert .tga to .dds (DXT1, DXT3, DXT5)
* **pctops2_bgm:** Convert PC .bgm to PS2 .bgm _(experimental)_
* **bgm_optimize:** Optimize BGM Files for FlatOut 2 PC _(experimental)_

## Credits &amp; Notes
- [Chloe](https://github.com/gaycoderprincess) for her work on the model format ([FlatOutW32BGMTool](https://github.com/gaycoderprincess/FlatOutW32BGMTool))
- [mrwonko](https://github.com/mrwonko/) for his work on the collision (cdb2) format ([flatout-open-level-editor](https://github.com/mrwonko/flatout-open-level-editor))
