.. _calwebb_spec3:

calwebb_spec3: Stage 3 Spectroscopic Processing
===============================================

:Config: calwebb_spec3.cfg
:Class: `~jwst.pipeline.Spec3Pipeline`

Stage 3 processing for spectroscopic observations is intended for combining the 
calibrated data from multiple exposures (e.g. a dither/nod pattern) into a single
combined 2D or 3D spectral product and a combined 1D spectrum.
Before being combined, the exposures may receive additional corrections for the
purpose of background matching and subtraction, as well as outlier rejection.
The steps applied by the ``calwebb_spec3`` pipeline are shown below.
This pipeline is intended for non-TSO spectra only. TSO spectral data should be
processed using the :ref:`calwebb_tso3 <calwebb_tso3>` pipeline.

.. |c| unicode:: U+2713 .. checkmark

+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+
| Instrument/Mode                                   |     NIRSpec     |    MIRI   | NIRISS | NIRCam |
+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+
| Step                                              | FS  | MOS | IFU | FS  | MRS | WFSS   | WFSS   |
+===================================================+=====+=====+=====+=====+=====+========+========+
| :ref:`master_background <master_background_step>` | |c| | |c| | |c| | |c| | |c| |        |        |
+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+
| :ref:`exp_to_source <exp_to_source>`              | |c| | |c| |     |     |     |  |c|   |  |c|   |
+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+
| :ref:`mrs_imatch <mrs_imatch_step>`               |     |     |     |     | |c| |        |        |
+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+
| :ref:`outlier_detection <outlier_detection_step>` | |c| | |c| | |c| | |c| | |c| |        |        |
+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+
| :ref:`resample_spec <resample_step>`              | |c| | |c| |     | |c| |     |        |        |
+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+
| :ref:`cube_build <cube_build_step>`               |     |     | |c| |     | |c| |        |        |
+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+
| :ref:`extract_1d <extract_1d_step>`               | |c| | |c| | |c| | |c| | |c| |  |c|   |  |c|   |
+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+
| :ref:`combine_1d <combine_1d_step>`               |     |     |     |     |     |  |c|   |  |c|   |
+---------------------------------------------------+-----+-----+-----+-----+-----+--------+--------+

Arguments
---------

The ``calwebb_spec3`` pipeline does not have any optional arguments.

Inputs
------

2D calibrated data
^^^^^^^^^^^^^^^^^^

:Data model: `~jwst.datamodels.ImageModel`, `~jwst.datamodels.IFUImageModel`,
             `~jwst.datamodels.SlitModel`, or `~jwst.datamodels.MultiSlitModel`
:File suffix: _cal

The inputs to ``calwebb_spec3`` should be in the form of an ASN file that
lists the multiple exposures to be processed into combined output products.
The individual exposures should be calibrated the ("_cal") products from
``calwebb_spec2`` processing.

The member list for each product in the ASN file can also contain exposures
of dedicated background targets, that are intended for use in the
:ref:`master_background <master_background_step>` step. These input exposures
must be the "x1d" products (extracted 1-D spectra) of the background target(s)
and are usually the "x1d" files produced by the
:ref:`calwebb_spec2 <calwebb_spec2>` pipeline. They must be listed in the ASN
file with "exptype" values of "background" in order to be correctly identified
as background exposures. See the :ref:`master_background <master_background_step>`
for more details.

Outputs
-------

Source-based calibrated data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Data model: `~jwst.datamodels.MultiExposureModel`
:File suffix: _cal

For NIRSpec fixed-slit, NIRSpec MOS, and NIRCam and NIRISS WFSS, which have a defined
set of slits or sources, the data from the input calibrated exposures is reorganized
by :ref:`exp_to_source <exp_to_source>` so that all of the instances of data for a
particular source/slit are contained in a
single product. These are referred to as "source-based" products, as opposed to the
input exposure-based products. The source-based collections of data are saved in
intermediate files, one per source/slit. The root names of the source-based files
contain the source ID as an identifier and use the same "_cal" suffix as the input
calibrated exposure files. An example source-based file name is
"jw00042-o001_s0002_niriss_gr150r_f150w_cal.fits", where "s0002" is the source id.

The reorganized sets of data are sent to subsequent steps to process and combine
all the data for one source at a time.

CR-flagged exposures
^^^^^^^^^^^^^^^^^^^^

:Data model: `~jwst.datamodels.ImageModel`
:File suffix: _crf

If the :ref:`outlier_detection <outlier_detection_step>` step is applied, a new version of
each input calibrated exposure is created, in which the DQ array has been updated to
flag pixels detected as outliers. These files use the "_crf" (CR-Flagged)
product type suffix and also includes the association candidate ID as a
new field in the original product root name, e.g.
"jw96090001001_03101_00001_nrs2_o001_crf.fits."


2D resampled and combined spectral data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Data model: `~jwst.datamodels.DrizProductModel`
:File suffix: _s2d

When processing non-IFU modes, a resampled/rectified 2D product of type
"_s2d" is created containing the rectified and combined data for a given
slit/source, which is the output of the :ref:`resample_spec <resample_step>` step.

3D resampled and combined spectral data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Data model: `~jwst.datamodels.IFUCubeModel`
:File suffix: _s3d

When processing IFU exposures, a resampled/rectified 3D IFU cube product
created by the :ref:`cube_build <cube_build_step>` step is saved as an "_s3d" file.

1D extracted spectral data
^^^^^^^^^^^^^^^^^^^^^^^^^^

:Data model: `~jwst.datamodels.MultiSpecModel`
:File suffix: _x1d

All types of inputs result in a 1D extracted spectral data product, which is
saved as a "_x1d" file, which is normally the result of performing the
:ref:`extract_1d <extract_1d_step>` step on the combined "_s2d" or "_s3d" product.
For NIRCam and NIRISS WFSS data, the :ref:`extract_1d <extract_1d_step>` is
performed on the individual unresampled 2D cutout images, and then the
:ref:`combine_1d <combine_1d_step>` is used to combine those spectra into a
final combined "x1d" product for each source.
