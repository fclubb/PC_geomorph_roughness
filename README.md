# Point Cloud (PC) Geomorphologic roughness and topographic detrending
Detrending Point Cloud (PC) data with slope and calculating topographic roughness and curvature from PCs.

The code reads in a ground-classified PC from a LAS/LAZ file and calculates several geomorphology-relevant metrics on the PC. Input files can be from lidar or SfM PC, but should be ground-classified. The algorithm allows defining a radius which is used to fit a linear plane through the point cloud to detrend the data (i.e., normalize the point cloud with mean elevation of 0). These data are used to calculate deviations from the mean (roughness) and identify rills, arroyos, incised canyons, and other forms of erosion processes. By varying the radius over which the plane is fitted, several scales of the landscape can be analyzed (similar to varying radii of topographic relief).  The algorithm choses seed points from the PC with a user-defined spacing (for example 1m) and calculated statistics for each seed point with a given radius (for example 2m).

Output includes a set of shapefile and geotiffs that show statistics of the PC within the given radius. Also, CSV and H5 files are created that contain lists of seed point location and statistical results for further analysis in Python or Matlab.


# Installation
This is a Python 3.5 code that will run on any OS, which supports the packages. It runs and has been tested on Linux (Ubuntu/Debian), Windows 10, and Mac OS X. You will need several packages for python to run this code. These are standard packages and are included in many distributionss. If you use [conda](https://conda.io/docs/index.html), you can install the required packages (using Python 3.5):
```
conda create -n py35 python=3.5 scipy pandas numpy matplotlib gdal scikit-image gdal ipython python=3.5 spyder h5py
```

You can active this environment with ```source activate py35```.

You don't need ipython or spyder to run this code and you can remove these repositories in the command line above, but they usually come in handy. Also, if you plan to store and process large PC datasets, it may come in handy storing the processed data in compressed HDF5 or H5 files. However, for some installations, the above installed h5py does not contain the gzip compression (or any compression). Update installation with:
```
source activate py35
conda install -y -c conda-forge h5py
```

In order to read and write zipped LAS files (LAZ) files, install lastools. These will come in handy. Note that if you have installed pdal, you usually don't need this:
```
source activate py35
conda install -y -c conda-forge lastools
```

Install a fast and simple LAS/LAZ reader/writer. You can do similar steps through lastools, but this interface is fairly simple to use:
```
source activate py35
pip install laspy
```

For some color rescaling (e.g., writing points to LAS files with new colors), scikit-image comes in handy:
```
source activate py35
conda install -y scikit-image
```

This code uses [scipy.spatial.cKDTree](https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.spatial.cKDTree.html). The KDTree search could be made much faster with [pykdtree](https://github.com/storpipfugl/pykdtree). However, pyKDTree doesn't allow to easily save the tree - so, if you intend to run this repeatedly iterating through various search radii, the cKDTree implementation is usefull. For very large point clouds, the pyKDTree algorithm is significantly faster for generating and querying the KDtree and will increase processing speed. To install pyKDTree:
```
source activate py35
conda install -c conda-forge pykdtree

```

Last, install the repository into your favorite github directory, for example ~/github:
```
cd ~
git clone https://github.com/UP-RS-ESP/PC_geomorph_roughness

```
You are now ready to run the code from the command line.


# Command line parameters
The code can be run from the command line with
```
python pc_geomorph_roughness.py

```

Parameters to be chosen include (can also be listed with '''python pc_geomorph_roughness.py -h''':
## Required Parameters
+ -i or --inlas: LAS/LAZ file with point-cloud data. Ideally, this file contains only ground points (class == 2)
+ -r_m or --raster_m: Raster spacing for subsampling seed points on LAS/LAZ PC. Usually 0.5 to 2 m, default = 1.
+ -srd_m or --sphere_radius_m: Radius of sphere used for selecting lidar points around seed points. These points are used for range, roughness, and density calculations. Default radius 1.5m, i.e., points within a sphere of 3m are chosen.
+ -slope_srd_m or --slope_sphere_radius_m: Radius of sphere used for fitting a linear plane and calculating slope and detrending data (slope normalization). By default this is similar to the radius used for calculation roughness indices (srd_m), but this can be set to a different value. For example, larger radii use the slope of larger area to detrend data.
+ -shp_clp or --shapefile_clip: Name / path of shapefile to clip interpolated output. Because the gridding procedure may introduce artefacts at the boundaries, a shapefile will be used to clip data to its original area. This is likely the same shapefile that has been used to generate the las data in the first place.
## Suggested Parameters
+ -epsg or --epsg_code: EPSG code (integer) to define projection information. This should be the same EPSG code as the input data (no re-projection included yet) and can be taken from LAS/LAZ input file. Add this to ensure that output shapefile and GeoTIFFs are properly geocoded.

## Additional Parameters
+ -o or --outlas: LAS file to be created (currently only LAS files are supported). This has the same dimension and number of points as the input LAS/LAZ file, but replaced color values reflecting roughness calculated over a given radius. Note that this will replace existing color information in the output file.
+ -shape_out or --shapefile_out: Output shapefile storing calculated attributes for seed points only. Default filename will be generated with radius in the filename.
+ -odir --outputdir: Output directory to store plots and pickle files. Default is directory containing LAS/LAZ file.
+ -fig or --figure: Generate figures while processing. This often takes significant amount of time and can be turned off with -fig False.
+ -color or --store_color_las: Generate a LAS file where deviation from the plane are saved in the color attribute of the LAS file for every point. *Note* that this will replace the color information in the LAS file (but will be saved to separate file). Default is True, can be turned off with --store_color False.


# Examples

In order to generate a 1-m raster file which contains PC information from a radius of 2 m for each 1-m step (seed points, default parameters), you can use:

```
python pc_geomorph_roughness.py -i Pozo_USGS_UTM11_NAD83_all_color_cl2.las -r_m 1 -srd_m 2
```

Additional examples with generated output are included in [examples](examples/README.md)

