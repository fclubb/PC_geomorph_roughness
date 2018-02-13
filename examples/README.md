# Example - Run and output from a small catchment on Santa Cruz Island (California, subcatchment Pozo)
## Pre-processing
Input LAZ file is called _Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz_. This file has been previously clipped with SC12.shp from the USGS Lidar PC dataset. You can use lasclip.exe to clip the data: 
```
wine /opt/LAStools/bin/lasclip.exe -keep_class 2 -i /raid/data/SCI/Pozo/Pozo_USGS_UTM11_NAD83_all_color_cl.laz -poly SC12.shp -olaz -o Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz
```

Obtaining information about the LAZ files with
```
wine /opt/LAStools/bin/lasinfo.exe -i Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz
```

If you don't have a DEM, use las2dem or other means to to generate a DEM. This is useful if you want to match outputs to an existing grid. The code will take the coordinates from the geotiff file:
```
wine /opt/LAStools/bin/las2dem.exe -keep_class 2 -cores 8 -utm 11N -nad83 -meter -elevation_meter -elevation -step 1 -i Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz -o Blanca_in_Pozo_USGS_UTM11_NAD83_SC12_1m.tif
```

Using gdal, you can calculate statistics and generate a hillshade file:
```
gdalinfo -hist -stats -mm Blanca_in_Pozo_USGS_UTM11_NAD83_SC12_1m.tif
gdaldem hillshade Blanca_in_Pozo_USGS_UTM11_NAD83_SC12_1m.tif Blanca_in_Pozo_USGS_UTM11_NAD83_SC12_1m_HS.tif
```

## Calculating neighborhood statistics from the point cloud
Make sure to start the environment containg all necessary modules, e.g. through
```
source activate py35
```

On the input LAZ, we will run the python code with (make sure to use the proper directory where your python code _pc_geomorph_roughness.py_ is stored or set the PATH accordingly). The command assumes that you are in the directory containing the LAZ file. It will take about ~5 minutes to run on a file with ~500k points, including 10 subsampling iterations for uncertainty estimation (bootstrapping):
```
python /home/bodo/Dropbox/soft/github/PC_geomorph_roughness/pc_geomorph_roughness.py \
--inlas Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz \
--dem_fname Blanca_in_Pozo_USGS_UTM11_NAD83_SC12_1m.tif \
--raster_m 1 \
--sphere_radius_m 1.5 \
--slope_sphere_radius_m 1.5 \
--nr_random_sampling 10 \
--epsg_code 26911 \
--shapefile_clip SC12.shp \
--nr_random_sampling 10 \
--raw_pt_cloud 1
```
Make sure to set the EPSG Code (in this case NAD83, UTM Zone 11) so that output shapefiles and geotiff files have the proper geographic coordinates. Explore the outputs in the directories _figures_ and _geotif_ and look at the generated LAS file, for example with ```displaz Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_raster_1.00m_rsphere1.50m.las```. [displaz](https://github.com/c42f/displaz) is a simple, but fast and versatile lidar viewer running on most OS.
Currently, the raw point cloud analysis (--raw_pt_cloud) is turned off.

This example directory contains the output from this run. 

### The following data (.h5, .vrt, .csv) output files are created:
- _*_iter10_mean_seed_pts_stats_raster_1.00m_radius_1.50m.*_ (*_iter10_mean_seed_pts_stats_raster_1.00m_radius_1.50m.csv and *_iter10_mean_seed_pts_stats_raster_1.00m_radius_1.50m.h5), for example Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_iter10_mean_seed_pts_stats_raster_1.00m_radius_1.50m.csv. Contains the average of the iterations (in this case 10) for various topographic metrics calculated from the homogenous and subsampled point cloud. Spacing is adjusted to input DEM (if filename is given --dem_fname). A .vrt file is created to allow conversion to geotif, shapefiles, or other formats.

- _*_raw_seed_pts_stats_raster_1.00m_radius_1.50m.*_ (*_raw_seed_pts_stats_raster_1.00m_radius_1.50m.csv and *_raw_seed_pts_stats_raster_1.00m_radius_1.50m.h5), for example Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_raw_seed_pts_stats_raster_1.00m_radius_1.50m.csv. Contains the point-cloud statistics from the raw point cloud (no subsampling). Spacing is adjusted to input DEM (if filename is given --dem_fname). A .vrt file is created to allow conversion to shapefiles or other formats.

- _PC_density_stats_n0442_r1.50m.h5_ are the density statistics (number of neighboring points) for the point cloud with a given radius (in this case r=1.5 for n=442 neighboring points)

- _PC_equal_density_n0442_r1.50m.h5_ is the output of a homogenous, subsampled pointcloud based on the density statistics from _PC_density_stats_n0442_r1.50m.h5_

- _PC_bootstrap_iter*.h5_ is the output of the individual iterations of the homogenous, subsampled point cloud (similar to _PC_equal_density_n0442_r1.50m.h5_ but for each iteration)

- _PC_bootstrap_stats_all_iter10_r1.50m.h5_ contains the statistical results (outputs) from each iteration for all seed points (no point cloud is saved, these are in the individual _PC_bootstrap_iter*.h5_ files)

You can convert the *.vrt files with the following gdal command to a geotif file (example):
```
gdal_grid -zfield "18Nr_lidar" -outsize 271 281 -a linear:radius=2.0:nodata=-9999 -of GTiff -ot Int16 Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_iter10_mean_seed_pts_stats_raster_1.00m_radius_1.50m.vrt Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_iter10_mean_seed_pts_stats_raster_1.00m_radius_1.50m_nrlidar.tif --config GDAL_NUM_THREADS ALL_CPUS -co COMPRESS=DEFLATE -co ZLEVEL=7
```

### The following overview figures files are created in the subdirectory _figures_:
- _*_bootstrap_iter10_seed_pts_overview1_raster_1.00m_radius_1.50m.png_ containing six subfigures showing the seed points and their values from the homogenous, subsampled point cloud (_bootstrap_iter10_) for topography, nr. of lidar measurements, point-cloud slope, point-cloud curvature, surface roughness, and point-cloud intensity.
- _*_bootstrap_iter10_seed_pts_overview2_raster_1.00m_radius_1.50m.png_ same as above: containing subfigures for the intensity IQR range, point-cloud density, and IQR range of surface roughness
- _*_raw_seed_pts_overview1_raster_1.00m_radius_1.50m.png_ containing six subfigures showing the seed points and their values from 
- _*_raw_seed_pts_overview2_raster_1.00m_radius_1.50m.png_ same as above: containing subfigures for the intensity IQR range, point-cloud density, and IQR range of surface roughness
- several point-cloud views of individual parameters deduced from the raw point cloud (curvature, slope, surface roughness, standard deviation of points from a fitted plane

### The following geotif files are created in the subdirectory _geotif_:
There are two groups of geotif files created: one directly from the entire raw pointcloud (_raw_*.tif_) and one from the average of all iterations (_iter10_*.tif_). There are several topographic parameters stored for each seed points with the given radius using the following scheme: 
- slope of fitted plane (_planeslope.tif_), curvature (_curv.tif_), standard deviation of points (_intensity_ieq.tif_)
- point-cloud density (_density.tif_), nr of lidar points (_nrlidar.tif_)
- distance from a plane IQR range (_dz_iqr.tif_), maximum distance from a plane (_dzmax.tif_), minimum distance from a plane (_dzmin.tif_), range of 10th and 90th percentile distance from a plane (_dz_range9010.tif_)
- intensity IQR (_intensity_iqr.tif_), intensity mean (_intensity_mean.tif_)


**Note that the directories _log_ and _pickle_ are not include, also the _*.h5_ files are not included in these examples and on github because of their file sizes**


Subsequent runs will be faster, because there is no need to sort through the point cloud and generate the KDTree again. Experiment with different radii for the slope normalization and roughness calculation. Note that output filesnames will be automatically adjusted:
