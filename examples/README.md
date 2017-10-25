# Example
## Run and output from a small catchment on Santa Cruz Island (California, subcatchment Pozo)
Input LAZ file is called _Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz_. This file has been previously clipped with SC12.shp from the USGS Lidar PC dataset. You can use lasclip.exe to clip the data: 
```
wine /opt/LAStools/bin/lasclip.exe -keep_class 2 -i /raid/data/SCI/Pozo/Pozo_USGS_UTM11_NAD83_all_color_cl.laz -poly SC12.shp -olaz -o Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz
```

Make sure to start the environment containg all necessary modules, e.g. through
```
source activate py35
```

On the input LAZ, we will run the python code with (make sure to use the proper directory where your python code pc_geomorph_roughness.py is stored or set the PATH accordingly). The command assumes that you are in the directory containing the LAZ file. It will take about ~1 minute to run on a file with ~1M points:
```
python /home/bodo/Dropbox/soft/github/PC_geomorph_roughness/pc_geomorph_roughness.py -i Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz -r_m 1 -srd_m 2 --epsg_code 26911 -shp_clp SC12.shp --store_color True
```
Make sure to set the EPSG Code (in this case NAD83, UTM Zone 11) so that output shapefiles and geotiff files have the proper geographic coordinates. Explore the outputs in the directories _figures_ and _geotif_ and look at the generated LAS file, for example with ```displaz Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_raster_1.00m_rsphere2.00m.las```. [displaz](https://github.com/c42f/displaz) is a simple, but fast and versatile lidar viewer running on most OS.


This example directory contains the output from this run. You can perform additional runs with other radii using the example commands below.

**Note that the directories _log_ and _pickle_ are not include, also the _*.h5_ files are not included in these examples and on github because of their file sizes**

Subsequent runs will be faster, because there is no need to sort through the point cloud and generate the KDTree again. Experiment with different radii for the slope normalization and roughness calculation. Note that output filesnames will be automatically adjusted:
```
python /home/bodo/Dropbox/soft/github/PC_geomorph_roughness/pc_geomorph_roughness.py -i Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz -r_m 1 -srd_m 5 --epsg_code 26911 -shp_clp SC12.shp
python /home/bodo/Dropbox/soft/github/PC_geomorph_roughness/pc_geomorph_roughness.py -i Blanca_in_Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz -r_m 1 -srd_m 10 --epsg_code 26911 -shp_clp SC12.shp
```
