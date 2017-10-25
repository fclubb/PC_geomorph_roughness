#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:22:03 2017

@author: Bodo Bookhagen, Oct-Nov, 2017

"""

from laspy.file import File
import copy, glob, time
import numpy as np, os, argparse, pickle, h5py, subprocess, gdal, osr, datetime
from numpy.linalg import svd
from pykdtree.kdtree import KDTree
from scipy import interpolate
from scipy import spatial
from scipy import linalg
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from skimage import exposure

### Command Line Parsing
parser = argparse.ArgumentParser(description='PointCloud processing to estimate local elevation difference, range, and roughness. B. Bookhagen (bodo.bookhagen@uni-potsdam.de), Oct 2017.')
# Important and required:
parser.add_argument('-i', '--inlas', type=str, default='',  help='LAS/LAZ file with point-cloud data. Ideally, this file contains only ground points (class == 2)')
parser.add_argument('-r_m', '--raster_m', type=float, default=1,  help='Raster spacing for subsampling seed points on LAS/LAZ PC. Usually 0.5 to 2 m, default = 1.')
parser.add_argument('-srd_m', '--sphere_radius_m', type=float, default=1.5,  help='Radius of sphere used for selecting lidar points around seed points. These points are used for range, roughness, and density calculations. Default radius 1.5m, i.e., points within a sphere of 3m are chosen.')
parser.add_argument('-slope_srd_m', '--slope_sphere_radius_m', type=float, default=0,  help='Radius of sphere used for fitting a linear plane and calculating slope and detrending data (slope normalization). By default this is similar to the radius used for calculation roughness indices (srd_m), but this can be set to a different value. For example, larger radii use the slope of larger area to detrend data.')
parser.add_argument('-shp_clp', '--shapefile_clip', type=str, default='',  help='Name of shapefile to be used to clip interpolated surfaces too. This is likely the shapefile you have previously generated to subset/clip the point-cloud data.')

# Optional / additional parameters
parser.add_argument('-epsg', '--epsg_code', type=int, default=26911,  help='EPSG code (integer) to define projection information. This should be the same EPSG code as the input data (no re-projection included yet) and can be taken from LAS/LAZ input file. Add this to ensure that output shapefile and GeoTIFFs are properly geocoded.')
parser.add_argument('-o', '--outlas', type=str, default='',  help='LAS file to be created (currently no writing of LAZ files supported). This has the same dimension and number of points as the input LAS/LAZ file, but replaced color values reflecting roughness calculated over a given radius. Note that this will overwrite existing color information in the output file.')
parser.add_argument('-shape_out', '--shapefile_out', type=str, default='',  help='Output shapefile storing calculated attributes for seed points only. Default filename will be generated with radius in the filename.')
parser.add_argument('-odir', '--outputdir', type=str, default='',  help='Output directory to store plots and pickle files. Default is directory containing LAS/LAZ file.')
parser.add_argument('-fig', '--figure', type=bool, default=True,  help='Generate figures while processing. This often takes significant amount of time and can be turned off with -fig False.')
parser.add_argument('-color', '--store_color', type=bool, default=False,  help='Generate a LAS file where deviation from the plane are saved in the color attribute of the LAS file for every point. *Note* that this will replace the color information in the LAS file (but will be saved to separate file). Default is False, can be turned on with --store_color True.')
args = parser.parse_args()

#args.inlas = '/home/bodo/Dropbox/California/SCI/Pozo/catch4bodo/blanca/Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.laz'
#args.inlas='/home/bodo/Dropbox/foo/Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12.las'
#args.shapefile_clip = '/home/bodo/Dropbox/California/SCI/Pozo/catch4bodo/blanca/SC12.shp'

### Function definitions
def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    source = https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    """
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    plane_normal = svd(M)[0][:,-1]
    d = -ctr.dot(plane_normal)
    z = (-plane_normal[0] * points[0,:] - plane_normal[1] * points[1,:] - d) * 1. /plane_normal[2]
    errors = z - points[2,:]
    residual = np.linalg.norm(errors)

    return ctr, plane_normal, residual

def curvFit(points, order=2):
    """
    Fitting a second order polynom to a point cloud and deriving the curvature in a simplified form.
    More details: https://gis.stackexchange.com/questions/37066/how-to-calculate-terrain-curvature
    """
    
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    points = points.T
    
    X,Y = np.meshgrid(np.arange(np.nanmin(points[:,0]),np.nanmax(points[:,0]), args.raster_m/10), np.arange(np.nanmin(points[:,1]),np.nanmax(points[:,1]), args.raster_m/10))
    XX = X.flatten()
    YY = Y.flatten()
    if order == 1:
        # best-fit linear plane
        A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
        C,_,_,_ = linalg.lstsq(A, points[:,2])    # coefficients
        
        # evaluate it on grid
        #Z = C[0]*X + C[1]*Y + C[2]
        
        # or expressed using matrix/vector product
        #Z_order1 = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
        slope = np.mean(C[0:2])
        curvature = np.nan
        curvature_gaussian_mean = np.nan
    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(points.shape[0]), points[:,:2], np.prod(points[:,:2], axis=1), points[:,:2]**2]
        C,_,_,_ = linalg.lstsq(A, points[:,2])
        
        # evaluate it on a grid
        Z_order2 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
        #Z = Ax²y² + Bx²y + Cxy² + Dx² + Ey² + Fxy + Gx + Hy + I
        #Curvature = -2(D + E) * 100
        curvature = -2 * (C[4] + C[5])
        curvature_gaussian_mean = np.nanmean(gaussian_curvature(Z_order2/10))        
        Z_pts = np.dot(np.c_[np.ones(points.shape[0]), points[:,0], points[:,1], points[:,0]*points[:,1], points[:,0]**2, points[:,1]**2], C)
        errors = points[:,2] - Z_pts
        dZ_residuals = np.linalg.norm(errors)
        slope = np.nan
    return C, slope, curvature, curvature_gaussian_mean, dZ_residuals

def gaussian_curvature(Z):
    Zy, Zx = np.gradient(Z)                                                     
    Zxy, Zxx = np.gradient(Zx)                                                  
    Zyy, _ = np.gradient(Zy)                                                    
    K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2             
    return K


def griddata_clip_geotif(tif_fname, points, data2i, xxyy, ncols, nrows, geotransform):
    datai = interpolate.griddata(points, data2i, xxyy, method='nearest')
    output_raster = gdal.GetDriverByName('GTiff').Create(tif_fname,ncols, nrows, 1 ,gdal.GDT_Float32,['TFW=YES', 'COMPRESS=DEFLATE', 'ZLEVEL=9'])  # Open the file, see here for information about compression: http://gis.stackexchange.com/questions/1104/should-gdal-be-set-to-produce-geotiff-files-with-compression-which-algorithm-sh
    output_raster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(args.epsg_code)
    output_raster.SetProjection( srs.ExportToWkt() )
    output_raster.GetRasterBand(1).WriteArray(datai) 
    output_raster.FlushCache()
    output_raster=None
    tif_fname2 = os.path.join(os.path.dirname(tif_fname),'.'.join(os.path.basename(tif_fname).split('.')[0:-1]) + '2.tif')
    cmd = ['gdalwarp', '-dstnodata', '-9999', '-co', 'COMPRESS=DEFLATE', '-co', 'ZLEVEL=9', '-crop_to_cutline', '-cutline', args.shapefile_clip, tif_fname, tif_fname2]
    logfile_fname = os.path.join(args.outputdir, 'log') + '/gdalwarp_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt'
    logfile_error_fname = os.path.join(args.outputdir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt'
    with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
        subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
        subprocess_p.wait()
    os.remove(tif_fname)
    os.rename(tif_fname2, tif_fname)
    cmd = ['gdalinfo', '-hist', '-stats', '-mm', tif_fname]
    logfile_fname = os.path.join(args.outputdir, 'log') + '/gdalinfo_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt'
    logfile_error_fname = os.path.join(args.outputdir, 'log') + '/gdalinfo_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt'
    with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
        subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
        subprocess_p.wait()
    ds = gdal.Open(tif_fname)
    datai = np.array(ds.GetRasterBand(1).ReadAsArray())
    datai[np.where(datai == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
    ds = None
    return datai
def calc_stats_for_seed_points_wrapper(i):
    print('starting {}/{}'.format(i+1, len(pos_array)))
    
    from_pos = pos_array[i] #Get start/end from position array
    to_pos = pos_array[i+1]
    #Setup array for seed point results:
    subarr = np.arange(from_pos,to_pos) #Slice the data into the selected part...
    pts_seed_stats_result = np.empty((subarr.shape[0], nr_of_datasets))
    
    #Setup array for PC results (X, Y, Z, Dz)
    dxyzn_subarr_result = np.empty((subarr.shape[0], dxyzn_max_nre, 4))
 
    for ii in range(subarr.shape[0]):
        pts_seed_stats_result[ii,:], dxyzn_subarr_result[ii,:,:] = calc_stats_for_seed_points(subarr[ii]) #Run point cloud processing for this inddex

    pickle_fn = os.path.join(pickle_dir, 'PC_seed_points_{}.pickle'.format(str(i).zfill(4)))
    pickle.dump((pts_seed_stats_result, dxyzn_subarr_result), open(pickle_fn,'wb'))
    print('...stored {}'.format(str(i).zfill(7)))
    pts_seed_stats_result = None
    dxyzn_subarr_result = None
        
def calc_stats_for_seed_points(i):
    pts_xyz = pcl_xyzg[pcl_xyzg_radius[i]]
    pts_xyz_slope = pcl_xyzg[pcl_xyzg_radius_slope[i]]
        
    nr_pts_xyz = pts_xyz.shape[0]
    if pts_xyz.shape[0] < 5:
        print('Less than 5 points, plane fitting not meaningful for i = %s'%"{:,}".format(i))
        pts_xyz_meanpt = np.nan
        pts_xyz_normal = np.nan
        pts_seed_stats = [pcl_xyzg_rstep_seed[i,0], pcl_xyzg_rstep_seed[i,1], pcl_xyzg_rstep_seed[i,2], 
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                   np.nan, np.nan, np.nan, np.nan, nr_pts_xyz, np.nan, np.nan, np.nan]
        dxyzn = np.empty((dxyzn_max_nre, 4))
        dxyzn.fill(np.nan)
    else:
        pts_xyz_meanpt, pts_xyz_normal, plane_residual = planeFit(pts_xyz_slope.T)
        #residual calculated from = np.linalg.norm(errors)

        #calculate curvature
        C, _, curvature, curvature_gaussian_mean, curv_residuals = curvFit(pts_xyz_slope.T, order=2)
        
        #normalize /detrend points with plane
        d = -pts_xyz_meanpt.dot(pts_xyz_normal)
        z = (-pts_xyz_normal[0] * pts_xyz[:,0] - pts_xyz_normal[1] * pts_xyz[:,1] - d) * 1. /pts_xyz_normal[2]
        plane_slope = pts_xyz_normal[2]
        #calculate offset for each point from plane
        dz = pts_xyz[:,2] - z
    
        #stack points into X, Y, Z, delta-Z for each point
        dxyzn = np.empty((dxyzn_max_nre, 4))
        dxyzn.fill(np.nan)
        dxyzn[range(pts_xyz.shape[0]),:] = np.vstack([np.vstack((pts_xyz[:,0], pts_xyz[:,1], pts_xyz[:,2], dz)).T])
    
        #for each seed point, store relevant point statistics. Columns are:
        #Seed-X, Seed-Y, Seed-Z, Mean-X, Mean-Y, Mean-Z, Z-min, Z-max, Dz-max, Dz-min,  Dz-std.dev, Dz-range, Dz-90-10th percentile range, slope of fitted plane, plane residuals, nr. of lidar points, curvature
        pts_seed_stats = np.array([pcl_xyzg_rstep_seed[i,0], pcl_xyzg_rstep_seed[i,1], pcl_xyzg_rstep_seed[i,2], 
                       pts_xyz_meanpt[0], pts_xyz_meanpt[1], pts_xyz_meanpt[2], 
                       np.min(pts_xyz, axis=0)[2], np.max(pts_xyz, axis=0)[2], dz.max(), dz.min(), np.std(dz), dz.max()-dz.min(), \
                       np.percentile(dz, 90)-np.percentile(dz,10), np.percentile(dz, 75)-np.percentile(dz,25), plane_slope, plane_residual, np.var(dz), nr_pts_xyz, curvature, curvature_gaussian_mean, curv_residuals])
    return pts_seed_stats, dxyzn

### Program starts here
### Defining input and setting global variables
if args.inlas == '':
    print('No input LAS/LAZ file given. Rerun with -i for input LAS/LAZ file. Exit.')
    exit()

if args.outputdir == '':
    args.outputdir = os.path.dirname(args.inlas)

inFile = File(args.inlas, mode='r')
if args.outlas == '':
    args.outlas = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + '_raster_%0.2fm_rsphere%0.2fm.laz'%(args.raster_m,args.sphere_radius_m))

if args.slope_sphere_radius_m == 0:
    #set to sphere_radius_m
    args.slope_sphere_radius_m = args.sphere_radius_m

if args.shapefile_out == '':
    args.shapefile_out = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + '_shp_%0.2fm_rsphere%0.2fm_EPSG%d.shp'%(args.raster_m,args.sphere_radius_m, args.epsg_code))

if os.path.exists(os.path.join(args.outputdir, 'log')) == False:
    os.mkdir(os.path.join(args.outputdir, 'log'))
pickle_dir = os.path.join(args.outputdir, 'pickle')
if os.path.exists(pickle_dir) == False:
    os.mkdir(pickle_dir)

figure_dir = os.path.join(args.outputdir, 'figures')
if os.path.exists(figure_dir) == False:
    os.mkdir(figure_dir)

geotif_dir = os.path.join(args.outputdir, 'geotif')
if os.path.exists(geotif_dir) == False:
    os.mkdir(geotif_dir)
nrlidari_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_%0.2fm_rsphere%0.2fm_EPSG%d_nrlidar.tif'%(args.raster_m,args.sphere_radius_m, args.epsg_code))
da_stdi_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_%0.2fm_rsphere%0.2fm_EPSG%d_stddev.tif'%(args.raster_m,args.sphere_radius_m, args.epsg_code))
dz_range9010i_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_%0.2fm_rsphere%0.2fm_EPSG%d_da_range9010.tif'%(args.raster_m,args.sphere_radius_m, args.epsg_code))
dz_range7525i_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_%0.2fm_rsphere%0.2fm_EPSG%d_da_range7525.tif'%(args.raster_m,args.sphere_radius_m, args.epsg_code))
plane_slopei_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_%0.2fm_rsphere%0.2fm_EPSG%d_planeslope.tif'%(args.raster_m,args.sphere_radius_m, args.epsg_code))
dz_max_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_%0.2fm_rsphere%0.2fm_EPSG%d_dzmax.tif'%(args.raster_m,args.sphere_radius_m, args.epsg_code))
dz_min_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_%0.2fm_rsphere%0.2fm_EPSG%d_dzmin.tif'%(args.raster_m,args.sphere_radius_m, args.epsg_code))
plane_curvi_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_%0.2fm_rsphere%0.2fm_EPSG%d_curv.tif'%(args.raster_m,args.sphere_radius_m, args.epsg_code))

### Loading data and filtering
print('Loading input file: %s'%args.inlas)
pcl_xyzc = np.vstack((inFile.get_x()*inFile.header.scale[0]+inFile.header.offset[0], inFile.get_y()*inFile.header.scale[1]+inFile.header.offset[1], inFile.get_z()*inFile.header.scale[2]+inFile.header.offset[2], inFile.get_classification())).transpose()
#pcl_xyzc is now a point cloud with x, y, z, and classification
#if args.store_color == True:
#    pcl_i = inFile.get_intensity().copy()
#    pcl_blue = inFile.get_blue().copy()
#    pcl_green = inFile.get_green().copy()
#    pcl_red = inFile.get_red().copy()
print('Loaded %s points'%"{:,}".format(pcl_xyzc.shape[0]))

print('Filtering points to only work with ground points (class == 2)... ',end='')
#get only ground points:
idx_ground = np.where(pcl_xyzc[:,3] == 2)[0]
pcl_xyzg = pcl_xyzc[idx_ground,0:3]
#pcl_xyzg is a point cloud with x, y, z, and for class == 2 only
pcl_xyg = pcl_xyzc[idx_ground,0:2]
#if args.store_color == True:
##    pcl_i = pcl_i[idx_ground,:]
#    pcl_blue = pcl_blue[idx_ground,:]
#    pcl_green = pcl_green[idx_ground,:]
#    pcl_red = pcl_red[idx_ground,:]
print('done.')

### cKDTree setup and calculation
#Generate KDTree for fast searching
#cKDTree is faster than KDTree, pyKDTree is fast then cKDTree
print('Generating XY-cKDTree... ',end='', flush=True)
pcl_xyg_ckdtree_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + '_xyg_cKDTree.pickle')
if os.path.exists(pcl_xyg_ckdtree_fn):
    pcl_xyg_ckdtree = pickle.load(open( pcl_xyg_ckdtree_fn, "rb" ))
else:
    pcl_xyg_ckdtree = spatial.cKDTree(pcl_xyg, leafsize=10)
    pickle.dump(pcl_xyg_ckdtree, open(pcl_xyg_ckdtree_fn,'wb'))
print('done.')

print('Generating XYZ-cKDTree... ',end='',flush=True)
pcl_xyzg_ckdtree_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + '_xyzg_cKDTree.pickle')
if os.path.exists(pcl_xyzg_ckdtree_fn):
    pcl_xyzg_ckdtree = pickle.load(open( pcl_xyzg_ckdtree_fn, "rb" ))
else:
    pcl_xyzg_ckdtree = spatial.cKDTree(pcl_xyzg, leafsize=10)
    pickle.dump(pcl_xyzg_ckdtree,open(pcl_xyzg_ckdtree_fn,'wb'))
print('done.')

    
### Search KDTree with points on a regularly-spaced raster
#generating equally-spaced raster overlay from input coordinates with stepsize rstep_size
#This will be used to query the point cloud. Step_size should be small enough and likely 1/2 of the output file resolution. 
#Note that this uses a 2D raster overlay to slice a 3D point cloud.
rstep_size = args.raster_m
[x_min, x_max] = np.min(pcl_xyzg[:,0]), np.max(pcl_xyzg[:,0])
[y_min, y_max] = np.min(pcl_xyzg[:,1]), np.max(pcl_xyzg[:,1])
[z_min, z_max] = np.min(pcl_xyzg[:,2]), np.max(pcl_xyzg[:,2])
x_elements = len(np.arange(x_min.round(), x_max.round(), args.raster_m))
y_elements = len(np.arange(y_min.round(), y_max.round(), args.raster_m))

#get coordinate range and shift coordinates by half of the step size to make sure rater overlay is centered. 
#This is not really necessary and only matters for very small point clouds with edge effects or for very large steps sizes:
x_coords = np.arange(x_min.round(), x_max.round(), args.raster_m) + args.raster_m / 2
y_coords = np.arange(y_min.round(), y_max.round(), args.raster_m) + args.raster_m / 2

#create combination of all coordinates (this is using lists and could be optimized)
xy_coordinates = np.array([(x,y) for x in x_coords for y in y_coords])

#using the 2D KDTree to find the points that are closest to the defined 2D raster overlay
[pcl_xyg_ckdtree_distance, pcl_xyg_ckdtree_id] = pcl_xyg_ckdtree.query(xy_coordinates, k=1)

#take only points that are within the search radius (may omit some points at the border regions
pcl_distances_lt_rstep_size = np.where(pcl_xyg_ckdtree_distance <= args.raster_m)[0]

#the following list contains all IDs to the actual lidar points that are closest to the raster overlay. 
#We will use these points as seed points for determining slope, planes, and point-cloud ranges
pcl_xyg_ckdtree_id = pcl_xyg_ckdtree_id[pcl_distances_lt_rstep_size]

#remove from memory
pcl_xyg_ckdtree = None

#now select these points from the 3D pointcloud with X, Y, Z coordinates.
#We refer to these as seed points from the rstep part of this script
pcl_xyzg_rstep_seed = pcl_xyzg[pcl_xyg_ckdtree_id]

#Generate figure:
seed_fn = '_seed_pts_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
pcl_seed_pts_fig_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + seed_fn)
print('Generating figure %s... '%pcl_seed_pts_fig_fn, end='', flush=True)
if os.path.exists(pcl_seed_pts_fig_fn) == False:
    #plot seed points as map
    fig = plt.figure(figsize=(16.53,11.69), dpi=300)
    fig.clf()
    ax = fig.add_subplot(211, projection='3d')
    #plt.scatter(pcl_xyzg[:,0], pcl_xyzg[:,1], pcl_xyzg[:,2], c='gray', s=1)
    ax.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], pcl_xyzg_rstep_seed[:,2], c=pcl_xyzg_rstep_seed[:,2], s=0.2)
    #ax.cbar()
    ax.grid()
    ax.set_title('3D view of lidar seed points (ground only)')
    ax.set_xlabel('UTM-X (m)')
    ax.set_ylabel('UTM-Y (m)')
    ax.set_zlabel('Elevation (m)')
    
    ax = fig.add_subplot(212)
    ax.scatter(pcl_xyzg[:,0], pcl_xyzg[:,1], c='gray', s=0.01)
    cax = ax.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=pcl_xyzg_rstep_seed[:,2], s=0.1)
    ax.grid()
    ax.set_title('Map view of lidar seed points (colored) and all lidar point (gray) (ground only)')
    cbar = fig.colorbar(cax)
    ax.set_xlabel('UTM-X (m)')
    ax.set_ylabel('UTM-Y (m)')
    ax.axis('equal')
    fig.savefig(pcl_seed_pts_fig_fn, bbox_inches='tight')
    plt.close()
print('done.')

### Query points - use KDTree
#find points from 3D seed / query points  / raster overlay with radius = args.sphere_radius_m
pickle_fn = '_xyzg_raster_%0.2fm_radius_%0.2fm.pickle'%(args.raster_m, args.sphere_radius_m)
pcl_xyzg_radius_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + pickle_fn)
print('Querying cKDTree with radius %0.2f and storing in pickle: %s... '%(args.sphere_radius_m, pcl_xyzg_radius_fn), end='', flush=True)
if os.path.exists(pcl_xyzg_radius_fn):
    pcl_xyzg_radius = pickle.load(open( pcl_xyzg_radius_fn, "rb" ))
else:
    pcl_xyzg_radius = pcl_xyzg_ckdtree.query_ball_point(pcl_xyzg_rstep_seed,r=args.sphere_radius_m)
    pickle.dump(pcl_xyzg_radius, open(pcl_xyzg_radius_fn,'wb'))
print('done.')

# Find sphere points for slope and curvature calculation
if args.slope_sphere_radius_m != args.sphere_radius_m:    
    pickle_fn = '_xyzg_raster_%0.2fm_radius_%0.2fm.pickle'%(args.raster_m, args.slope_sphere_radius_m)
    pcl_xyzg_radius2_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + pickle_fn)
    print('Querying cKDTree with radius %0.2f and storing in pickle: %s... '%(args.slope_sphere_radius_m, pcl_xyzg_radius2_fn), end='', flush=True)
    if os.path.exists(pcl_xyzg_radius2_fn):
        pcl_xyzg_radius_slope = pickle.load(open( pcl_xyzg_radius2_fn, "rb" ))
    else:
        pcl_xyzg_radius_slope = pcl_xyzg_ckdtree.query_ball_point(pcl_xyzg_rstep_seed,r=args.slope_sphere_radius_m)
        pickle.dump(pcl_xyzg_radius_slope, open(pcl_xyzg_radius2_fn,'wb'))
    print('done.')
elif args.slope_sphere_radius_m == args.sphere_radius_m:
    pcl_xyzg_radius_slope = pcl_xyzg_radius
  
#NOTE: Seperately storing and loading points for slope sphere and roughness sphere. This could be optimized so that only one will be loaded.
    
### Calculate statistics for each sphere: normalization, elevation range, std. dev., mean, median
#Setup variables
pcl_xyzg_radius_nr = len(pcl_xyzg_radius)
nr_of_datasets = 21
nr_of_processes = 100 #splitting the for loop into 100 processes and dividing array into 100 steps in pos_array
pos_array = np.array(np.linspace(0, pcl_xyzg_radius_nr,nr_of_processes), dtype=int) #This creates a position array so you can select from:to in each loop
dxyzn_max_nre = np.max([len(x) for x in pcl_xyzg_radius])
dxyzn_nre = np.sum([len(x) for x in pcl_xyzg_radius])
dxyzn_nre_pos_array = np.array(np.linspace(0, dxyzn_nre, nr_of_processes), dtype=int)
#pcl_xyzg_radius_nre = np.sum([len(x) for x in pcl_xyzg_radius])

    
### Initiate parallel run
pts_seed_stats = np.empty((pcl_xyzg_radius_nr,18))
pcl_xyzg_radius_nre = np.sum([len(x) for x in pcl_xyzg_radius])
dxyzn = np.empty((pcl_xyzg_radius_nre, 4))
counter = 0
#generate seed HDF filename and load data from HDF file if available
seed_pts_stats_hdf = '_seed_pts_stats_raster_%0.2fm_radius_%0.2fm.h5'%(args.raster_m, args.sphere_radius_m)
pcl_seed_pts_stats_hdf_fn = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_hdf)
print('Extracting point-cloud statistics from %s'%os.path.basename(pcl_seed_pts_stats_hdf_fn))
if os.path.exists(pcl_seed_pts_stats_hdf_fn) == False:
    # generate index array and split into nr_of_processes
    p = Pool()
    ts = time.time()
    #indices = range(0, pcl_xyzg_radius_nr - 1)
    for _ in p.imap_unordered(calc_stats_for_seed_points_wrapper, np.arange(0,len(pos_array)-1)):
        pass    
    print(str(time.time() - ts) + 's, ' + str((time.time() - ts)/60) + 'm')

    #combine pickle files
    pkls = glob.glob(os.path.join(pickle_dir, 'PC_seed_points_*')) #Now get all the pickle files we made
    pkls.sort() #make sure they're sorted
    dxyzn = np.empty((dxyzn_nre, 4)) #output for every lidar point (dz value)
    pts_seed_stats = np.empty((pcl_xyzg_radius_nr,nr_of_datasets)) #output for seed points
    count = 0
    dxyzn_counter = 0
    for fid in pkls:
        seed_res, dxyzn_res = pickle.load(open(fid,'rb')) #Loop through and put each pickle into the right place in the output array
        if seed_res.shape[0] != pos_array[count+1] - pos_array[count]:
            print('File %s, length of records do not match. file: %d vs pos_array: %d'%(fid, seed_res.shape[0], pos_array[count+1] - pos_array[count]))
            if seed_res.shape[0] < pos_array[count+1] - pos_array[count]:
                pts_seed_stats[range(pos_array[count],pos_array[count+1]-1),:] = seed_res
            elif seed_res.shape[0] > pos_array[count+1] - pos_array[count]:
                pts_seed_stats[range(pos_array[count],pos_array[count+1]),:] = seed_res[:-1]
        else:
            pts_seed_stats[range(pos_array[count],pos_array[count+1]),:] = seed_res
        #re-arrange dxyzn and remove nans
        dxyzn_reshape = dxyzn_res.reshape((dxyzn_res.shape[0]*dxyzn_res.shape[1], dxyzn_res.shape[2]))
        idx_nonan = np.where(np.isnan(dxyzn_reshape[:,0]) == False)[0]
        dxyzn_reshape = dxyzn_reshape[idx_nonan,:]
        dxyzn[range(dxyzn_counter,dxyzn_counter+dxyzn_reshape.shape[0]),:] = dxyzn_reshape
        dxyzn_counter = dxyzn_counter + dxyzn_reshape.shape[0]
        count += 1
        del seed_res, dxyzn_res, dxyzn_reshape
    #remove pickle files
    for i in pkls:
        os.remove(i)
    pkls=None

    #pts_seed_stats: clean up Nan and 0
    idxnan = np.where(np.isnan(pts_seed_stats[:,0]) == False)[0]
    pts_seed_stats = pts_seed_stats[idxnan,:]
    idx0 = np.where(pts_seed_stats[:,0] != 0)[0]
    pts_seed_stats = pts_seed_stats[idx0,:]

    #dxyzn: clean up Nan and 0
    idxnan = np.where(np.isnan(dxyzn[:,0]) == False)[0]
    dxyzn = dxyzn[idxnan,:]
    idx0 = np.where(dxyzn[:,0] != 0)[0]
    dxyzn = dxyzn[idx0,:]    
    idxnan = None
    idx0 = None
else:
    hdf_in = h5py.File(pcl_seed_pts_stats_hdf_fn,'r')
    pts_seed_stats = np.array(hdf_in['pts_seed_stats'])
    dxyzn = np.array(hdf_in['dxyzn'])
    print('Statistics loaded from file: %s'%os.path.basename(pcl_seed_pts_stats_hdf_fn))
print('done.')

### Write to LAS/LAZ file
if args.store_color == True:
    if os.path.exists(args.outlas) == False:
        print('Writing dz values to LAZ file: %s... '%args.outlas, end='', flush=True)    
        # use dxyzn and find unique points by x, y, z coordinate
        xyz_points = dxyzn[:,:-1]
        xyz_points_unique = np.array(list(set(tuple(p) for p in xyz_points)))
        dz = np.empty(xyz_points_unique.shape[0])
        #now find corresponding roughness (dz) values for each unique pair
        dxyzn_pykdtree = KDTree(xyz_points)
        dxyzn_dist, dxyzn_id = dxyzn_pykdtree.query(xyz_points_unique, k=1)
    
        for i in np.arange(dxyzn_id.shape[0]):
            dz[i] = dxyzn[dxyzn_id[i],3]
            
        #normalize input and generate colors using colormap
        v = dz
        #stretch to 10-90th percentile
        v_1090p = np.percentile(v, [10, 90])
        v_rescale = exposure.rescale_intensity(v, in_range=(v_1090p[0], v_1090p[1]))
        colormap_PuOr = mpl.cm.PuOr
        rgb = colormap_PuOr(v_rescale)
        #remove last column - alpha value
        rgb = (rgb[:, :3] * (np.power(2,16)-1)).astype('uint16')
    
        outFile = File(args.outlas, mode='w', header=inFile.header)
        new_header = copy.copy(outFile.header)
        #setting some variables
        new_header.created_year = datetime.datetime.now().year
        new_header.created_day = datetime.datetime.now().timetuple().tm_yday
        new_header.x_max = xyz_points_unique[:,0].max()
        new_header.x_min = xyz_points_unique[:,0].min()
        new_header.y_max = xyz_points_unique[:,1].max()
        new_header.y_min = xyz_points_unique[:,1].min()
        new_header.z_max = xyz_points_unique[:,2].max()
        new_header.z_min = xyz_points_unique[:,2].min()
        new_header.point_records_count = dz.shape[0]
        new_header.point_return_count = 0
        outFile.header.count = dz.shape[0]
    #    outFile.Classification = np.ones((dz.shape[0])).astype('uint8') * 2
        outFile.X = xyz_points_unique[:,0]
        outFile.Y = xyz_points_unique[:,1]
        outFile.Z = xyz_points_unique[:,2]
        outFile.Red = rgb[:,0]
        outFile.Green = rgb[:,1]
        outFile.Blue = rgb[:,2]    
        outFile.close()    
        print('done.')

print('Writing seed points and statistics to HDF, CSV, and shapefiles... ', end='', flush=True)
### Write Seed point statistics to file
if os.path.exists(pcl_seed_pts_stats_hdf_fn) == False:
    hdf_out = h5py.File(pcl_seed_pts_stats_hdf_fn,'w')
    hdf_out.attrs['help'] = 'Array from pc_dh_roughness.py with raster size %0.2fm and sphere radius %0.2fm'%(args.raster_m, args.sphere_radius_m)
    pts_seeds_stats_fc = hdf_out.create_dataset('pts_seed_stats',data=pts_seed_stats, chunks=True, compression="gzip", compression_opts=7)
    pts_seeds_stats_fc.attrs['help'] = 'Seed-X, Seed-Y, Seed-Z, Mean-X, Mean-Y, Mean-Z, Z-min, Z-max, Dz-max, Dz-min,  Dz-std.dev, Dz-range, Dz-90-10thp, Dz-75-25thp, PlaneSlope, residuals, PlaneVariance, NrLidarPoints, curvature, curvature_gaussian_mean, curv_residuals'
    dxyzn_fc = hdf_out.create_dataset('dxyzn',data=dxyzn, chunks=True, compression="gzip", compression_opts=7)
    dxyzn_fc.attrs['help'] = 'Lidar points and their deviation from a plane with radius %0.2f'%args.sphere_radius_m
    hdf_out.close()

#write csv
header_str='1SeedX, 2SeedY, 3SeedZ, 4MeanX, 5MeanY, 6MeanZ, 7Z_min, 8Z_max, 9Dz_max, 10Dz_min,  11Dz_std, 12Dz_range, 13Dz_9010p, 14Dz_7525p, 15_Pl_slp, 16Pl_res, 17Pl_Var, 18Nr_lidar, 19curv, 20curv_gaussian, 21curv_res'
seed_pts_stats_csv = '_seed_pts_stats_raster_%0.2fm_radius_%0.2fm.csv'%(args.raster_m, args.sphere_radius_m)
pcl_seed_pts_stats_csv_fn = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_csv)
seed_pts_stats_vrt = '_seed_pts_stats_raster_%0.2fm_radius_%0.2fm.vrt'%(args.raster_m, args.sphere_radius_m)
pcl_seed_pts_stats_vrt_fn = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_vrt)
idxnan = np.where(np.isnan(pts_seed_stats))
if os.path.exists(pcl_seed_pts_stats_csv_fn) == False:
    #before writing to CSV file, replace all np.nan in pts_seed_stats with -9999
    pts_seed_stats_nonan = np.copy(pts_seed_stats)
    pts_seed_stats_nonan[idxnan] = -9999.
    np.savetxt(pcl_seed_pts_stats_csv_fn, pts_seed_stats_nonan, fmt='%.4f', delimiter=',', header=header_str)
pts_seed_stats_nonan = None
idxnan = None

# write VRT for shapefile generation
vrt_f = open(pcl_seed_pts_stats_vrt_fn,'w')
vrt_f.write('<OGRVRTDataSource>\n')
vrt_f.write('\t<OGRVRTLayer name="%s">\n'%os.path.basename(pcl_seed_pts_stats_vrt_fn))
vrt_f.write('\t\t<SrcDataSource>%s</SrcDataSource>\n'%os.path.basename(pcl_seed_pts_stats_csv_fn))
vrt_f.write('\t\t<SrcLayer>%s</SrcLayer>\n'%'.'.join(os.path.basename(pcl_seed_pts_stats_csv_fn).split('.')[0:-1]))
vrt_f.write('\t\t<LayerSRS>EPSG:%d</LayerSRS>\n'%args.epsg_code)
vrt_f.write('\t\t<GeometryType>wkbPoint</GeometryType>\n')
vrt_f.write('\t\t<GeometryField encoding="PointFromColumns" x="4MeanX" y="5MeanY"/>\n')
vrt_f.write('\t\t\t<Field name="1SeedX" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="2SeedY" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="3SeedZ" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="4MeanX" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="5MeanY" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="6MeanZ" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="7Z_min" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="8Z_max" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="9Dz_max" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="10Dz_min" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="11Dz_std" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="12Dz_range" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="13Dz_9010p" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="14Dz_7525p" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="15Pl_slp" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="16Pl_res" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="17Pl_var" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="18Nr_lidar" type="Real" width="8"/>\n')
vrt_f.write('\t\t\t<Field name="19curv" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="20curv_gaussian" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t\t\t<Field name="21curv_res" type="Real" width="8" precision="7"/>\n')
vrt_f.write('\t</OGRVRTLayer>\n')
vrt_f.write('</OGRVRTDataSource>\n')
vrt_f.close()

# Generate shapefile from vrt
if os.path.exists(args.shapefile_out) == False:
    cmd = ['ogr2ogr', args.shapefile_out, pcl_seed_pts_stats_vrt_fn]
    logfile_fname = os.path.join(args.outputdir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt'
    logfile_error_fname = os.path.join(args.outputdir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt'
    with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
        subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
        subprocess_p.wait()
print('done.')

### Interpolate to equally-spaced grid and generate GeoTIFF output
if os.path.exists(nrlidari_tif_fn) == False or os.path.exists(da_stdi_tif_fn) == False or \
    os.path.exists(dz_range9010i_tif_fn) == False or os.path.exists(plane_slopei_tif_fn) == False:
    print('\nInterpolating seed points (mean-X, mean-Y) to geotiff rasters and writing geotiff raster... ',end='', flush=True)
    x = np.arange(x_min.round(), x_max.round(), args.raster_m)
    y = np.arange(y_min.round(), y_max.round(), args.raster_m)
    xx,yy = np.meshgrid(x,y)
    idx_nonan = np.where(np.isnan(pts_seed_stats[:,3])==False)
    points = np.hstack((pts_seed_stats[idx_nonan,3].T, pts_seed_stats[idx_nonan,4].T))

    nrows,ncols = np.shape(xx)
    xres = (x.max()-x.min())/float(ncols)
    yres = (y.max()-y.min())/float(nrows)
    geotransform=(x.min(),xres,0,y.min(),0, yres) 

    #interpolate nr_lidar_measurements
    if os.path.exists(nrlidari_tif_fn) == False:
        nr_lidari = griddata_clip_geotif(nrlidari_tif_fn, points, pts_seed_stats[idx_nonan,17][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(nrlidari_tif_fn)
        nr_lidari = np.array(ds.GetRasterBand(1).ReadAsArray())
        nr_lidari[np.where(nr_lidari == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None

    if os.path.exists(da_stdi_tif_fn) == False:
        dz_stdi = griddata_clip_geotif(da_stdi_tif_fn, points, pts_seed_stats[idx_nonan,10][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(da_stdi_tif_fn)
        dz_stdi = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_stdi[np.where(dz_stdi == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None

    #interpolate Dz_range 90-10 percentile
    if os.path.exists(dz_range9010i_tif_fn) == False:
        dz_range9010i = griddata_clip_geotif(dz_range9010i_tif_fn, points, pts_seed_stats[idx_nonan,12][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_range9010i_tif_fn)
        dz_range9010i = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_range9010i[np.where(dz_range9010i == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz_range 75-25 percentile
    if os.path.exists(dz_range7525i_tif_fn) == False:
        dz_range7525i = griddata_clip_geotif(dz_range7525i_tif_fn, points, pts_seed_stats[idx_nonan,13][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_range7525i_tif_fn)
        dz_range7525i = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_range7525i[np.where(dz_range7525i == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz-max
    if os.path.exists(dz_max_tif_fn) == False:
        dz_maxi = griddata_clip_geotif(dz_max_tif_fn, points, pts_seed_stats[idx_nonan,8][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_max_tif_fn)
        dz_maxi = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_maxi[np.where(dz_maxi == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None

    #interpolate Dz-min
    if os.path.exists(dz_min_tif_fn) == False:
        dz_mini = griddata_clip_geotif(dz_min_tif_fn, points, pts_seed_stats[idx_nonan,9][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_min_tif_fn)
        dz_mini = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_mini[np.where(dz_mini == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None

    #interpolate Plane_slope
    if os.path.exists(plane_slopei_tif_fn) == False:
        plane_slopei = griddata_clip_geotif(plane_slopei_tif_fn, points, pts_seed_stats[idx_nonan,14][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(plane_slopei_tif_fn)
        plane_slopei = np.array(ds.GetRasterBand(1).ReadAsArray())
        plane_slopei[np.where(plane_slopei == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None

    #interpolate Plane_slope
    if os.path.exists(plane_curvi_tif_fn) == False:
        plane_curvi = griddata_clip_geotif(plane_curvi_tif_fn, points, pts_seed_stats[idx_nonan,18][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(plane_curvi_tif_fn)
        plane_curvi = np.array(ds.GetRasterBand(1).ReadAsArray())
        plane_curvi[np.where(plane_curvi == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None

    print('done.')
   
    # Could use gdal_grid to generate TIF from VRT/Shapefile:
    #gdal_grid -zfield "15Nr_lidar" -outsize 271 280 -a linear:radius=2.0:nodata=-9999 -of GTiff -ot Int16 Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_seed_pts_stats_raster_1.00m_radius_1.50m.vrt Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_seed_pts_stats_raster_1.00m_radius_1.50m_nrlidar2.tif --config GDAL_NUM_THREADS ALL_CPUS -co COMPRESS=DEFLATE -co ZLEVEL=9
    
if os.path.exists(args.shapefile_out) == False:
    print('Writing shapefile',end='', flush=True)
    cmd = ['ogr2ogr', args.shapefile_out, pcl_seed_pts_stats_vrt_fn]
    logfile_fname = os.path.join(args.outputdir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt'
    logfile_error_fname = os.path.join(args.outputdir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt'
    with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
        subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
        subprocess_p.wait()
print('done.')

### Plot output to figures
seed_pts_stats_png = '_seed_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
pcl_seed_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_png)
print('Generating overview figure %s... '%os.path.basename(pcl_seed_pts_fig_overview_fn), end='', flush=True)
if os.path.exists(pcl_seed_pts_fig_overview_fn) == False:
    fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
    #fig = plt.figure(figsize=(11.69,8.27), dpi=150)
    fig.clf()
    
    ax1 = fig.add_subplot(231)
    ax1.grid()
    cax1 = ax1.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,2], s=0.5, cmap=plt.get_cmap('terrain'), linewidth=0)
    ax1.set_title('Lidar seed point elevation with raster=%0.2fm'%args.raster_m,y=1.05)
    cbar = fig.colorbar(cax1)
    cbar.set_label('Max-Min elevation in sphere (m)')
    ax1.set_xlabel('UTM-X (m)')
    ax1.set_ylabel('UTM-Y (m)')
    ax1.axis('equal')

    ax2 = fig.add_subplot(232)
    ax2.grid()
    cax2 = ax2.imshow(nr_lidari,cmap=plt.get_cmap('gnuplot'))
    #ax2.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,15], s=0.1, cmap=plt.get_cmap('gnuplot'), linewidth=0)    
    ax2.set_title('Nr. of lidar measurements for each seed point',y=1.05)
    cbar = fig.colorbar(cax2)
    cbar.set_label('#')
    ax2.set_xlabel('UTM-X (m)')
    ax2.set_ylabel('UTM-Y (m)')
    ax2.axis('equal')
    
    ax3 = fig.add_subplot(233)
    ax3.grid()
    #cax3 = ax3.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,13], s=0.5, cmap=plt.get_cmap('seismic'), vmin=np.nanpercentile(pts_seed_stats[:,13], 10), vmax=np.nanpercentile(pts_seed_stats[:,13], 90), linewidth=0)
    cax3 = ax3.imshow(plane_slopei, cmap=plt.get_cmap('seismic'), vmin=np.nanpercentile(plane_slopei, 10), vmax=np.nanpercentile(plane_slopei, 90))
    ax3.set_title('Slope of fitted plane for each sphere (r=%0.2f)'%args.sphere_radius_m,y=1.05)
    cbar = fig.colorbar(cax3)
    cbar.set_label('Slope (m/m)')
    ax3.set_xlabel('UTM-X (m)')
    ax3.set_ylabel('UTM-Y (m)')
    ax3.axis('equal')

    ax4 = fig.add_subplot(234)
    ax4.grid()
    #cax4 = ax4.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,12], s=0.5, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(pts_seed_stats[:,12], 10), vmax=np.nanpercentile(pts_seed_stats[:,12], 90), linewidth=0)
    cax4 = ax4.imshow(dz_range9010i, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(dz_range9010i, 10), vmax=np.nanpercentile(dz_range9010i, 90))
    ax4.set_title('Surface roughness I: Range of offsets from linear plane (90-10th) with r=%0.2f)'%args.sphere_radius_m,y=1.05)
    cbar = fig.colorbar(cax4)
    cbar.set_label('Range (90-10th) of plane offsets (m)')
    ax4.set_xlabel('UTM-X (m)')
    ax4.set_ylabel('UTM-Y (m)')
    ax4.axis('equal')

    ax5 = fig.add_subplot(235)
    ax5.grid()
    cax5 = ax5.imshow(plane_curvi, cmap=plt.get_cmap('jet'), vmin=np.nanpercentile(plane_curvi, 10), vmax=np.nanpercentile(plane_curvi, 90))
    ax5.set_title('Curvature of sphere/disc with r=%0.2f)'%args.sphere_radius_m,y=1.05)
    cbar = fig.colorbar(cax5)
    cbar.set_label('Curvature (1/m^2)')
    ax5.set_xlabel('UTM-X (m)')
    ax5.set_ylabel('UTM-Y (m)')
    ax5.axis('equal')

    ax6 = fig.add_subplot(236)
    ax6.grid()
    #cax6 = ax6.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,10], s=0.5, cmap=plt.get_cmap('jet'), vmin=np.nanpercentile(pts_seed_stats[:,10], 10), vmax=np.nanpercentile(pts_seed_stats[:,10], 90), linewidth=0)
    cax6 = ax6.imshow(dz_stdi, cmap=plt.get_cmap('jet'), vmin=np.nanpercentile(dz_stdi, 10), vmax=np.nanpercentile(dz_stdi, 90))
    ax6.set_title('Surface roughness II: Std. deviation of lidar points from plane with r=%0.2f)'%args.sphere_radius_m,y=1.05)
    cbar = fig.colorbar(cax6)
    cbar.set_label('Std. deviation (m)')
    ax6.set_xlabel('UTM-X (m)')
    ax6.set_ylabel('UTM-Y (m)')
    ax6.axis('equal')

    fig.savefig(pcl_seed_pts_fig_overview_fn, bbox_inches='tight')
    plt.close()
print('done.')

nrlidar_pts_stats_png = '_nrlidar_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
pcl_nrlidar_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + nrlidar_pts_stats_png)
print('Generating figure for number of lidar measurement: %s... '%os.path.basename(pcl_nrlidar_pts_fig_overview_fn), end='', flush=True)
if os.path.exists(pcl_seed_pts_fig_overview_fn) == False:
    fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
    fig.clf()
    
    ax2 = fig.add_subplot(111)
    ax2.grid()
    cax2 = ax2.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,15], s=3, cmap=plt.get_cmap('gnuplot'), linewidth=0)
    ax2.set_title('Nr. of lidar measurements for each seed point',y=1.05)
    cbar = fig.colorbar(cax2)
    cbar.set_label('#')
    ax2.set_xlabel('UTM-X (m)')
    ax2.set_ylabel('UTM-Y (m)')
    ax2.axis('equal')
    fig.savefig(pcl_nrlidar_pts_fig_overview_fn, bbox_inches='tight')
    plt.close()
print('done.')

slope_pts_stats_png = '_slope_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
pcl_slope_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + slope_pts_stats_png)
print('Generating figure for slope of fitted plane: %s... '%os.path.basename(pcl_slope_pts_fig_overview_fn), end='', flush=True)
if os.path.exists(pcl_slope_pts_fig_overview_fn) == False:
    fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
    fig.clf()
    
    ax3 = fig.add_subplot(111)
    ax3.grid()
    cax3 = ax3.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,14], s=3, cmap=plt.get_cmap('seismic'), vmin=np.nanpercentile(pts_seed_stats[:,14], 10), vmax=np.nanpercentile(pts_seed_stats[:,14], 90), linewidth=0)
    ax3.set_title('Slope of fitted plane for each sphere (r=%0.2f)'%args.sphere_radius_m,y=1.05)
    cbar = fig.colorbar(cax3)
    cbar.set_label('Slope (m/m)')
    ax3.set_xlabel('UTM-X (m)')
    ax3.set_ylabel('UTM-Y (m)')
    ax3.axis('equal')
    fig.savefig(pcl_slope_pts_fig_overview_fn, bbox_inches='tight')
    plt.close()
print('done.')

sroughness9010p_pts_stats_png = '_sroughness9010p_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
pcl_sroughness9010p_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + sroughness9010p_pts_stats_png)
print('Generating figure for range off offsets from linear plane (90-10th p): %s... '%os.path.basename(pcl_sroughness9010p_pts_fig_overview_fn), end='', flush=True)
if os.path.exists(pcl_sroughness9010p_pts_fig_overview_fn) == False:
    fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
    fig.clf()
    
    ax4 = fig.add_subplot(111)
    ax4.grid()
    cax4 = ax4.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,12], s=3, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(pts_seed_stats[:,12], 10), vmax=np.nanpercentile(pts_seed_stats[:,12], 90), linewidth=0)
    ax4.set_title('Surface roughness I: Range of offsets from linear plane (90-10th) with r=%0.2f)'%args.sphere_radius_m,y=1.05)
    cbar = fig.colorbar(cax4)
    cbar.set_label('Range (90-10th) of plane offsets (m)')
    ax4.set_xlabel('UTM-X (m)')
    ax4.set_ylabel('UTM-Y (m)')
    ax4.axis('equal')
    fig.savefig(pcl_sroughness9010p_pts_fig_overview_fn, bbox_inches='tight')
    plt.close()
print('done.')

curv_pts_stats_png = '_curv_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
pcl_curv_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + curv_pts_stats_png)
print('Generating figure for range off offsets from linear plane (75-25th p): %s... '%os.path.basename(pcl_curv_pts_fig_overview_fn), end='', flush=True)
if os.path.exists(pcl_curv_pts_fig_overview_fn) == False:
    fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
    fig.clf()
    
    ax4 = fig.add_subplot(111)
    ax4.grid()
    cax4 = ax4.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,18], s=3, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(pts_seed_stats[:,13], 10), vmax=np.nanpercentile(pts_seed_stats[:,13], 90), linewidth=0)
    ax4.set_title('Curvature of sphere/disc with r=%0.2f)'%args.sphere_radius_m,y=1.05)
    cbar = fig.colorbar(cax4)
    cbar.set_label('Curvatre (1/m^2)')
    ax4.set_xlabel('UTM-X (m)')
    ax4.set_ylabel('UTM-Y (m)')
    ax4.axis('equal')
    fig.savefig(pcl_curv_pts_fig_overview_fn, bbox_inches='tight')
    plt.close()
print('done.')

stddev_pts_stats_png = '_stddev_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
pcl_stddev_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + stddev_pts_stats_png)
print('Generating figure for std. deviation of points from fitted plane: %s... '%os.path.basename(pcl_stddev_pts_fig_overview_fn), end='', flush=True)
if os.path.exists(pcl_stddev_pts_fig_overview_fn) == False:
    fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
    fig.clf()
    
    ax5 = fig.add_subplot(111)
    ax5.grid()
    cax5 = ax5.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,10], s=3, cmap=plt.get_cmap('jet'), vmin=np.nanpercentile(pts_seed_stats[:,10], 10), vmax=np.nanpercentile(pts_seed_stats[:,10], 90), linewidth=0)
    ax5.set_title('Surface roughness III: Std. deviation of lidar points from plane with r=%0.2f)'%args.sphere_radius_m,y=1.05)
    cbar = fig.colorbar(cax5)
    cbar.set_label('Std. deviation (m)')
    ax5.set_xlabel('UTM-X (m)')
    ax5.set_ylabel('UTM-Y (m)')
    ax5.axis('equal')
    fig.savefig(pcl_stddev_pts_fig_overview_fn, bbox_inches='tight')
    plt.close()
print('done.')

### Plotting commands: GRAVEYARD
### 3D plots, currently turned off
#    ax1 = fig.add_subplot(211, projection='3d')
#    pcl_points = ax1.scatter(dxyzn[:,0], dxyzn[:,1], dxyzn[:,2], c=dxyzn[:,3], s=dxyzn[:,3], marker='o')
#    cbar = fig.colorbar(pcl_points)
#    ax1.set_xlabel('UTM-X')
#    ax1.set_ylabel('UTM-Y')
#    ax1.set_zlabel('UTM-Z')
#    ax1.set_title('3D view of lidar point cloud: delta D (distance from plane to point) lidar points (colored)')
#    
#    ax2 = fig.add_subplot(212, projection='3d')
#    ax_seeds_range = ax2.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], pcl_xyzg_rstep_seed[:,2], c=pts_seed_stats[:,12], marker='o', s=np.exp((pts_seed_stats[:,12]/np.nanmax(pts_seed_stats[:,12]))*2))
#    cbar = fig.colorbar(ax_seeds_range)
#    cbar.set_label('Max-Min elevation in sphere (m)')
#    fig.colorbar
#    ax2.set_xlabel('UTM-X')
#    ax2.set_ylabel('UTM-Y')
#    ax2.set_zlabel('UTM-Z')
#    ax2.set_title('3D view of seed lidar points (sphere radius=%0.2f m) (colored) with range of point heights for each seed point (normalized)'%args.sphere_radius_m)
#    fig.savefig(pcl_seed_pts_fig_range_fn, bbox_inches='tight')
#    plt.close()


#Das mit der density habe ich wie folgt gemacht:
#
#def density(xy):
#    n = xy.shape[0]
#    p = np.zeros(n)
#    print('.. FLANN')
#    pyflann.set_distance_type('euclidean')
#    f = pyflann.FLANN()
#    k, d = f.nn(xy, xy, 51,
#            algorithm = 'kmeans', branching = 32, iterations = 7,
#checks = 16) del k
#    d = np.sqrt(d)
#    dmin = d.min(axis = 0)
#    dmin = dmin[-1]
#    print(dmin)
#    disk = np.pi * dmin * dmin
#    pb = pbar(maxval = n)
#    pb.start()
#    for i in range(n):
#        pb.update(i+1)
#        di = d[i]
#        p[i] = len(di[di <= dmin]) / disk
#
#    pb.finish()
#    return p

    ### Creating figure for every seed points - very time intensive and currently commented out    
        #calcuate z coordinate of seed point with respect to plane - only needed if creating figure
#z_seed_pt = (-pts_xyz_normal[0] * pcl_xyzg_rstep_seed[i,0] - pts_xyz_normal[1] * pcl_xyzg_rstep_seed[i,1] - d) * 1. /pts_xyz_normal[2]
#
##calculate mesh for plotting plane
#xx, yy = np.meshgrid(np.linspace(np.nanmin(pts_xyz[:,0]),np.nanmax(pts_xyz[:,0]), num=10), np.linspace(np.nanmin(pts_xyz[:,1]),np.nanmax(pts_xyz[:,1]), num=10))
#z = (-pts_xyz_normal[0] * xx - pts_xyz_normal[1] * yy - d) * 1. /pts_xyz_normal[2]
#
#fig = plt.figure(figsize=(16.53,11.69), dpi=300)
#fig.clf()
#ax1 = fig.add_subplot(211, projection='3d')
#ax1 = fig.add_subplot(111, projection='3d')
#pts_ax = ax1.scatter(pts_xyz[:,0], pts_xyz[:,1], pts_xyz[:,2], c=pts_xyz[:,2], marker='o')
#ax1.scatter(pcl_xyzg_rstep_seed[i,0], pcl_xyzg_rstep_seed[i,1], pcl_xyzg_rstep_seed[i,2], c='k', marker='x', s=250)
#ax1.plot_surface(xx, yy, z, alpha = 0.5, linewidth=0, antialiased=True, cstride=1)
#(xs, ys, zs) = drawSphere(pcl_xyzg_rstep_seed[i,0], pcl_xyzg_rstep_seed[i,1], pcl_xyzg_rstep_seed[i,2], args.sphere_radius_m)
#plt.grid('on')
#cbar = fig.colorbar(pts_ax)
#cbar.set_label('Point Elevation (m)')
#ax1.plot_wireframe(xs, ys, zs, color='k', linewidths=0.1)
#ax1.set_xlabel('UTM-X')
#ax1.set_ylabel('UTM-Y')
#ax1.set_zlabel('UTM-Z')
#ax1.set_title('3D view of selected lidar points (colored) and lidar seed point (black star) (ground only)')
#
#ax2 = fig.add_subplot(212, projection='3d')
#ax2.scatter(pts_xyz[:,0], pts_xyz[:,1], dz, c=dz, marker='o')
#ax2.scatter(pcl_xyzg_rstep_seed[i,0], pcl_xyzg_rstep_seed[i,1], pcl_xyzg_rstep_seed[i,2]-z_seed_pt, c='k', marker='x', s=250)
#ax2.set_xlabel('UTM-X')
#ax2.set_ylabel('UTM-Y')
#ax2.set_zlabel('UTM-Z')
#ax2.set_title('3D view of normalized lidar points (colored) and lidar seed point (black star) (ground only)')
#seed_fig_fname = 'seed_pts_fig_%d.png'%i     
#fig.savefig(seed_fig_fname, bbox_inches='tight')
#
## plot points and fitted surface
#fig = plt.figure()
#plt.clf()
#ax = fig.gca(projection='3d')
#ax.scatter(points[:,0], points[:,1], points[:,2], c='r', s=50)
#ax.plot_surface(X, Y, Z_order1, color='blue', alpha = 0.5, linewidth=0, antialiased=True, cstride=1)
#ax.plot_surface(X, Y, Z_order2, color='black', alpha = 0.5, linewidth=0, antialiased=True, cstride=1)
#plt.xlabel('X')
#plt.ylabel('Y')
#ax.set_zlabel('Z')
#ax.axis('equal')
#ax.axis('tight')
#plt.show()

