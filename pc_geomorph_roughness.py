#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:22:03 2017

@author: Bodo Bookhagen, Oct-Nov, 2017

"""

from laspy.file import File
import copy, glob, time, sys
import numpy as np, os, argparse, pickle, h5py, subprocess, gdal, osr, datetime
from numpy.linalg import svd
from pykdtree.kdtree import KDTree
from scipy import interpolate
from scipy import spatial
from scipy import linalg
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
from multiprocessing import Pool
from skimage import exposure

### Command Line Parsing
parser = argparse.ArgumentParser(description='PointCloud (PC) processing for geomorphologic purposes. Estimates slopes, curvature, local roughness, and other parameters. B. Bookhagen (bodo.bookhagen@uni-potsdam.de), Feb 2018.')
# Important and required:
parser.add_argument('--inlas', type=str, default='',  help='LAS/LAZ file with point-cloud data. Ideally, this file contains only ground points (class == 2)')
parser.add_argument('--raster_m', type=float, default=1,  help='Raster spacing for subsampling seed points on LAS/LAZ PC. Usually 0.5 to 2 m, default = 1.')
parser.add_argument('--sphere_radius_m', type=float, default=1.5,  help='Radius of sphere used for selecting lidar points around seed points. These points are used for range, roughness, and density calculations. Default radius 1.5m, i.e., points within a sphere of 3m are chosen.')
parser.add_argument('--slope_sphere_radius_m', type=float, default=0,  help='Radius of sphere used for fitting a linear plane and calculating slope and detrending data (slope normalization). By default this is similar to the radius used for calculation roughness indices (srd_m), but this can be set to a different value. For example, larger radii use the slope of larger area to detrend data.')
parser.add_argument('--dem_fname', type=str, default='',  help='Filename of DEM to extract point spacing. Used to identify seed-point coordinates')
parser.add_argument('--shapefile_clip', type=str, default='',  help='Name of shapefile to be used to clip interpolated surfaces too. This is likely the shapefile you have previously generated to subset/clip the point-cloud data.')
# Optional / additional parameters
parser.add_argument('--raw_pt_cloud', type=int, default=1,  help='Process raw point cloud (not homogenized) for seed-point statistcs (default=1).')
parser.add_argument('--nr_random_sampling', type=int, default=10,  help='Number of random-point cloud sampling iteration (how many iterations of bootstraping to estimate slope/curvature calculations).')
parser.add_argument('--range_radii', type=int, default=0,  help='Use a list of radii to calculate different length scales (i.e., iterate through different length scales to estimate dataset length scaling).')
parser.add_argument('-epsg', '--epsg_code', type=int, default=26911,  help='EPSG code (integer) to define projection information. This should be the same EPSG code as the input data (no re-projection included yet) and can be taken from LAS/LAZ input file. Add this to ensure that output shapefile and GeoTIFFs are properly geocoded.')
parser.add_argument('-o', '--outlas', type=str, default='',  help='LAS file to be created (currently no writing of LAZ files supported). This has the same dimension and number of points as the input LAS/LAZ file, but replaced color values reflecting roughness calculated over a given radius. Note that this will overwrite existing color information in the output file.')
#parser.add_argument('--shapefile_out', type=str, default='',  help='Output shapefile storing calculated attributes for seed points only. Default filename will be generated with radius in the filename.')
parser.add_argument('-odir', '--outputdir', type=str, default='',  help='Output directory to store plots and pickle files. Default is directory containing LAS/LAZ file.')
parser.add_argument('-fig', '--figure', type=bool, default=True,  help='Generate figures while processing. This often takes significant amount of time and can be turned off with -fig False.')
parser.add_argument('-color', '--store_color', type=bool, default=False,  help='Generate a LAS file where deviation from the plane are saved in the color attribute of the LAS file for every point. *Note* that this will replace the color information in the LAS file (but will be saved to separate file). Default is False, can be turned on with --store_color True.')
parser.add_argument('-cores', '--nr_of_cores', type=int, default=0,  help='Max. number of cores to use for multi-core processing. Default is to use all cores, set to --nr_of_cores 6 to use 6 cores.')
args = parser.parse_args()

#args.inlas = '/raid-cachi/bodo/Dropbox/California/SCI/Pozo/cat1/Pozo_USGS_UTM11_NAD83_all_color_cl_cat1.laz'
#args.shapefile_clip = '/raid-cachi/bodo/Dropbox/California/SCI/Pozo/shapefiles/Pozo_DTM_noveg_UTM11_NAD83_cat1_b50m.shp'
#args.dem_fname = '/raid-cachi/bodo/Dropbox/California/SCI/Pozo/cat1/Pozo_cat1_UTM11_NAD83_1m.tif'

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
    try:
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    except AssertionError:
        return np.nan, np.nan, np.nan
    
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    plane_normal = svd(M)[0][:,-1]
    d = -ctr.dot(plane_normal)
    z = (-plane_normal[0] * points[0,:] - plane_normal[1] * points[1,:] - d) * 1. /plane_normal[2]
    errors = z - points[2,:]
    residual = np.linalg.norm(errors)

    return ctr, plane_normal, residual

def curvFit_lstsq_polygon(points, order=2):
    """
    Fitting a second order polynom to a point cloud and deriving the curvature in a simplified form.
    More details: https://gis.stackexchange.com/questions/37066/how-to-calculate-terrain-curvature
    """
    
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    try:
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    except AssertionError:
        return np.nan, np.nan, np.nan
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
    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(points.shape[0]), points[:,:2], np.prod(points[:,:2], axis=1), points[:,:2]**2]
        C,_,_,_ = linalg.lstsq(A, points[:,2])
        
        # evaluate it on a grid
        Z_order2 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
        #Z = Dx² + Ey² + Fxy + Gx + Hy + I
        #Curvature = -2(D + E)
        #Slope = sqrt(G^2 + H ^2)
        curvature = -2 * (C[4] + C[5])
        slope = np.sqrt( C[1]**2 + C[2]**2 )

        Z_pts = np.dot(np.c_[np.ones(points.shape[0]), points[:,0], points[:,1], points[:,0]*points[:,1], points[:,0]**2, points[:,1]**2], C)
        errors = points[:,2] - Z_pts
        dZ_residuals = np.linalg.norm(errors)
    del A, C, Z_order2, Z_pts, errors
    return slope, curvature, dZ_residuals

def calc_pts_length_scale_multicore(ii):
    #setup multicore loop for all seed points
    #Setup array for seed point results:
    from_pos = pos_array[ii] #Get start/end from position array
    to_pos = pos_array[ii+1]
    subarr = np.arange(from_pos, to_pos) #Slice the data into the selected part...
    curv_scale_stats_results = np.empty((subarr.shape[0], nr_of_datasets_length_scale))
    curv_scale_stats_results.fill(np.nan)
    for j in range(subarr.shape[0]):
        pts_xyz = pcl_xyzg_rstep_seed[subarr[j]]
        pts_idx = points_k_idx[subarr[j]]
        pts = pcl_xyzg[pts_idx]
        pts_centroid = np.mean(pts, axis=0)
        #if pts > 5e4 (50'000), it's better to subsample/random sample
        d = spatial.distance.cdist(np.array([pts_xyz]), pts, 'euclidean')[0]
        d.sort()
        distances = np.hstack((np.array([d.shape[0], d[1::].min(), d.max(), d.mean(), np.std(d)]), np.percentile(d, [25, 50, 75])))
    
        A = np.c_[np.ones(pts.shape[0]), pts[:,:2], np.prod(pts[:,:2], axis=1), pts[:,:2]**2]
        C,_,_,_ = linalg.lstsq(A, pts[:,2])
        curvature = -2 * ( C[4] + C[5] )
        slope = np.sqrt( C[1]**2 + C[2]**2 )
        #evaluate fit:
        Z_pts = np.dot(np.c_[np.ones(pts.shape[0]), pts[:,0], pts[:,1], pts[:,0]*pts[:,1], pts[:,0]**2, pts[:,1]**2], C)
        errors = pts[:,2] - Z_pts
        dZ_residuals = np.linalg.norm(errors)
        curv_scale_stats_cat = np.concatenate((pts_centroid, distances, np.array([slope, curvature, dZ_residuals])))
        curv_scale_stats_results[j,:] = curv_scale_stats_cat
        del A, C, Z_pts, errors, d, pts, pts_idx, pts_xyz, curv_scale_stats_cat, pts_centroid

    pickle_fn = os.path.join(pickle_dir, 'PC_length_scale_{}.pickle'.format(str(ii).zfill(4)))
    pickle.dump((curv_scale_stats_results), open(pickle_fn,'wb'))
    if np.mod(ii,10) == 0:
        print('{}, '.format(str(ii).zfill(2)), end='', flush=True)
    del curv_scale_stats_results

def pcl_xyzg_p_slope_curvature_singlecore(ii, subarr):
    slope_lstsq = np.empty(subarr.shape[0])
    slope_lstsq.fill(np.nan)
    curvature_lstsq = np.empty(subarr.shape[0])
    curvature_lstsq.fill(np.nan)
    dZ_residuals_lstsq = np.empty(subarr.shape[0])
    dZ_residuals_lstsq.fill(np.nan)
    random_pts_xyz_normal = np.empty((subarr.shape[0], 3))
    random_pts_xyz_normal.fill(np.nan)
    slope_plane = np.empty(subarr.shape[0])
    slope_plane.fill(np.nan)

    for j in range(subarr.shape[0]):
        random_pts_xyz = pcl_xyzg_p_random[pcl_xyzg_p_random_seed_radius[subarr[j]],:].T
        slope_lstsq[j], curvature_lstsq[j], dZ_residuals_lstsq[j] = curvFit_lstsq_polygon(random_pts_xyz, order=2)
    
        random_pts_xyz_meanpt, random_pts_xyz_normal[j,:], plane_residual = planeFit(random_pts_xyz)
        slope_plane[j] = random_pts_xyz_normal[j,2]
        del random_pts_xyz
    return slope_lstsq, curvature_lstsq, dZ_residuals_lstsq, slope_plane


def pcl_xyzg_p_slope_curvature_multicore(ii):
    from_pos = pos_array[ii] #Get start/end from position array
    to_pos = pos_array[ii+1]
    subarr = np.arange(from_pos, to_pos) #Slice the data into the selected part...

    slope_lstsq = np.empty(subarr.shape[0])
    slope_lstsq.fill(np.nan)
    curvature_lstsq = np.empty(subarr.shape[0])
    curvature_lstsq.fill(np.nan)
    dZ_residuals_lstsq = np.empty(subarr.shape[0])
    dZ_residuals_lstsq.fill(np.nan)
    random_pts_xyz_normal = np.empty((subarr.shape[0], 3))
    random_pts_xyz_normal.fill(np.nan)
    slope_plane = np.empty(subarr.shape[0])
    slope_plane.fill(np.nan)

    for j in range(subarr.shape[0]):
        random_pts_xyz = pcl_xyzg_p_random[pcl_xyzg_p_random_seed_radius[subarr[j]],:].T
        slope_lstsq[j], curvature_lstsq[j], dZ_residuals_lstsq[j] = curvFit_lstsq_polygon(random_pts_xyz, order=2)
    
        random_pts_xyz_meanpt, random_pts_xyz_normal[j,:], plane_residual = planeFit(random_pts_xyz)
        slope_plane[j] = random_pts_xyz_normal[j,2]
        del random_pts_xyz
    return slope_lstsq, curvature_lstsq, dZ_residuals_lstsq, slope_plane


def calc_pts_length_scale_singlecore(ii, subarr):
    #setup multicore loop for all seed points
    #Setup array for seed point results:
    curv_scale_stats_results = np.empty((subarr.shape[0], nr_of_datasets_length_scale))
    curv_scale_stats_results.fill(np.nan)
    for j in range(subarr.shape[0]):
        pts_xyz = pcl_xyzg_rstep_seed[subarr[j]]
        pts_idx = points_k_idx[subarr[j]]
        pts = pcl_xyzg[pts_idx]
        pts_centroid = np.mean(pts, axis=0)
        #if pts > 5e4 (50'000), it's better to subsample/random sample
        d = spatial.distance.cdist(np.array([pts_xyz]), pts, 'euclidean')[0]
        d.sort()
        distances = np.hstack((np.array([d.shape[0], d[1::].min(), d.max(), d.mean(), np.std(d)]), np.percentile(d, [25, 50, 75])))
    
        A = np.c_[np.ones(pts.shape[0]), pts[:,:2], np.prod(pts[:,:2], axis=1), pts[:,:2]**2]
        C,_,_,_ = linalg.lstsq(A, pts[:,2])
        curvature = -2 * ( C[4] + C[5] )
        slope = np.sqrt( C[1]**2 + C[2]**2 )
        #evaluate fit:
        Z_pts = np.dot(np.c_[np.ones(pts.shape[0]), pts[:,0], pts[:,1], pts[:,0]*pts[:,1], pts[:,0]**2, pts[:,1]**2], C)
        errors = pts[:,2] - Z_pts
        dZ_residuals = np.linalg.norm(errors)
        curv_scale_stats_cat = np.concatenate((pts_centroid, distances, np.array([slope, curvature, dZ_residuals])))
        curv_scale_stats_results[j,:] = curv_scale_stats_cat
        del A, C, Z_pts, errors, d, pts, pts_idx, pts_xyz, curv_scale_stats_cat, pts_centroid

    pickle_fn = os.path.join(pickle_dir, 'PC_length_scale_{}.pickle'.format(str(ii).zfill(4)))
    pickle.dump((curv_scale_stats_results), open(pickle_fn,'wb'))
    if np.mod(ii,10) == 0:
        print('{}, '.format(str(ii).zfill(2)), end='', flush=True)
    del curv_scale_stats_results

def poisson_disk_sampling():
    # see here: http://devmag.org.za/2009/05/03/poisson-disk-sampling/
    #https://www.pdal.io/stages/filters.poisson.html
    #http://hhoppe.com/proj/thesis/
    #https://github.com/mkazhdan/PoissonRecon
    print('testing')
    
def plot_length_scales_map(pcl_xyzg_rstep_seed, curv_scale_stats, dist_range_list, distance_n = [0,5,10]):
    #Plotting Map view of mean distance of each queried point cloud
    for i in range(len(distance_n)):
        distance_n_current = distance_n[i]
        pts_distance_fig_fn = '_distances_%dm.png'%dist_range_list[distance_n_current]
        pts_distance_fig_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + pts_distance_fig_fn)
        fig = plt.figure(figsize=(16.53,11.69), dpi=150)
        fig.clf()
        ax1 = fig.add_subplot(231)
        cax1 = ax1.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=curv_scale_stats[distance_n_current,3,:], s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(curv_scale_stats[distance_n_current,3,:], 10), vmax=np.percentile(curv_scale_stats[distance_n_current,3,:], 90))
        cbar = fig.colorbar(cax1)
        cbar.set_label('Number of points')
        ax1.set_title('Number of points for each sphere for query of r=%0.2fm'%(dist_range_list[distance_n_current]),y=1.05)
        ax1.set_xlabel('UTM-X (m)')
        ax1.axis('equal')
        ax1.grid('on')
        
        ax2 = fig.add_subplot(232)
        cax2 = ax2.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=curv_scale_stats[distance_n_current,9,:], s=0.5, cmap=plt.get_cmap('PRGn_r'), linewidth=0, vmin=np.percentile(curv_scale_stats[distance_n_current,9,:], 10), vmax=np.percentile(curv_scale_stats[distance_n_current,9,:], 90))
        cbar = fig.colorbar(cax2)
        cbar.set_label('Distance (m)')
        ax2.set_title('Median distance of points for each sphere for query of r=%0.2fm'%(dist_range_list[distance_n_current]),y=1.05)
        ax2.set_xlabel('UTM-X (m)')
        ax2.axis('equal')
        ax2.grid('on')
    
        ax3 = fig.add_subplot(233)
        cax3 = ax3.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=curv_scale_stats[distance_n_current,10,:]-curv_scale_stats[distance_n_current,8,:], s=0.5, cmap=plt.get_cmap('spring'), linewidth=0, vmin=np.percentile(curv_scale_stats[distance_n_current,10,:]-curv_scale_stats[distance_n_current,8,:], 10), vmax=np.percentile(curv_scale_stats[distance_n_current,10,:]-curv_scale_stats[distance_n_current,8,:], 90))
        cbar = fig.colorbar(cax3)
        cbar.set_label('Range of 75-25th p distance (m)')
        ax3.set_title('Range of 75-25th perc. distance of points for each sphere for query of r=%0.2fm'%(dist_range_list[distance_n_current]),y=1.05)
        ax3.set_xlabel('UTM-X (m)')
        ax3.axis('equal')
        ax3.grid('on')
    
        ax4 = fig.add_subplot(234)
        cax4 = ax4.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=curv_scale_stats[distance_n_current,11,:], s=0.5, cmap=plt.get_cmap('seismic'), linewidth=0, vmin=np.percentile(curv_scale_stats[distance_n_current,11,:], 10), vmax=np.percentile(curv_scale_stats[distance_n_current,11,:], 90))
        cbar = fig.colorbar(cax4)
        cbar.set_label('Slope (m/m)')
        ax4.set_title('Slope of fitted sphere for query of r=%0.2fm'%(dist_range_list[distance_n_current]),y=1.05)
        ax4.set_xlabel('UTM-X (m)')
        ax4.set_ylabel('UTM-Y (m)')
        ax4.axis('equal')
        ax4.grid('on')
    
        ax5 = fig.add_subplot(235)
        cax5 = ax5.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=curv_scale_stats[distance_n_current,12,:], s=0.5, cmap=plt.get_cmap('PiYG'), linewidth=0,vmin=np.percentile(curv_scale_stats[distance_n_current,12,:], 10), vmax=np.percentile(curv_scale_stats[distance_n_current,12,:], 90))
        cbar = fig.colorbar(cax5)
        cbar.set_label('Curvature (1/m)')
        ax5.set_title('Curvature distance of points for each sphere for query of r=%0.2fm'%(dist_range_list[distance_n_current]),y=1.05)
        ax5.set_xlabel('UTM-X (m)')
        ax5.set_ylabel('UTM-Y (m)')
        ax5.axis('equal')
        ax5.grid('on')
    
        ax6 = fig.add_subplot(236)
        cax6 = ax6.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=curv_scale_stats[distance_n_current,13,:], s=0.5, cmap=plt.get_cmap('autumn_r'), linewidth=0, vmin=np.percentile(curv_scale_stats[distance_n_current,13,:], 10), vmax=np.percentile(curv_scale_stats[distance_n_current,13,:], 90))
        cbar = fig.colorbar(cax6)
        cbar.set_label('Residual (m)')
        ax6.set_title('Plane residual for each sphere for query of r=%0.2fm'%(dist_range_list[distance_n_current]),y=1.05)
        ax6.set_xlabel('UTM-X (m)')
        ax6.set_ylabel('UTM-Y (m)')
        ax6.axis('equal')
        ax6.grid('on')
    
        fig.savefig(pts_distance_fig_fn, bbox_inches='tight')
        plt.close()

def plot_length_scales_comparison():
    fig = plt.figure(figsize=(16.53,11.69), dpi=150)
    fig.clf()
    ax1 = fig.add_subplot(231)
    pt_density = curv_scale_stats[0,3,:] / (curv_scale_stats[0,9,:]**2*np.pi)
    cax1 = ax1.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], s=0.75, c=pt_density, vmin=np.percentile(pt_density, 10), vmax=np.percentile(pt_density, 90))
    cbar=fig.colorbar(cax1)
    cbar.set_label('Point density (pts/m^2)')
    ax1.set_xlabel('UTM-X')
    ax1.set_ylabel('UTM_Y')
    ax1.set_title('Point density for sphere = 1m',y=1.05)

    ax2 = fig.add_subplot(232)
    ax2.plot(curv_scale_stats[:,9,:], curv_scale_stats[:,3,:] / (curv_scale_stats[:,6,:]**2*np.pi), '.', markersize=0.2, color='gray', label='values')
    ax2.plot(np.nanmedian(curv_scale_stats[:,9,:], axis=1), np.nanmedian(curv_scale_stats[:,3,:] / (curv_scale_stats[:,6,:]**2*np.pi), axis=1), 'bo', label='median')
    
    ax2.set_xlabel('Median distance (m)')
    ax2.set_ylabel('Point density (pts/m^2)')
    ax2.set_title('Median distance and point density ',y=1.05)
    ax2.grid('on')
#import numpy as np
#import seaborn as sns
#data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
#sns.set_style('whitegrid')
#sns.kdeplot(np.array(data), bw=0.5)

#see https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python for 2D KDE plot

def plot_length_scales_graph(pcl_xyzg_rstep_seed, curv_scale_stats, dist_range_list, xy_point_nr_list = [0,5000,10000]):

    #Plot functional relations for all divergent parts
    curv_scale_stats_divergent75p_median = np.empty(curv_scale_stats.shape[0])
    curv_scale_stats_divergent75p_median.fill(np.nan)
    curv_scale_stats_divergent75p_std = np.empty(curv_scale_stats.shape[0])
    curv_scale_stats_divergent75p_std.fill(np.nan)
    for i in range(curv_scale_stats.shape[0]):
        idx_divergent = np.where(curv_scale_stats[i,12,:] > 0.001)[0] #positive curvature, hillslopes
        idx_divergent75p = np.where(curv_scale_stats[i,12,idx_divergent] > np.percentile(curv_scale_stats[i,12,idx_divergent], 75))[0]
        idx_divergent75p = idx_divergent[idx_divergent75p]
        curv_scale_stats_divergent75p_median[i] = np.nanmedian(curv_scale_stats[i,12,idx_divergent75p])
        curv_scale_stats_divergent75p_std[i] = np.nanstd(curv_scale_stats[i,12,idx_divergent75p])
        
    idx_divergent90p = np.where(curv_scale_stats[0,12,idx_divergent] > np.percentile(curv_scale_stats[0,12,idx_divergent], 90))[0]
    idx_divergent90p = idx_divergent[idx_divergent90p]

    fig = plt.figure(figsize=(16.53,11.69), dpi=150)
    fig.clf()
    ax1 = fig.add_subplot(231)



    ax1.plot(np.nanmedian(curv_scale_stats[:,9,idx_divergent90p], axis=1), curv_scale_stats_divergent75p_std, 'o-')
    ax1.set_xlabel('Median distance (m)')
    ax1.set_ylabel('Median curvature (1/m)')
    ax1.set_title('Median distance and curvature of 90th percentile divergent curvature for all k distances',y=1.05)
    ax1.grid('on')
    
    idx_convergent = np.where(curv_scale_stats[0,12,:] < -0.001)[0] #negative curvature, channels
    ax2 = fig.add_subplot(232)
    ax2.plot(curv_scale_stats[:,9,idx_convergent[100]], curv_scale_stats[:,12,idx_convergent[100]], 'bo')
    
    
    ax2.plot(curv_scale_stats[:,9,idx_divergent], curv_scale_stats[:,12,idx_divergent], 'r+')
    
    ax2 = fig.add_subplot(232)
    ax2.plot(np.nanmedian(curv_scale_stats[:,9,:], axis=1), np.nanmean(curv_scale_stats[:,12,:], axis=1), 'bo-')
    ax2.plot(np.nanmedian(curv_scale_stats[:,9,:], axis=1), np.nanmean(curv_scale_stats[:,12,:], axis=1)+np.nanstd(curv_scale_stats[:,12,:], axis=1), 'b-')
    ax2.plot(np.nanmedian(curv_scale_stats[:,9,:], axis=1), np.nanmean(curv_scale_stats[:,12,:], axis=1)-np.nanstd(curv_scale_stats[:,12,:], axis=1), 'b-')
    ax2.set_xlabel('Median distance (m)')
    ax2.set_ylabel('std. deviation of curvature (1/m)')
    ax2.set_title('Median distance and mean curvature of all point',y=1.05)
    ax2.grid('on')

    ax3 = fig.add_subplot(233)
    ax3.plot(np.nanmedian(curv_scale_stats[:,9,:], axis=1), np.nanmean(curv_scale_stats[:,11,:], axis=1), 'bo-')
    ax3.plot(np.nanmedian(curv_scale_stats[:,9,:], axis=1), np.nanmean(curv_scale_stats[:,11,:], axis=1)+np.nanstd(curv_scale_stats[:,11,:], axis=1), 'b-')
    ax3.plot(np.nanmedian(curv_scale_stats[:,9,:], axis=1), np.nanmean(curv_scale_stats[:,11,:], axis=1)-np.nanstd(curv_scale_stats[:,11,:], axis=1), 'b-')
    ax3.set_xlabel('Median distance (m)')
    ax3.set_ylabel('Median slope (1/m)')
    ax3.set_title('Median distance and median slope of all point',y=1.05)
    ax3.grid('on')

    ax3 = fig.add_subplot(234)
    ax3.plot(np.nanmedian(curv_scale_stats[:,6,:], axis=1), np.nanmedian(curv_scale_stats[:,3,:], axis=1), 'bo', label='median')
    ax3.plot(np.nanmedian(curv_scale_stats[:,6,:], axis=1), np.nanmean(curv_scale_stats[:,3,:], axis=1), 'g+', label='mean')
    ax3.plot(np.nanmedian(curv_scale_stats[:,6,:], axis=1), np.nanmax(curv_scale_stats[:,3,:], axis=1), 'r.', label='max')
    ax3.set_xlabel('Median distance (m)')
    ax3.set_ylabel('Median/Mean/Max number of measurements')
    ax3.set_title('Median distance and number of neighboring points',y=1.05)
    ax3.legend()
    ax3.grid('on')

    ax4 = fig.add_subplot(234)
    ax4.plot(np.nanmedian(curv_scale_stats[:,6,:], axis=1), np.nanmedian(curv_scale_stats[:,13,:], axis=1), 'bo', label='median')
    ax4.plot(np.nanmedian(curv_scale_stats[:,6,:], axis=1), np.nanmean(curv_scale_stats[:,13,:], axis=1), 'g+', label='mean')
    ax4.plot(np.nanmedian(curv_scale_stats[:,6,:], axis=1), np.nanmax(curv_scale_stats[:,13,:], axis=1), 'r.', label='max')
    ax4.plot(np.nanmedian(curv_scale_stats[:,6,:], axis=1), np.nanmin(curv_scale_stats[:,13,:], axis=1), 'k.', label='min')
    ax4.set_xlabel('Median distance (m)')
    ax4.set_ylabel('Median/Mean/Max/Min residuals (m)')
    ax4.set_title('Median distance and residuals of plane fit',y=1.05)
    ax4.legend()
    ax4.grid('on')

def gaussian_curvature(Z):
    #Gaussian curvature only for gridded data
    Zy, Zx = np.gradient(Z)                                                     
    Zxy, Zxx = np.gradient(Zx)                                                  
    Zyy, _ = np.gradient(Zy)                                                    
    K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2             
    return K


def griddata_clip_geotif(tif_fname, points, data2i, xxyy, ncols, nrows, geotransform):
    #interpolate point to a gridded dataset using interpolate.griddata and nearest neighbor interpolation. Next, data will be clipped by shapefile to remove potential interpolation artifacts
    #sample call:
    #griddata_clip_geotif(nrlidari_tif_fn, points, pts_seed_stats[idx_nonan,17][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    datai = interpolate.griddata(points, data2i, xxyy, method='nearest')
    output_raster = gdal.GetDriverByName('GTiff').Create(tif_fname,ncols, nrows, 1 ,gdal.GDT_Float32,['TFW=YES', 'COMPRESS=DEFLATE', 'ZLEVEL=9'])  # Open the file, see here for information about compression: http://gis.stackexchange.com/questions/1104/should-gdal-be-set-to-produce-geotiff-files-with-compression-which-algorithm-sh
    output_raster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(args.epsg_code)
    output_raster.SetProjection( srs.ExportToWkt() )
    output_raster.GetRasterBand(1).WriteArray(datai) 
    output_raster.FlushCache()
    output_raster=None
    if os.path.exists(args.shapefile_clip) == False:
        print('Shapefile does not exist: %s'%args.shapefile_clip)
        exit()
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

def pc_density(pts_xyz, pcl_xyzg_ckdtree, nn=51, show_density_information=0):
    '''
    Takes the seed-point PC and uses the KDTree of the entire pointcloud to calculate statistics. 
    In this approach, the point density is only calculated for the points given in pts_xyz. One
    can also feed the entire point cloud to get the point density for every point.
    Number of neighbors are the max. number of points in a 1.5m neighborhood.
    
    For lidar seed point call with:
    p_min, p_median, density_min, density_median = pc_density(pcl_xyzg_rstep_seed, pcl_xyzg_ckdtree, nn=nn=int(dxyzn_mean_nre))
    
    For all lidar points, call with:
    p_min, p_median, density_min, density_median = pc_density(pcl_xyzg, pcl_xyzg_ckdtree, nn=dxyzn_max_nre)
    '''

    n = pts_xyz.shape[0]
    density_min = np.zeros(n)
    density_median = np.zeros(n)
    #density_max = np.zeros(n)
    p_min = np.zeros(n)
    p_median = np.zeros(n)
    d,k = pcl_xyzg_ckdtree.query(pts_xyz, k=nn, p=2, n_jobs=-1)
    #euclidean distance p=2
    del k

    #remove first number from array, because it is mostly 0 or the seed point itself
    #d = d[:,1::]
    #d = np.sqrt(d)
    
    #Get minimum distances for all neighbors and use the minim distance of the last element (largest d)
    #Could improve this by using an adaptive kernel to adjust for number of nearest neighbor for each seed point
    dmin = d.min(axis = 0)
    dmin = dmin[-1] #select last element, 
    
    #Repeat for median distances
    dmedian = np.median(d, axis = 0)
    dmedian = dmedian[-1]
        
    #Repeat for max distances
    #dmax = np.max(d, axis = 0)
    #dmax = dmax[-1]

    #calculate area, assuming min/median distance is a radius
    disk_min = np.pi * dmin * dmin
    disk_median = np.pi * dmedian * dmedian
    #disk_max = np.pi * dmax * dmax
    
    #Calculate density and probability for each seed point
    for i in range(n):
        di = d[i]
        density_min[i] = len(di[di <= dmin]) / disk_min
        density_median[i] = len(di[di <= dmedian]) / disk_median
        #density_max[i] = len(di[di <= dmax]) / disk_max
        #probability is the inverse of the density
        if len(di[di <= dmin]) < 1:
            print('no point in minimum distance (i=%d)'%i)
            p_min[i] = 0
        else:
            p_min[i] = disk_min / len(di[di <= dmin])
        p_median[i] = disk_median / len(di[di <= dmedian])
    
    #normalize probabilities by their sum
    p_min /= p_min.sum()
    p_median /= p_median.sum()
    
    #probability-based subsampling    
    if show_density_information == 1:
        print('PC density: Find %d nearest neighbors from each of %s seed points with a total of %s points.'%(nn,"{:,}".format(pts_xyz.shape[0]), "{:,}".format(pcl_xyzg_ckdtree.n)))
        print('PC density: min distance: %0.3fm, median distance: %0.3fm'%(dmin, dmedian))
        print('PC density: Average point density for min.   distance (pts/m^2): %0.3f pts/m^2'%(np.mean(density_min)))
        print('PC density: Average point density for median distance (pts/m^2): %0.3f pts/m^2'%(np.mean(density_median)))
        #print('PC density: Average point density for max.   distance (pts/m^2): %0.3f pts/m^2'%(np.mean(density_max)))
        print('PC density: Standard deviation of avg. point density from median distances (pts/m^2): %0.3f pts/m^2'%(np.std(density_median)))

    del d, disk_min, disk_median, dmedian, dmin, n, pts_xyz
    return p_min, p_median, density_min, density_median


def pc_random_p_subsampling(pts_xyz, p, nr_of_points):
    '''
    Sub-samples indices of PC pcl_xyzg_radius with probability weight p based 
    on point density of each point. Will result in greatly reduced point cloud. 
    Give nr_of_points for subsampled point cloud, usually len(p)/2
    
    call with a probability
    #pcl_xyzg_radius_equal_nr_random = pc_random_p_subsampling(pcl_xyzg_radius, pts_xyz, nr_of_points)
    '''
    
    #iterate through n number of points (length of seed  points)
    n = len(p)
    if pts_xyz.shape[1] > 3:
        pcl_xyzg_p_random = np.empty((int(nr_of_points),4))
    elif pts_xyz.shape[1] == 3:
        pcl_xyzg_p_random = np.empty((int(nr_of_points),3))
    pcl_xyzg_p_random.fill(np.nan)
    i = np.random.choice(n, size = int(nr_of_points), replace = False, p = p)
    pcl_xyzg_p_random[:,0] = pts_xyz[i,0]
    pcl_xyzg_p_random[:,1] = pts_xyz[i,1]
    pcl_xyzg_p_random[:,2] = pts_xyz[i,2]
    if pts_xyz.shape[1] > 3:
        pcl_xyzg_p_random[:,3] = pts_xyz[i,3]
    return pcl_xyzg_p_random

def pc_random_p_intensity_subsampling(pts_xyzi, p, nr_of_points):
    '''
    Sub-samples indices of PC pcl_xyzg_radius with probability weight p based 
    on point density of each point. Will result in greatly reduced point cloud. 
    Give nr_of_points for subsampled point cloud, usually len(p)/2
    
    call with a probability
    #pcl_xyzg_radius_equal_nr_random = pc_random_p_subsampling(pcl_xyzg_radius, pts_xyz, nr_of_points)
    '''
    
    #iterate through n number of points (length of seed  points)
    n = len(p)
    if pts_xyzi.shape[1] > 3:
        pcl_xyzg_p_random = np.empty((int(nr_of_points),4))
    elif pts_xyzi.shape[1] == 3:
        pcl_xyzg_p_random = np.empty((int(nr_of_points),3))
    pcl_xyzg_p_random.fill(np.nan)
    i = np.random.choice(n, size = int(nr_of_points), replace = False, p = p)
    pcl_xyzg_p_random[:,0] = pts_xyzi[i,0]
    pcl_xyzg_p_random[:,1] = pts_xyzi[i,1]
    pcl_xyzg_p_random[:,2] = pts_xyzi[i,2]
    pcl_xyzg_p_random[:,3] = pts_xyzi[i,3]
    if pts_xyzi.shape[1] > 3:
        pcl_xyzg_p_random[:,3] = pts_xyzi[i,3]
    return pcl_xyzg_p_random

def plt_point_cloud_densities(pcl_xyzg, pcl_xyzg_rstep_seed,
                              pcl_xyzg_p_random, pcl_xyzg_p_min,
                              pcl_xyzg_density_min, pcl_random_density_min, pcl_densities= '_pcl_densities_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)):
    pcl_densities_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + pcl_densities)
    
    fig = plt.figure(figsize=(16.53,11.69), dpi=150)
    fig.clf()
    ax1 = fig.add_subplot(221)
    cax1 = ax1.scatter(pcl_xyzg[:,0], pcl_xyzg[:,1], c=pcl_xyzg_density_min, s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(pcl_xyzg_density_min, 10), vmax=np.percentile(pcl_xyzg_density_min, 90))
    cbar = fig.colorbar(cax1)
    cbar.set_label('Point density (pts/m^2)')
    ax1.set_title('Point density for full point cloud \nwith nr=%s points.'%"{:,}".format(pcl_xyzg.shape[0]),y=1.05)
    #ax1.set_xlabel('UTM-X (m)')
    ax1.set_ylabel('UTM-Y (m)')
    ax1.axis('equal')
    ax1.grid('on')

    ax2 = fig.add_subplot(222)
    cax2 = ax2.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=pcl_xyzg_rstep_seed_density_min, s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(pcl_xyzg_density_min, 10), vmax=np.percentile(pcl_xyzg_density_min, 90))
    cbar = fig.colorbar(cax2)
    cbar.set_label('Point density (pts/m^2)')
    ax2.set_title('Point density for equal-spaced PC \nevery %0.2fm (r=%0.2fm), total: %s points.'%(args.raster_m, args.sphere_radius_m, "{:,}".format(pcl_xyzg_rstep_seed.shape[0])),y=1.05)
    #ax2.set_xlabel('UTM-X (m)')
    #ax2.set_ylabel('UTM-Y (m)')
    ax2.axis('equal')
    ax2.grid('on')

    ax3 = fig.add_subplot(223)
    cax3 = ax3.scatter(pcl_xyzg_p_random[:,0], pcl_xyzg_p_random[:,1], c=pcl_random_density_min, s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(pcl_xyzg_density_min, 10), vmax=np.percentile(pcl_xyzg_density_min, 90))
    cbar = fig.colorbar(cax3)
    cbar.set_label('Point density (pts/m^2)')
    ax3.set_title('Point density for p-random point cloud \nwith avg. density = %0.2f and nr=%s points.'%(np.mean(pcl_random_density_min), "{:,}".format(pcl_xyzg_p_random.shape[0])),y=1.05)
    #ax3.set_xlabel('UTM-X (m)')
    #ax3.set_ylabel('UTM-Y (m)')
    ax3.axis('equal')
    ax3.grid('on')

    ax6 = fig.add_subplot(224)
    cax6 = ax6.scatter(pcl_xyzg_p_random[:,0], pcl_xyzg_p_random[:,1], c=pcl_random_density_min, s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(pcl_random_density_min, 10), vmax=np.percentile(pcl_random_density_min, 90))
    cbar = fig.colorbar(cax6)
    cbar.set_label('Point density (pts/m^2)')
    ax6.set_title('Point density for p-random point cloud \nwith avg. density = %0.2f and nr=%s points.'%(np.mean(pcl_random_density_min), "{:,}".format(pcl_xyzg_p_random.shape[0])),y=1.05)
    ax6.set_xlabel('UTM-X (m)')
    #ax6.set_ylabel('UTM-Y (m)')
    ax6.axis('equal')
    ax6.grid('on')
    fig.savefig(pcl_densities_overview_fn, bbox_inches='tight')
    plt.close()

  
def plt_ensemble_slope_curvature():
    #plot figure with std. dev and variance
    ensemble_densities= '_pcl_ensembles_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
    pcl_random_ensembles_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + ensemble_densities)
    fig = plt.figure(figsize=(16.53,11.69), dpi=150)
    fig.clf()

    ax1 = fig.add_subplot(231)
    cax1 = ax1.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=pc_random_density_min_res[1,:], s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(pcl_random_density_min[1,:], 10), vmax=np.percentile(pcl_random_density_min[1,:], 90))
    cbar = fig.colorbar(cax1)
    cbar.set_label('Point density (ts/m^2)')
    ax1.set_title('Point Density of last ensemble for equal-spaced PC \nevery %0.2fm (r=%0.2fm), total: %s points.'%(args.raster_m, args.sphere_radius_m, "{:,}".format(pcl_xyzg_rstep_seed.shape[0])),y=1.05)
    #ax2.set_xlabel('UTM-X (m)')
    ax1.set_ylabel('UTM-Y (m)')
    ax1.axis('equal')
    ax1.grid('on')

    ax4 = fig.add_subplot(234)
    cax4 = ax4.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=pc_random_density_min_res[2,:], s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(pcl_random_density_min_res[2,:], 10), vmax=np.percentile(pcl_random_density_min_res[2,:], 90))
    cbar = fig.colorbar(cax4)
    cbar.set_label('std. dev. of point density (pts/m^2)')
    ax4.set_title('Std. Dev. of point density (n=%02d ensembles) for equal-spaced PC \nevery %0.2fm (r=%0.2fm), total: %s points.'%(args.nr_of_bootstraps, args.raster_m, args.sphere_radius_m, "{:,}".format(pcl_xyzg_rstep_seed.shape[0])),y=1.05)
    ax4.set_xlabel('UTM-X (m)')
    ax4.set_ylabel('UTM-Y (m)')
    ax4.axis('equal')
    ax4.grid('on')

    ax2 = fig.add_subplot(232)
    cax2 = ax2.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=slope_plane_stats[1,:], s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(slope_plane_stats[1,:], 10), vmax=np.percentile(slope_plane_stats[1,:], 90))
    cbar = fig.colorbar(cax2)
    cbar.set_label('Slope (m/m)')
    ax2.set_title('Mean slope (n=%02d ensembles) for equal-spaced PC \nevery %0.2fm (r=%0.2fm), total: %s points.'%(nr_of_bootstraps, args.raster_m, args.sphere_radius_m, "{:,}".format(pcl_xyzg_rstep_seed.shape[0])),y=1.05)
    #ax2.set_xlabel('UTM-X (m)')
    #ax2.set_ylabel('UTM-Y (m)')
    ax2.axis('equal')
    ax2.grid('on')

    ax5 = fig.add_subplot(235)
    cax5 = ax5.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=slope_plane_stats[2,:], s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(slope_plane_stats[2,:], 10), vmax=np.percentile(slope_lstsq_stats[2,:], 90))
    cbar = fig.colorbar(cax5)
    cbar.set_label('Std. Dev. of slope (m/m)')
    ax5.set_title('Std. dev. of slope (n=%02d ensembles) for equal-spaced PC \nevery %0.2fm (r=%0.2fm), total: %s points.'%(nr_of_bootstraps, args.raster_m, args.sphere_radius_m, "{:,}".format(pcl_xyzg_rstep_seed.shape[0])),y=1.05)
    #ax5.set_xlabel('UTM-X (m)')
    #ax5.set_ylabel('UTM-Y (m)')
    ax5.axis('equal')
    ax5.grid('on')

    ax3 = fig.add_subplot(233)
    cax3 = ax3.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=curvature_lstsq_stats[1,:], s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(curvature_lstsq_stats[1,:], 10), vmax=np.percentile(curvature_lstsq_stats[1,:], 90))
    cbar = fig.colorbar(cax3)
    cbar.set_label('Mean curvature (1/m)')
    ax3.set_title('Mean curvature (n=%02d ensembles) for equal-spaced PC \nevery %0.2fm (r=%0.2fm), total: %s points.'%(nr_of_bootstraps, args.raster_m, args.sphere_radius_m, "{:,}".format(pcl_xyzg_rstep_seed.shape[0])),y=1.05)
    #ax3.set_xlabel('UTM-X (m)')
    #ax3.set_ylabel('UTM-Y (m)')
    ax3.axis('equal')
    ax3.grid('on')

    ax6 = fig.add_subplot(236)
    cax6 = ax6.scatter(pcl_xyzg_rstep_seed[:,0], pcl_xyzg_rstep_seed[:,1], c=curvature_lstsq_stats[2,:], s=0.5, cmap=plt.get_cmap('jet'), linewidth=0, vmin=np.percentile(curvature_lstsq_stats[2,:], 10), vmax=np.percentile(curvature_lstsq_stats[2,:], 90))
    cbar = fig.colorbar(cax6)
    cbar.set_label('Std. Dev. of curvature (1/m)')
    ax6.set_title('Std. dev. of curvature (n=%02d ensembles) for equal-spaced PC \nevery %0.2fm (r=%0.2fm), total: %s points.'%(nr_of_bootstraps, args.raster_m, args.sphere_radius_m, "{:,}".format(pcl_xyzg_rstep_seed.shape[0])),y=1.05)
    #ax6.set_xlabel('UTM-X (m)')
    #ax6.set_ylabel('UTM-Y (m)')
    ax6.axis('equal')
    ax6.grid('on')

    fig.savefig(pcl_random_ensembles_overview_fn, bbox_inches='tight')
    plt.close()


def pc_random_equal_subsampling(pcl_xyzg_radius, pts_xyz, nr_of_points):
    '''
    Sub-samples indices of PC pcl_xyzg_radius with point locations pts_xyz based 
    with an equal number of points nr_of_points. nr_of_points can be calculated
    as an average point density.
    
    call with a probability
    #pcl_xyzg_radius_equal_nr_random = pc_random_equal_subsampling(pcl_xyzg_radius, pts_xyz, nr_of_points)
    '''
    
    #iterate through n number of points (length of seed  points)
    n = len(pcl_xyzg_radius)
    pcl_xyzg_radius_random = np.empty((n,int(nr_of_points),3))
    pcl_xyzg_radius_random.fill(np.nan)
    
    #find indices for each seed point and randomly select points based on probability p
    for i in range(n):
        current_ptx_idx = np.array(pcl_xyzg_radius[i]) #get indices for current point
        #Use the following if you have varying number of points
        #random_pt_idx = np.random.choice(current_ptx_idx, size = int(nr_of_points[i]), replace = True)
        random_pt_idx = np.random.choice(current_ptx_idx, size = int(nr_of_points), replace = True)
        
        #pcl_xyzg_radius_random contains the same number of points for each seed point
        pcl_xyzg_radius_random[i,:,:] = np.array([pcl_xyzg[random_pt_idx,0], pcl_xyzg[random_pt_idx,1], pcl_xyzg[random_pt_idx,2]]).T
        del current_ptx_idx
    return pcl_xyzg_radius_random


def calc_stats_for_bootstrap_seed_points_wrapper(i):
    #print('starting {}/{}'.format(i+1, len(pos_array)))
    from_pos = pos_array[i] #Get start/end from position array
    to_pos = pos_array[i+1]
    #Setup array for seed point results:
    subarr = np.arange(from_pos,to_pos) #Slice the data into the selected part...
    pts_seed_stats_result = np.empty((subarr.shape[0], nr_of_datasets))
    
    #Setup array for PC results (X, Y, Z, Dz)
    dxyzn_subarr_result = np.empty((subarr.shape[0], dxyzn_max_nre, 4))
 
    for ii in range(subarr.shape[0]):
        pts_seed_stats_result[ii,:], dxyzn_subarr_result[ii,:,:] = calc_stats_for_bootstrap_seed_points(subarr[ii]) #Run point cloud processing for this index

    pickle_fn = os.path.join(pickle_dir, 'PC_seed_points_{}.pickle'.format(str(i).zfill(4)))
    pickle.dump((pts_seed_stats_result, dxyzn_subarr_result), open(pickle_fn,'wb'))
    if np.mod(i,10) == 0:
        print('{}, '.format(str(i).zfill(2)), end='', flush=True)
    pts_seed_stats_result = None
    dxyzn_subarr_result = None
        
def calc_stats_for_bootstrap_seed_points(i):
    pts_xyz = pcl_xyzg_p_random[pcl_xyzg_equal_density_radius[i],0:3]
    #pts_xyz_slope = pcl_xyzg[pcl_xyzg_p_random_seed_radius_slope[i]]
    pts_xyz_slope = pcl_xyzg_p_random[pcl_xyzg_equal_density_radius_slope[i],0:3]
    #getting intensity values
    pts_xyzi = pcl_xyzg_p_random[pcl_xyzg_equal_density_radius[i],3]
    
    nr_pts_xyz = pts_xyz.shape[0]
    if pts_xyz.shape[0] < 5:
        #print('Less than 5 points, plane fitting not meaningful for i = %s'%"{:,}".format(i))
        pts_xyz_meanpt = np.nan
        pts_xyz_normal = np.nan
        pts_seed_stats = np.array([pcl_xyzg_rstep_seed[i,0], pcl_xyzg_rstep_seed[i,1], pcl_xyzg_rstep_seed[i,2], 
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                   np.nan, np.nan, np.nan, np.nan, nr_pts_xyz, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        dxyzn = np.empty((dxyzn_max_nre, 4))
        dxyzn.fill(np.nan)
    else:
        pts_xyz_meanpt, pts_xyz_normal, plane_residual = planeFit(pts_xyz_slope.T)
        #residual calculated from = np.linalg.norm(errors)

        #calculate curvature
        slope_lstsq, curvature_lstsq, curv_residuals = curvFit_lstsq_polygon(pts_xyz_slope.T, order=2)

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
    
        #calculate intensity statistics for this seed point
        i_mean = np.mean(pts_xyzi)
        i_stddev = np.std(pts_xyzi)
        i_median = np.median(pts_xyzi)
        i_10_25_75_90p = np.percentile(pts_xyzi, [10, 25, 75, 90])
        
        #for each seed point, store relevant point statistics. Columns are:
        #0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  10: Dz-std.dev, 11: Dz-range, \
        #12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: slope of fitted plane, 15: plane residuals, \
        #16: variance dz, 17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals, \
        #21:intensity mean, 22:intensity std., 23: intensity median, 24: 10, 25: 27, 26: 75, 27: 90th percentile intensity
        pts_seed_stats = np.array([pcl_xyzg_rstep_seed[i,0], pcl_xyzg_rstep_seed[i,1], pcl_xyzg_rstep_seed[i,2], 
                       pts_xyz_meanpt[0], pts_xyz_meanpt[1], pts_xyz_meanpt[2], 
                       np.min(pts_xyz, axis=0)[2], np.max(pts_xyz, axis=0)[2], dz.max(), dz.min(), np.std(dz), dz.max()-dz.min(), \
                       np.percentile(dz, 90)-np.percentile(dz,10), np.percentile(dz, 75)-np.percentile(dz,25), plane_slope, plane_residual, \
                       np.var(dz), nr_pts_xyz, slope_lstsq, curvature_lstsq, curv_residuals, i_mean, i_stddev, i_median, i_10_25_75_90p[0], i_10_25_75_90p[1], i_10_25_75_90p[2], i_10_25_75_90p[3],])
    return pts_seed_stats, dxyzn

def calc_stats_for_seed_points_wrapper(i):
    #print('starting {}/{}'.format(i+1, len(pos_array)))
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
    if np.mod(i,10) == 0:
        print('{}, '.format(str(i).zfill(2)), end='', flush=True)
    pts_seed_stats_result = None
    dxyzn_subarr_result = None
        
def calc_stats_for_seed_points(i):
    pts_xyz = pcl_xyzg[pcl_xyzg_radius[i]]
    pts_xyz_slope = pcl_xyzg[pcl_xyzg_radius_slope[i]]
    #getting intensity values
    pts_xyzi = pcl_xyzig[pcl_xyzg_radius[i],3]
        
    nr_pts_xyz = pts_xyz.shape[0]
    if pts_xyz.shape[0] < 5:
        #print('Less than 5 points, plane fitting not meaningful for i = %s'%"{:,}".format(i))
        pts_xyz_meanpt = np.nan
        pts_xyz_normal = np.nan
        pts_seed_stats = np.array([pcl_xyzg_rstep_seed[i,0], pcl_xyzg_rstep_seed[i,1], pcl_xyzg_rstep_seed[i,2], 
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                   np.nan, np.nan, np.nan, np.nan, nr_pts_xyz, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        dxyzn = np.empty((dxyzn_max_nre, 4))
        dxyzn.fill(np.nan)
    else:
        pts_xyz_meanpt, pts_xyz_normal, plane_residual = planeFit(pts_xyz_slope.T)
        #residual calculated from = np.linalg.norm(errors)

        #calculate curvature
        slope_lstsq, curvature_lstsq, curv_residuals = curvFit_lstsq_polygon(pts_xyz_slope.T, order=2)

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
    
        #calculate intensity statistics for this seed point
        i_mean = np.mean(pts_xyzi)
        i_stddev = np.std(pts_xyzi)
        i_median = np.median(pts_xyzi)
        i_10_25_75_90p = np.percentile(pts_xyzi, [10, 25, 75, 90])
        
        #for each seed point, store relevant point statistics. Columns are:
        #0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  10: Dz-std.dev, 11: Dz-range, \
        #12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: slope of fitted plane, 15: plane residuals, \
        #16: variance dz, 17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals, \
        #21:intensity mean, 22:intensity std., 23: intensity median, 24: 10, 25: 27, 26: 75, 27: 90th percentile intensity
        pts_seed_stats = np.array([pcl_xyzg_rstep_seed[i,0], pcl_xyzg_rstep_seed[i,1], pcl_xyzg_rstep_seed[i,2], 
                       pts_xyz_meanpt[0], pts_xyz_meanpt[1], pts_xyz_meanpt[2], 
                       np.min(pts_xyz, axis=0)[2], np.max(pts_xyz, axis=0)[2], dz.max(), dz.min(), np.std(dz), dz.max()-dz.min(), \
                       np.percentile(dz, 90)-np.percentile(dz,10), np.percentile(dz, 75)-np.percentile(dz,25), plane_slope, plane_residual, \
                       np.var(dz), nr_pts_xyz, slope_lstsq, curvature_lstsq, curv_residuals, i_mean, i_stddev, i_median, i_10_25_75_90p[0], i_10_25_75_90p[1], i_10_25_75_90p[2], i_10_25_75_90p[3],])
    return pts_seed_stats, dxyzn

### Program starts here
### Defining input and setting global variables

if args.inlas == '':
    print('No input LAS/LAZ file given. Rerun with -i for input LAS/LAZ file. Exit.')
    exit()

if args.outputdir == '':
    if len(os.path.dirname(args.inlas)) > 0:
        args.outputdir = os.path.dirname(args.inlas)
    else:
        args.outputdir = os.getcwd()
        
if args.outlas == '':
    args.outlas = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + '_raster_%0.2fm_rsphere%0.2fm.las'%(args.raster_m,args.sphere_radius_m))

if args.slope_sphere_radius_m == 0:
    #set to sphere_radius_m
    args.slope_sphere_radius_m = args.sphere_radius_m

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

### Loading data and filtering
print('\nLoading input file: %s'%args.inlas)
inFile = File(args.inlas, mode='r')
pcl_xyzic = np.vstack((inFile.get_x()*inFile.header.scale[0]+inFile.header.offset[0], inFile.get_y()*inFile.header.scale[1]+inFile.header.offset[1], inFile.get_z()*inFile.header.scale[2]+inFile.header.offset[2], inFile.get_intensity(), inFile.get_classification())).transpose()
#pcl_xyzic is now a point cloud with x, y, z, intensity, and classification
#if args.store_color == True:
#    pcl_i = inFile.get_intensity().copy()
#    pcl_blue = inFile.get_blue().copy()
#    pcl_green = inFile.get_green().copy()
#    pcl_red = inFile.get_red().copy()
print('Loaded %s points'%"{:,}".format(pcl_xyzic.shape[0]))

print('\nFiltering points to only work with ground points (class == 2)... ',end='\n')
#get only ground points:
idx_ground = np.where(pcl_xyzic[:,4] == 2)[0]
pcl_xyzig = pcl_xyzic[idx_ground,0:4]
pcl_xyzg = pcl_xyzic[idx_ground,0:3]
#pcl_xyzg is a point cloud with x, y, z, and for class == 2 only
pcl_xyg = pcl_xyzic[idx_ground,0:2]
#if args.store_color == True:
##    pcl_i = pcl_i[idx_ground,:]
#    pcl_blue = pcl_blue[idx_ground,:]
#    pcl_green = pcl_green[idx_ground,:]
#    pcl_red = pcl_red[idx_ground,:]
if np.max(pcl_xyzic[:,4]) > 2:
    idx_vegetation = np.where(pcl_xyzic[:,4] == 5)[0] # and pcl_xyzic[:,4] == 4 and pcl_xyzic[:,4] == 5)[0]
    #getting vegetation indices
    vegetation_intensity= pcl_xyzic[idx_vegetation,3]
    vegetation_intensity_mean = np.mean(vegetation_intensity)
    vegetation_intensity_std = np.std(vegetation_intensity)
    print('\nNumber of vegetation points (class==5): %s'%"{:,}".format(idx_vegetation.shape[0]))
    print('Vegetation intensity mean: %2.1f +-std.dev.: %2.1f, 10th perc.: %2.1f, 90th perc.: %2.1f'%(vegetation_intensity_mean, vegetation_intensity_std, np.percentile(vegetation_intensity, 10), np.percentile(vegetation_intensity, 90)) )


#getting ground values
ground_intensity = pcl_xyzic[idx_ground,3]
ground_intensity_mean = np.mean(ground_intensity)
ground_intensity_std = np.std(ground_intensity)
print('\nNumber of ground points (class==2): %s'%"{:,}".format(idx_ground.shape[0]))
print('Ground intensity mean: %2.1f +-std.dev.: %2.1f, 10th perc.: %2.1f, 90th perc.: %2.1f'%(ground_intensity_mean, ground_intensity_std, np.percentile(ground_intensity,10), np.percentile(ground_intensity,90)) )

### cKDTree setup and calculation
#Generate KDTree for fast searching
#cKDTree is faster than KDTree, pyKDTree is fast then cKDTree
print('\nGenerating XY-cKDTree... ',end='', flush=True)
pcl_xyg_ckdtree_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + '_xyg_cKDTree.pickle')
if os.path.exists(pcl_xyg_ckdtree_fn):
    pcl_xyg_ckdtree = pickle.load(open( pcl_xyg_ckdtree_fn, "rb" ))
else:
    pcl_xyg_ckdtree = spatial.cKDTree(pcl_xyg, leafsize=32)
    pickle.dump(pcl_xyg_ckdtree, open(pcl_xyg_ckdtree_fn,'wb'))
print('done.')

print('Generating XYZ-cKDTree... ',end='',flush=True)
pcl_xyzg_ckdtree_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + '_xyzg_cKDTree.pickle')
if os.path.exists(pcl_xyzg_ckdtree_fn):
    pcl_xyzg_ckdtree = pickle.load(open( pcl_xyzg_ckdtree_fn, "rb" ))
else:
    pcl_xyzg_ckdtree = spatial.cKDTree(pcl_xyzg, leafsize=32)
    pickle.dump(pcl_xyzg_ckdtree,open(pcl_xyzg_ckdtree_fn,'wb'))
print('done.')


# Loading coordinates from GeoTIFF file
if args.dem_fname != '':
    try:
        src_ds = gdal.Open( args.dem_fname )
    except RuntimeError:
        print('Unable to open {}'.format(args.dem_fname))
        sys.exit(1)

    bandnr = 1
    try:
        src_band = src_ds.GetRasterBand(bandnr)
    except RuntimeError:
        print('Band ( {} ) not found in {}'.format(bandnr, args.dem_fname))
        sys.exit(1)

    #print("{}: [ MIN ] = {}, [ MAX ] = {}".format(os.path.basename(args.dem_fname), src_band.GetMinimum(), src_band.GetMaximum()))
    cols = src_band.XSize
    rows = src_band.YSize
    geo_transform = src_ds.GetGeoTransform()
    x_min = geo_transform[0] 
    x_max = geo_transform[0] + geo_transform[1] * cols
    y_min = geo_transform[3] 
    y_max = geo_transform[3] + geo_transform[5] * rows

    x_elements = len(np.arange(x_min, x_max, args.raster_m))
    if y_min > y_max:
        y_elements = len(np.arange(y_max, y_min, args.raster_m))
        y_coords = np.arange(y_max, y_min, args.raster_m) + args.raster_m / 2
    else:
        y_elements = len(np.arange(y_min, y_max, args.raster_m))
        y_coords = np.arange(y_min, y_max, args.raster_m) + args.raster_m / 2
    
    #get coordinate range and shift coordinates by half of the step size to make sure rater overlay is centered. 
    #This is not really necessary and only matters for very small point clouds with edge effects or for very large steps sizes:
    x_coords = np.arange(x_min, x_max, args.raster_m) + args.raster_m / 2
    
    #create combination of all coordinates (this is using lists and could be optimized)
    xy_coordinates = np.array([(x,y) for x in x_coords for y in y_coords])
else:
    #no GeoTiff file given, using min/max coordinates to generate equally-spaced grid
        
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
[pcl_xyg_ckdtree_distance, pcl_xyg_ckdtree_id] = pcl_xyg_ckdtree.query(xy_coordinates, k=1, n_jobs=-1)

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

### Query points - use KDTree
#find points from 3D seed / query points  / raster overlay with radius = args.sphere_radius_m
pickle_fn = '_xyzg_raster_%0.2fm_radius_%0.2fm.pickle'%(args.raster_m, args.sphere_radius_m)
pcl_xyzg_radius_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + pickle_fn)
print('\nQuerying cKDTree with radius %0.2f and storing in pickle: %s... '%(args.sphere_radius_m, pcl_xyzg_radius_fn), end='', flush=True)
if os.path.exists(pcl_xyzg_radius_fn):
    pcl_xyzg_radius = pickle.load(open( pcl_xyzg_radius_fn, "rb" ))
else:
    pcl_xyzg_radius = pcl_xyzg_ckdtree.query_ball_point(pcl_xyzg_rstep_seed, r=args.sphere_radius_m, n_jobs=-1)
    pickle.dump(pcl_xyzg_radius, open(pcl_xyzg_radius_fn,'wb'))
print('done.')

# Find sphere points for slope and curvature calculation (if slope and curvature radius is different) from sphere_radius_m
if args.slope_sphere_radius_m != args.sphere_radius_m:    
    pickle_fn = '_xyzg_raster_%0.2fm_radius_%0.2fm.pickle'%(args.raster_m, args.slope_sphere_radius_m)
    pcl_xyzg_radius2_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + pickle_fn)
    print('\nSlope and Curvature point extraction: Querying cKDTree with radius %0.2f and storing in pickle: %s... '%(args.slope_sphere_radius_m, pcl_xyzg_radius2_fn), end='', flush=True)
    if os.path.exists(pcl_xyzg_radius2_fn):
        pcl_xyzg_radius_slope = pickle.load(open( pcl_xyzg_radius2_fn, "rb" ))
    else:
        pcl_xyzg_radius_slope = pcl_xyzg_ckdtree.query_ball_point(pcl_xyzg_rstep_seed,r=args.slope_sphere_radius_m, n_jobs=-1)
        pickle.dump(pcl_xyzg_radius_slope, open(pcl_xyzg_radius2_fn,'wb'))
    print('done.')
elif args.slope_sphere_radius_m == args.sphere_radius_m:
    pcl_xyzg_radius_slope = pcl_xyzg_radius
  
#NOTE: Seperately storing and loading points for slope sphere and roughness sphere. This could be optimized so that only one will be loaded.
    
### Calculate statistics for each sphere: normalization, elevation range, std. dev., mean, median
#Setup variables
pcl_xyzg_radius_nr = len(pcl_xyzg_radius)
nr_of_datasets = 28 #nr of columns to save
nr_of_processes = 100 #splitting the for loop into 100 processes and dividing array into 100 steps in pos_array
dxyzn_max_nre = np.max([len(x) for x in pcl_xyzg_radius]) #extract list of neighboring points
dxyzn_nre = np.sum([len(x) for x in pcl_xyzg_radius])
dxyzn_nre_pos_array = np.array(np.linspace(0, dxyzn_nre, nr_of_processes), dtype=int)
#pcl_xyzg_radius_nre = np.sum([len(x) for x in pcl_xyzg_radius])

### PC Density adjustment / normalization
# Get PC density of entire point cloud using a fixed number of neighbors determined by point density of seed points

if args.nr_random_sampling > 0:
    pc_density_fn = os.path.join(args.outputdir, 'PC_density_stats_n%04d_r%0.2fm.h5'%(dxyzn_max_nre, args.sphere_radius_m))
    if os.path.exists(pc_density_fn) == False:
        print('\nDensity of complete ground-classified XYZ pointcloud (pcl_xyzg, n=%s points) for %d neighbor points from %2.1f m radius:'%("{:,}".format(int(pcl_xyzg.shape[0])), int(dxyzn_max_nre),args.sphere_radius_m) )
        pcl_xyzg_p_min, pcl_xyzg_p_median, \
            pcl_xyzg_density_min, pcl_xyzg_density_median = \
            pc_density(pcl_xyzg, pcl_xyzg_ckdtree, nn=int(dxyzn_max_nre), show_density_information=1)
        
        #write as HDF file
        print('\nWriting PC density results to HDF... ', end='', flush=True)
        hdf_out = h5py.File(pc_density_fn,'w')
        hdf_out.attrs['help'] = 'PC Density and probability (p) for n=%d neighbor points from pc_geomorph_roughness.py: sphere radius %0.2fm'%(dxyzn_max_nre, args.sphere_radius_m) 
        pcl_xyzg_p_min_fc = hdf_out.create_dataset('pcl_xyzg_p_min',data=pcl_xyzg_p_min, chunks=True, compression="gzip", compression_opts=7)
        pcl_xyzg_p_min_fc.attrs['help'] = 'Minimum probability for each points'
        pcl_xyzg_p_median_fc = hdf_out.create_dataset('pcl_xyzg_p_median',data=pcl_xyzg_p_median, chunks=True, compression="gzip", compression_opts=7)
        pcl_xyzg_p_median_fc.attrs['help'] = 'Median probability for each points'
        pcl_xyzg_density_min_fc = hdf_out.create_dataset('pcl_xyzg_density_min',data=pcl_xyzg_density_min, chunks=True, compression="gzip", compression_opts=7)
        pcl_xyzg_density_min_fc.attrs['help'] = 'Minimum density for each points'
        pcl_xyzg_density_median_fc = hdf_out.create_dataset('pcl_xyzg_density_median',data=pcl_xyzg_density_median, chunks=True, compression="gzip", compression_opts=7)
        pcl_xyzg_density_median_fc.attrs['help'] = 'Median density for each points'
        pcl_xyzg_fc = hdf_out.create_dataset('pcl_xyzg',data=pcl_xyzg, chunks=True, compression="gzip", compression_opts=7)
        pcl_xyzg_fc.attrs['help'] = 'XYZ coordinates of pointcloud'
        hdf_out.close()
        print('done.')
    elif os.path.exists(pc_density_fn) == True:
        hdf_in = h5py.File(pc_density_fn,'r')
        pcl_xyzg_p_min = np.array(hdf_in['pcl_xyzg_p_min'])
        pcl_xyzg_density_min = np.array(hdf_in['pcl_xyzg_density_min'])
     
    # The following is just an example how the random subsampling based on a density can be carried out. 
    # As input is taken the full point cloud and the number of points that each seed point should contain for a given radius.
    # Here we chose half of the minimum number of points for a given density calculated in the previous step (pcl_xyzg_p_min)
    # Probability subsampling using PC density estimates and subsample points based on their probabilities
    nr_of_points = pcl_xyzg.shape[0]/2
    pc_equal_density_fn = os.path.join(args.outputdir, 'PC_equal_density_n%04d_r%0.2fm.h5'%(dxyzn_max_nre, args.sphere_radius_m))
    if os.path.exists(pc_equal_density_fn) == False:
        print('\nHomogenous subsampling of pointcloud to create equal point density with %s points to %s points'%("{:,}".format(pcl_xyzg.shape[0]), "{:,}".format(int(nr_of_points))) )
        pcl_xyzg_p_random = pc_random_p_subsampling(pcl_xyzg, pcl_xyzg_p_min, nr_of_points = nr_of_points)
        #Generate cKDTree for density estimation 
        print('\nGenerating XYZ-cKDTree for homogenous random point cloud... ',end='',flush=True)
        pcl_xyzg_p_random_ckdtree = spatial.cKDTree(pcl_xyzg_p_random, leafsize=32)
        print('done.')
        #calculate density of pcl_xyzg_p_random:
        #This shoud give similar density values (min, median, and max. densities, and a relatively low standard deviation of median point distances)
        print('\nDensity of homogenous random subsampled ground-classified XYZ pointcloud (pcl_xyzg_p_random, n=%s points) for %d neighbor points from %2.1f m radius:'%("{:,}".format(int(nr_of_points)), int(dxyzn_max_nre),args.sphere_radius_m) )
        _, _, pcl_random_density_min, _ = \
            pc_density(pcl_xyzg_p_random, pcl_xyzg_p_random_ckdtree, nn=int(dxyzn_max_nre), show_density_information=1)
        pcl_xyzg_equal_density_radius = pcl_xyzg_p_random_ckdtree.query_ball_point(pcl_xyzg_rstep_seed,r=args.slope_sphere_radius_m, n_jobs=-1)
        pcl_xyzg_equal_density_radius_slope = pcl_xyzg_equal_density_radius
        
        #write as HDF file
        print('\nWriting PC density results to HDF... ', end='', flush=True)
        hdf_out = h5py.File(pc_equal_density_fn,'w')
        hdf_out.attrs['help'] = 'Homogenous PC with equal point density based on random subsampling with %d nr of total points for n=%d neighbor points from pc_geomorph_roughness.py: sphere radius %0.2fm'%(nr_of_points, dxyzn_max_nre, args.sphere_radius_m) 
        pcl_xyzg_p_random_fc = hdf_out.create_dataset('pcl_xyzg_p_random',data=pcl_xyzg_p_random, chunks=True, compression="gzip", compression_opts=7)
        pcl_xyzg_p_random_fc.attrs['help'] = 'XYZ coordinates of subsampled PC (homogenous based on probabilities)'
        pcl_random_density_min_fc = hdf_out.create_dataset('pcl_random_density_min',data=pcl_random_density_min, chunks=True, compression="gzip", compression_opts=7)
        pcl_random_density_min_fc.attrs['help'] = 'PC density after subsampling'
        hdf_out.close()
        print('done.')
    elif os.path.exists(pc_equal_density_fn) == True:
        hdf_in = h5py.File(pc_equal_density_fn,'r')
        pcl_xyzg_p_random = np.array(hdf_in['pcl_xyzg_p_random'])
        pcl_xyzg_p_random_ckdtree = spatial.cKDTree(pcl_xyzg_p_random, leafsize=32)
        pcl_xyzg_equal_density_radius = pcl_xyzg_p_random_ckdtree.query_ball_point(pcl_xyzg_rstep_seed,r=args.slope_sphere_radius_m, n_jobs=-1)
        pcl_xyzg_equal_density_radius_slope = pcl_xyzg_equal_density_radius

### Bootstraping and uncertainty estimation for fitting data
### Calculate slope and curvature for random point cloud 'pcl_xyzg_p_random' with roughly homogenous point densities
# get points from sub-sampled point cloud for all seed points to calculate curvature and slope and their uncertainties
if args.nr_random_sampling > 0:
    intensity_mean_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_intensity_mean.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    intensity_stdev_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_intensity_stdev.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    intensity_iqr_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_intensity_iqr.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    nrlidar_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_nrlidar.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    dz_std_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_stddev.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    dz_range9010_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_dz_range9010.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    dz_iqr_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_dz_iqr.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    plane_slope_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_planeslope.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    dz_max_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_dzmax.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    dz_min_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_dzmin.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    plane_curv_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_curv.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    pc_density_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_iter%02d_%0.2fm_rsphere%0.2fm_density.tif'%(args.nr_random_sampling, args.raster_m,args.sphere_radius_m))
    #perform bootstrapping analysis
    nr_of_seed_points = pcl_xyzg_rstep_seed.shape[0]    

    pos_array = np.array(np.linspace(0, pcl_xyzg_radius_nr, nr_of_processes), dtype=int) #This creates a position array so you can select from:to in each loop

    pc_bootstrapping_fn = os.path.join(args.outputdir, 'PC_bootstrap_stats_all_iter%02d_r%0.2fm.h5'%(args.nr_random_sampling, args.sphere_radius_m))
    nr_of_bootstraps = args.nr_random_sampling #number of iterations how often a random point cloud should be generated and tested for slope/curvature calculation
    if os.path.exists(pc_bootstrapping_fn):
        hdf_in = h5py.File(pc_bootstrapping_fn,'r')
        pts_seed_stats_bootstrap = np.array(hdf_in['pts_seed_stats_bootstrap'])
        pts_seed_stats_bootstrap_mean = np.array(hdf_in['pts_seed_stats_bootstrap_mean'])
        pc_random_density_min_bootstrap = np.array(hdf_in['pc_random_density_min_bootstrap'])
    else:
        if args.nr_of_cores == 0:
            args.nr_of_cores = multiprocessing.cpu_count()
        #Use only 1/2 of the max. number of cores
        
        print('\nBootstrapping: Using %d/%d cores for %s seed points'%(np.round(args.nr_of_cores / 2).astype(int), multiprocessing.cpu_count(), "{:,}".format(nr_of_seed_points) ))
        slope_lstsq = np.empty((nr_of_bootstraps, nr_of_seed_points))
        slope_lstsq.fill(np.nan)
        slope_plane = np.empty((nr_of_bootstraps, nr_of_seed_points))
        slope_plane.fill(np.nan)
        curvature_lstsq = np.empty((nr_of_bootstraps, nr_of_seed_points))
        curvature_lstsq.fill(np.nan)
        dZ_iqr = np.empty((nr_of_bootstraps, nr_of_seed_points))
        dZ_iqr.fill(np.nan)
        intensity_mean = np.empty((nr_of_bootstraps, nr_of_seed_points))
        intensity_mean.fill(np.nan)
        intensity_stddev = np.empty((nr_of_bootstraps, nr_of_seed_points))
        intensity_stddev.fill(np.nan)
        intensity_iqr = np.empty((nr_of_bootstraps, nr_of_seed_points))
        intensity_iqr.fill(np.nan)
        pc_random_density_min_bootstrap = np.empty((nr_of_bootstraps, nr_of_seed_points))
        pc_random_density_min_bootstrap.fill(np.nan)
    
        pts_seed_stats_bootstrap = np.empty((nr_of_bootstraps, nr_of_seed_points, nr_of_datasets))
        pts_seed_stats_bootstrap.fill(np.nan)    
    
        print('Selecting %s points from %s total points for %s seed points with distance=%s m.'%("{:,}".format(int(nr_of_points)), "{:,}".format(pcl_xyzg.shape[0]), "{:,}".format(nr_of_seed_points), args.sphere_radius_m), end='\n', flush=True)
        ts = time.time()
        for i in range(nr_of_bootstraps):
            pcl_xyzg_p_random = pc_random_p_intensity_subsampling(pcl_xyzig, pcl_xyzg_p_min, nr_of_points = nr_of_points)
            #print('Writing randomly sampled PC to HDF (%d/%d) ... '%(i, nr_of_bootstraps-1), end='', flush=True)
            pc_ptsxyzig_bootstrapping_fn = os.path.join(args.outputdir, 'PC_bootstrap_iter%02d.h5'%(i))
            hdf_out = h5py.File(pc_ptsxyzig_bootstrapping_fn,'w')
            hdf_out.attrs['help'] = 'XYZ-intensity of (randomly) sampled PC with %s points for %d neighborhood points from pc_geomorph_roughness.py: sphere radius %0.2fm'%("{:,}".format(int(nr_of_points)), dxyzn_max_nre, args.sphere_radius_m) 
            pts_pts_bootstrap_fc = hdf_out.create_dataset('pcl_xyzg_p_random',data=pcl_xyzg_p_random, chunks=True, compression="gzip", compression_opts=7)
            pts_pts_bootstrap_fc.attrs['help'] = 'Randomly sampled XYZ and intensity'
            pcl_xyzg_p_min_fc = hdf_out.create_dataset('pcl_xyzg_p_min',data=pcl_xyzg_p_min, chunks=True, compression="gzip", compression_opts=7)
            pcl_xyzg_p_min_fc.attrs['help'] = 'Minimum probability for each points'
            hdf_out.close()
            #print('done.',end='', flush=True)
            
            #have array with random points and intensity            
            pcl_xyzg_p_random_ckdtree = spatial.cKDTree(pcl_xyzg_p_random[:,0:3], leafsize=32)
        
            pcl_xyzg_equal_density_radius = pcl_xyzg_p_random_ckdtree.query_ball_point(pcl_xyzg_rstep_seed,r=args.slope_sphere_radius_m, n_jobs=-1)
            pcl_xyzg_equal_density_radius_slope = pcl_xyzg_equal_density_radius
        
            #calculating density for seed points
            _, _, pc_random_density_min_res, _ = \
            pc_density(pcl_xyzg_rstep_seed, pcl_xyzg_p_random_ckdtree, nn=int(dxyzn_max_nre) )
            pc_random_density_min_bootstrap[i,:] = pc_random_density_min_res
    
            #calculating statistics for each seed point: using parallelization
            print('\nAt %d/%d [%%]: '%(i+1, nr_of_bootstraps), end='', flush=True)
            p = Pool(processes=np.round(args.nr_of_cores / 2).astype(int))
            for _ in p.imap_unordered(calc_stats_for_bootstrap_seed_points_wrapper, np.arange(0,len(pos_array)-1)):
                pass    
        
            #non-parallized loop (there are still some statistics call that run on multiple cores)
            #for ii in np.arange(0,len(pos_array)-1):
            #    calc_stats_for_bootstrap_seed_points_wrapper(ii)
                
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
            for ii in pkls:
                os.remove(ii)
            pkls=None
            pcl_xyzg_p_random_seed_radius=None
            pcl_xyzg_p_random = None
            pcl_xyzg_p_random_ckdtree = None
            #pts_seed_stats contains a list of the following:
            #0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  10: Dz-std.dev, 11: Dz-range, \
            #12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: slope of fitted plane, 15: plane residuals, \
            #16: variance dz, 17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals, \
            #21:intensity mean, 22:intensity std., 23: intensity median, 24: 10, 25: 27, 26: 75, 27: 90th percentile intensity
        
            #exctract specific parameters
            slope_lstsq[i,:] = pts_seed_stats[:,18]
            slope_plane[i,:] = pts_seed_stats[:,14]
            curvature_lstsq[i,:] = pts_seed_stats[:,19]
            dZ_iqr[i,:] = pts_seed_stats[:,13]
            intensity_mean[i,:] = pts_seed_stats[:,19]
            intensity_stddev[i,:] = pts_seed_stats[:,20]
            intensity_iqr[i,:] = pts_seed_stats[:,24]-pts_seed_stats[:,23]
            pts_seed_stats_bootstrap[i,:,:] = pts_seed_stats
            pc_random_density_min_bootstrap[i,:] = pc_random_density_min_res
            del pts_seed_stats
            del pc_random_density_min_res
            
        # The following parallization is not working yet:
        ##    # single-core processing is the following for loop
        ##    # NOTE that the python functions within calc_pts_length_scale_multicore also have some multi-core functionality
        #    for ii in np.arange(0,len(pos_array)-1):
        #        from_pos = pos_array[ii] #Get start/end from position array
        #        to_pos = pos_array[ii+1]
        #        subarr = np.arange(from_pos, to_pos) #Slice the data into the selected part...
        #        slope_lstsq_res, curvature_lstsq_res, dZ_residuals_lstsq_res, slope_plane_res = \
        #            pcl_xyzg_p_slope_curvature_singlecore(ii, subarr)
        #        slope_lstsq[i,from_pos:to_pos] = slope_lstsq_res
        #        curvature_lstsq[i,from_pos:to_pos] = curvature_lstsq_res
        #        dZ_residuals_lstsq[i,from_pos:to_pos] = dZ_residuals_lstsq_res
        #        slope_plane[i,from_pos:to_pos] = slope_plane_res
        #        del slope_lstsq_res, curvature_lstsq_res, dZ_residuals_lstsq_res, slope_plane_res
        #    del pcl_random_density_min_res
        #    #multi-core processing (not set up properly yet)
        #    #results = p.map(pcl_xyzg_p_slope_curvature_multicore, np.arange(0,len(pos_array)-1) )
        
        
        print('\ntotal time: %0.2fs or %0.2fm'%(time.time() - ts, (time.time() - ts)/60))        
    #The following is creating warning, because some columns may be nan
    #    curvature_lstsq_stats = np.array([np.nanmean(curvature_lstsq, axis=0), np.nanstd(curvature_lstsq, axis=0), np.nanmedian(curvature_lstsq, axis=0), np.nanvar(curvature_lstsq, axis=0)])
    #    dZ_iqr_stats = np.array([np.nanmean(dZ_iqr, axis=0), np.nanstd(dZ_iqr, axis=0), np.nanmedian(dZ_iqr, axis=0), np.nanvar(dZ_iqr, axis=0)])
    #    slope_plane_stats = np.array([np.nanmean(slope_plane, axis=0), np.nanstd(slope_plane, axis=0), np.nanmedian(slope_plane, axis=0), np.nanvar(slope_plane, axis=0)])
    #    slope_lstsq_point_stats = np.array([np.nanmean(slope_lstsq, axis=0), np.nanstd(slope_lstsq, axis=0), np.nanmedian(slope_lstsq, axis=0), np.nanvar(slope_lstsq, axis=0)])
    #    slope_lstsq_uncertainty_percentage = slope_lstsq_point_stats[1,:] / slope_lstsq_point_stats[0,:]
    #    print('\nMean slope uncertainty: %2.2f'%(np.nanmean(slope_lstsq_uncertainty_percentage)) )    
        
        #calculating mean slope, curvature, intensity
        pts_seed_stats_bootstrap_mean = np.nanmean(pts_seed_stats_bootstrap, axis=0)
        
        #write as HDF file
        print('\nWriting bootstrap results to HDF, CSV, and shapefiles... ', end='', flush=True)
        hdf_out = h5py.File(pc_bootstrapping_fn,'w')
        hdf_out.attrs['help'] = 'Bootstrapping results (i=%d) from pc_geomorph_roughness.py: sphere radius %0.2fm'%(nr_of_bootstraps, args.sphere_radius_m) 
        pts_seed_stats_bootstrap_fc = hdf_out.create_dataset('pts_seed_stats_bootstrap',data=pts_seed_stats_bootstrap, chunks=True, compression="gzip", compression_opts=7)
        pts_seed_stats_bootstrap_fc.attrs['help'] = '''Nr. of seed pts: %d,  pts_seed_stats_bootstrap shape: %d x %d x %d, with :,x,: (col: name) 
        0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  
        10: Dz-std.dev, 11: Dz-range, 12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: slope of fitted plane, 
        15: plane residuals, 16: variance dz, 17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals, 
        21:intensity mean, 22:intensity std., 23: intensity median, 24: 10, 25: 27, 26: 75, 27: 90th percentile intensity'''\
        %(pcl_xyzg_rstep_seed.shape[0], pts_seed_stats_bootstrap.shape[0], pts_seed_stats_bootstrap.shape[1], pts_seed_stats_bootstrap.shape[2])
        pts_seed_stats_bootstrap_mean_fc = hdf_out.create_dataset('pts_seed_stats_bootstrap_mean',data=pts_seed_stats_bootstrap_mean, chunks=True, compression="gzip", compression_opts=7)
        pts_seed_stats_bootstrap_mean_fc.attrs['help'] = '''Nr. of seed pts: %d,  mean of bootstrapping (col: name) 
        0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  
        10: Dz-std.dev, 11: Dz-range, 12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: slope of fitted plane, 
        15: plane residuals, 16: variance dz, 17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals, 
        21:intensity mean, 22:intensity std., 23: intensity median, 24: 10, 25: 27, 26: 75, 27: 90th percentile intensity'''\
        %(pcl_xyzg_rstep_seed.shape[0])
        pc_random_density_min_fc = hdf_out.create_dataset('pc_random_density_min_bootstrap',data=pc_random_density_min_bootstrap, chunks=True, compression="gzip", compression_opts=7)
        pc_random_density_min_fc.attrs['help'] = 'Nr. of seed pts: %d,  density of homogeneously sampled point cloud (minimum) for each iteration'%(pcl_xyzg_rstep_seed.shape[0])
        hdf_out.close()

    #write csv
    header_str='1SeedX, 2SeedY, 3SeedZ, 4MeanX, 5MeanY, 6MeanZ, 7Z_min, 8Z_max, 9Dz_max, 10Dz_min, 11Dz_std, 12Dz_range, 13Dz_9010p, 14Dz_7525p, 15_Pl_slp, 16Pl_res, 17Pl_Var, 18Nr_lidar, 19SlopeLSQ, 20CurvLSQ, 21CurvRes, 22I_mean, 23I_std, 24I_med, 25I_p10, 26I_p25, 27I_p75, 28I_p90'
    seed_pts_stats_csv = '_iter%02d_mean_seed_pts_stats_raster_%0.2fm_radius_%0.2fm.csv'%(nr_of_bootstraps, args.raster_m, args.sphere_radius_m)
    pcl_seed_pts_stats_csv_fn = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_csv)
    seed_pts_stats_vrt = '_iter%02d_mean_seed_pts_stats_raster_%0.2fm_radius_%0.2fm.vrt'%(nr_of_bootstraps, args.raster_m, args.sphere_radius_m)
    pcl_seed_pts_stats_vrt_fn = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_vrt)
    seed_pts_stats_shp = '_iter%02d_mean_seed_pts_stats_raster_%0.2fm_radius_%0.2fm.shp'%(nr_of_bootstraps, args.raster_m, args.sphere_radius_m)
    pcl_seed_pts_stats_shp_fn = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_shp)
    idxnan = np.where(np.isnan(pts_seed_stats_bootstrap_mean))
    if os.path.exists(pcl_seed_pts_stats_csv_fn) == False:
        #before writing to CSV file, replace all np.nan in pts_seed_stats with -9999
        pts_seed_stats_nonan = np.copy(pts_seed_stats_bootstrap_mean)
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
    vrt_f.write('\t\t\t<Field name="19SlopeLSQ" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="20CurvLSQ" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="21CurvRes" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="22I_mean" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="23I_std" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="24I_med" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="25I_p10" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="26I_p25" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="27I_p75" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="28I_p90" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t</OGRVRTLayer>\n')
    vrt_f.write('</OGRVRTDataSource>\n')
    vrt_f.close()

    # Generate shapefile from vrt
    if os.path.exists(pcl_seed_pts_stats_shp_fn) == False:
        cwd=os.getcwd()
        os.chdir(args.outputdir)
        cmd = ['ogr2ogr', pcl_seed_pts_stats_shp_fn, pcl_seed_pts_stats_vrt_fn]
        logfile_fname = os.path.join(args.outputdir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt'
        logfile_error_fname = os.path.join(args.outputdir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt'
        with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
            subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
            subprocess_p.wait()
        os.chdir(cwd)
    print('done.')

    ### Interpolate to equally-spaced grid and generate GeoTIFF output
    print('\nInterpolating seed points (mean-X, mean-Y) to geotiff rasters and writing geotiff raster... ',end='', flush=True)
    xx,yy = np.meshgrid(x_coords,y_coords)
    idx_nonan = np.where(np.isnan(pts_seed_stats_bootstrap_mean[:,3])==False)
    points = np.hstack((pts_seed_stats_bootstrap_mean[idx_nonan,3].T, pts_seed_stats_bootstrap_mean[idx_nonan,4].T))

    ncols=cols
    nrows=rows
    xres = args.raster_m
    yres = args.raster_m
    geotransform = (x_coords.min() - (args.raster_m / 2),xres, 0 , y_coords.min() - (args.raster_m / 2),0, yres) 
    
    #interpolate lidar intensity mean
    if os.path.exists(intensity_mean_tif_fn) == False:
        print('intensity_mean, ', end='', flush=True)
        intensity_mean = griddata_clip_geotif(intensity_mean_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,21][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(intensity_mean_tif_fn)
        intensity_mean = np.array(ds.GetRasterBand(1).ReadAsArray())
        intensity_mean[np.where(intensity_mean == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate lidar intensity IQR
    if os.path.exists(intensity_iqr_tif_fn) == False:
        print('intensity_IQR, ', end='', flush=True)
        intensity_iqr = griddata_clip_geotif(intensity_iqr_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,26][0]-pts_seed_stats_bootstrap_mean[idx_nonan,25][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(intensity_iqr_tif_fn)
        intensity_iqr = np.array(ds.GetRasterBand(1).ReadAsArray())
        intensity_iqr[np.where(intensity_iqr == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate nr_lidar_measurements
    if os.path.exists(nrlidar_tif_fn) == False:
        print('nr_lidar_measurements, ', end='', flush=True)
        nr_lidar = griddata_clip_geotif(nrlidar_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,17][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(nrlidar_tif_fn)
        nr_lidar = np.array(ds.GetRasterBand(1).ReadAsArray())
        nr_lidar[np.where(nr_lidar == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    if os.path.exists(dz_std_tif_fn) == False:
        print('Dz std. dev., ', end='', flush=True)
        dz_std = griddata_clip_geotif(dz_std_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,10][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_std_tif_fn)
        dz_std = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_std[np.where(dz_std == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz_range 90-10 percentile
    if os.path.exists(dz_range9010_tif_fn) == False:
        print('range (90-10th perc.), ', end='', flush=True)
        dz_range9010 = griddata_clip_geotif(dz_range9010_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,12][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_range9010_tif_fn)
        dz_range9010 = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_range9010[np.where(dz_range9010 == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz_range 75-25 percentile
    if os.path.exists(dz_iqr_tif_fn) == False:
        print('range (75-25th perc.), ', end='', flush=True)
        dz_iqr = griddata_clip_geotif(dz_iqr_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,13][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_iqr_tif_fn)
        dz_iqr = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_iqr[np.where(dz_iqr == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz-max
    if os.path.exists(dz_max_tif_fn) == False:
        print('max. dz, ', end='', flush=True)
        dz_max = griddata_clip_geotif(dz_max_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,8][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_max_tif_fn)
        dz_max = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_max[np.where(dz_max == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz-min
    if os.path.exists(dz_min_tif_fn) == False:
        print('min. dz, ', end='', flush=True)
        dz_min = griddata_clip_geotif(dz_min_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,9][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_min_tif_fn)
        dz_min = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_min[np.where(dz_min == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Plane_slope
    if os.path.exists(plane_slope_tif_fn) == False:
        print('plane slope, ', end='', flush=True)
        plane_slope = griddata_clip_geotif(plane_slope_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,14][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(plane_slope_tif_fn)
        plane_slope = np.array(ds.GetRasterBand(1).ReadAsArray())
        plane_slope[np.where(plane_slope == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate LST-Square curvate
    if os.path.exists(plane_curv_tif_fn) == False:
        print('LSTSQ curvature, ', end='', flush=True)
        plane_curv = griddata_clip_geotif(plane_curv_tif_fn, points, pts_seed_stats_bootstrap_mean[idx_nonan,19][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(plane_curv_tif_fn)
        plane_curv = np.array(ds.GetRasterBand(1).ReadAsArray())
        plane_curv[np.where(plane_curv == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate PC Density
    if os.path.exists(pc_density_tif_fn) == False:
        print('PC density, ', end='', flush=True)
        pc_density_grid = griddata_clip_geotif(pc_density_tif_fn, points, np.nanmean(pc_random_density_min_bootstrap, axis=0)[idx_nonan], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(pc_density_tif_fn)
        pc_density_grid = np.array(ds.GetRasterBand(1).ReadAsArray())
        pc_density_grid[np.where(pc_density_grid == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    print('done.')
       
    ### Plot output to figures
    seed_pts_stats_png = '_bootstrap_iter%02d_seed_pts_overview1_raster_%0.2fm_radius_%0.2fm.png'%(args.nr_random_sampling, args.raster_m, args.sphere_radius_m)
    pcl_seed_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_png)
    print('\nGenerating overview figure 1 for bootstrap results %s... '%os.path.basename(pcl_seed_pts_fig_overview_fn), end='', flush=True)
    if os.path.exists(pcl_seed_pts_fig_overview_fn) == False:
        fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
        #fig = plt.figure(figsize=(11.69,8.27), dpi=150)
        fig.clf()
        
        ax1 = fig.add_subplot(231)
        ax1.grid()
        cax1 = ax1.scatter(pts_seed_stats_bootstrap_mean[:,0], pts_seed_stats_bootstrap_mean[:,1], c=pts_seed_stats_bootstrap_mean[:,2], s=0.5, cmap=plt.get_cmap('terrain'), linewidth=0)
        ax1.set_title('Bootstrap: Lidar seed point elevation with raster=%0.2fm'%args.raster_m,y=1.05)
        cbar = fig.colorbar(cax1)
        cbar.set_label('Mean elevation in sphere (m)')
        ax1.set_xlabel('UTM-X (m)')
        ax1.set_ylabel('UTM-Y (m)')
        ax1.axis('equal')
    
        ax2 = fig.add_subplot(232)
        ax2.grid()
        cax2 = ax2.imshow(nr_lidar,cmap=plt.get_cmap('gnuplot'))
        #ax2.scatter(pts_seed_stats_bootstrap_mean[:,0], pts_seed_stats_bootstrap_mean[:,1], c=pts_seed_stats_bootstrap_mean[:,15], s=0.1, cmap=plt.get_cmap('gnuplot'), linewidth=0)    
        ax2.set_title('Bootstrap: Nr. of lidar measurements for each seed point with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax2)
        cbar.set_label('nr. of lidar measurements')
        ax2.set_xlabel('UTM-X (m)')
        ax2.set_ylabel('UTM-Y (m)')
        ax2.axis('equal')
        
        ax3 = fig.add_subplot(233)
        ax3.grid()
        #cax3 = ax3.scatter(pts_seed_stats_bootstrap_mean[:,0], pts_seed_stats_bootstrap_mean[:,1], c=pts_seed_stats_bootstrap_mean[:,13], s=0.5, cmap=plt.get_cmap('seismic'), vmin=np.nanpercentile(pts_seed_stats_bootstrap_mean[:,13], 10), vmax=np.nanpercentile(pts_seed_stats_bootstrap_mean[:,13], 90), linewidth=0)
        cax3 = ax3.imshow(plane_slope, cmap=plt.get_cmap('seismic'), vmin=np.nanpercentile(plane_slope, 10), vmax=np.nanpercentile(plane_slope, 90))
        ax3.set_title('Bootstrap: Slope of fitted plane for each sphere with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax3)
        cbar.set_label('Slope (m/m)')
        ax3.set_xlabel('UTM-X (m)')
        ax3.set_ylabel('UTM-Y (m)')
        ax3.axis('equal')
    
        ax5 = fig.add_subplot(234)
        ax5.grid()
        cax5 = ax5.imshow(plane_curv, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(plane_curv, 10), vmax=np.nanpercentile(plane_curv, 90))
        ax5.set_title('Bootstrap: Curvature of sphere/disc with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax5)
        cbar.set_label('Curvature (1/m)')
        ax5.set_xlabel('UTM-X (m)')
        ax5.set_ylabel('UTM-Y (m)')
        ax5.axis('equal')
    
        ax4 = fig.add_subplot(235)
        ax4.grid()
        #cax4 = ax4.scatter(pts_seed_stats_bootstrap_mean[:,0], pts_seed_stats_bootstrap_mean[:,1], c=pts_seed_stats_bootstrap_mean[:,12], s=0.5, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(pts_seed_stats_bootstrap_mean[:,12], 10), vmax=np.nanpercentile(pts_seed_stats_bootstrap_mean[:,12], 90), linewidth=0)
        cax4 = ax4.imshow(dz_range9010, cmap=plt.get_cmap('Spectral'), vmin=np.nanpercentile(dz_range9010, 10), vmax=np.nanpercentile(dz_range9010, 90))
        ax4.set_title('Bootstrap: Surface roughness: Range of offsets from linear plane (90-10th percentile) with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax4)
        cbar.set_label('Range (90-10th) of plane offsets (m)')
        ax4.set_xlabel('UTM-X (m)')
        ax4.set_ylabel('UTM-Y (m)')
        ax4.axis('equal')
    
        ax6 = fig.add_subplot(236)
        ax6.grid()
        #cax6 = ax6.scatter(pts_seed_stats_bootstrap_mean[:,0], pts_seed_stats_bootstrap_mean[:,1], c=pts_seed_stats_bootstrap_mean[:,10], s=0.5, cmap=plt.get_cmap('jet'), vmin=np.nanpercentile(pts_seed_stats_bootstrap_mean[:,10], 10), vmax=np.nanpercentile(pts_seed_stats_bootstrap_mean[:,10], 90), linewidth=0)
        cax6 = ax6.imshow(intensity_mean, cmap=plt.get_cmap('gray'), vmin=np.nanpercentile(intensity_mean, 10), vmax=np.nanpercentile(intensity_mean, 90))
        ax6.set_title('Bootstrap: Point Cloud intensity for k=%d nearest neighbors and r=%0.2f'%(dxyzn_max_nre, args.sphere_radius_m),y=1.05)
        cbar = fig.colorbar(cax6)
        cbar.set_label('Intensity (mean)')
        ax6.set_xlabel('UTM-X (m)')
        ax6.set_ylabel('UTM-Y (m)')
        ax6.axis('equal')
        
        fig.savefig(pcl_seed_pts_fig_overview_fn, bbox_inches='tight')
        plt.close()
    print('done.')
    
    seed_pts_stats_png = '_bootstrap_iter%02d_seed_pts_overview2_raster_%0.2fm_radius_%0.2fm.png'%(args.nr_random_sampling, args.raster_m, args.sphere_radius_m)
    pcl_seed_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_png)
    print('\nGenerating overview figure 2 for bootstrap results %s... '%os.path.basename(pcl_seed_pts_fig_overview_fn), end='', flush=True)
    if os.path.exists(pcl_seed_pts_fig_overview_fn) == False:
        fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
        #fig = plt.figure(figsize=(11.69,8.27), dpi=150)
        fig.clf()
        
        ax1 = fig.add_subplot(221)
        ax1.grid()
        cax1 = ax1.imshow(intensity_iqr, cmap=plt.get_cmap('gray'), vmin=np.nanpercentile(intensity_iqr, 10), vmax=np.nanpercentile(intensity_iqr, 90))
        ax1.set_title('Bootstrap: Point Cloud intensity IQR for k=%d nearest neighbors and r=%0.2f'%(dxyzn_max_nre, args.sphere_radius_m),y=1.05)
        cbar = fig.colorbar(cax1)
        cbar.set_label('Intensity (IQR)')
        ax1.set_xlabel('UTM-X (m)')
        ax1.set_ylabel('UTM-Y (m)')
        ax1.axis('equal')
    
        ax2 = fig.add_subplot(222)
        ax2.grid()
        cax2 = ax2.imshow(pc_density_grid,cmap=plt.get_cmap('gnuplot'))
        ax2.set_title('Bootstrap: Point Cloud density for each seed point with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax2)
        cbar.set_label('density')
        ax2.set_xlabel('UTM-X (m)')
        ax2.set_ylabel('UTM-Y (m)')
        ax2.axis('equal')
    
        ax4 = fig.add_subplot(223)
        ax4.grid()
        #cax4 = ax4.scatter(pts_seed_stats_bootstrap_mean[:,0], pts_seed_stats_bootstrap_mean[:,1], c=pts_seed_stats_bootstrap_mean[:,12], s=0.5, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(pts_seed_stats_bootstrap_mean[:,12], 10), vmax=np.nanpercentile(pts_seed_stats_bootstrap_mean[:,12], 90), linewidth=0)
        cax4 = ax4.imshow(dz_iqr, cmap=plt.get_cmap('Spectral'), vmin=np.nanpercentile(dz_iqr, 10), vmax=np.nanpercentile(dz_iqr, 90))
        ax4.set_title('Bootstrap: Surface roughness: Range of offsets from linear plane (IQR) with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax4)
        cbar.set_label('IQR Range of plane offsets (m)')
        ax4.set_xlabel('UTM-X (m)')
        ax4.set_ylabel('UTM-Y (m)')
        ax4.axis('equal')
    
        fig.savefig(pcl_seed_pts_fig_overview_fn, bbox_inches='tight')
        plt.close()
    print('done.')

if args.range_radii == 1:
    ### Equal number of nearest neighbor subsampling based on pcl_xyzg_density_min
    # Get PC density for each seed point using full point cloud
    pcl_xyzg_rstep_seed_p_min, pcl_xyzg_rstep_seed_p_median, \
        pcl_xyzg_rstep_seed_density_min, pcl_xyzg_rstep_seed_density_median = \
        pc_density(pcl_xyzg_rstep_seed, pcl_xyzg_ckdtree, nn=int(dxyzn_max_nre))
    
    #Determine number of points for each coordinate pts_xyz based on density
    median_of_density_min = np.median(pcl_xyzg_rstep_seed_density_min) #all points should have that density
    min_of_density_min = np.min(pcl_xyzg_rstep_seed_density_min) #all points should have that density
    sphere_area = args.sphere_radius_m * args.sphere_radius_m * np.pi
    nr_of_points_median = median_of_density_min*sphere_area
    nr_of_points_min = min_of_density_min*sphere_area
    
    #call random subsampling with equal number of points (median point density)
    pcl_xyzg_radius_equal_nr_random = \
        pc_random_equal_subsampling(pcl_xyzg_radius, pcl_xyzg_rstep_seed, int(nr_of_points_median))
    #pcl_xyzg_radius_equal_nr_random is a n x m x 3 array containing the coordinates of each neighbor coordinate
    #This can be used for further fitting of slope and curvature values
    
    
    ### Plot point densities of original and subsampled point cloud
    plt_point_cloud_densities(pcl_xyzg=pcl_xyzg, pcl_xyzg_rstep_seed=pcl_xyzg_rstep_seed,
                                  pcl_xyzg_p_random=pcl_xyzg_p_random, pcl_xyzg_p_min=pcl_xyzg_p_min,
                                  pcl_xyzg_density_min=pcl_xyzg_density_min, pcl_random_density_min=pcl_random_density_min,
                                  pcl_densities = '_pcl_densities_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m) )
        
    
    ### Initiate parallel run for calculating curvature over different radii and length scales using full point cloud (not subsampled)
    #set variables:
    dist_range=[1, 25, 1]
    dist_range_list = np.arange(dist_range[0], dist_range[1]+dist_range[2], dist_range[2])
    
    #make array for each point list
    nr_of_distance_stats = 8
    nr_of_curv_scale_stats = nr_of_distance_stats + 3 + 3
    nr_of_datasets_length_scale = nr_of_curv_scale_stats
    
    pc_length_scale_fn = os.path.join(args.outputdir, 'PC_length_scale_dist_%02d_to_%02dm.h5'%(dist_range[0], dist_range[1]))
    if os.path.exists(pc_length_scale_fn):
        hdf_in = h5py.File(pc_length_scale_fn,'r')
        curv_scale_stats = np.array(hdf_in['curv_scale_stats'])
    else:
        print('iterating through %d nearest neighbors (%d to %d m in %d-m steps) for %s points'%(len(dist_range_list), dist_range[0], dist_range[1], dist_range[2], "{:,}".format(pcl_xyzg_rstep_seed.T.shape[1])))
        if args.nr_of_cores == 0:
            args.nr_of_cores = multiprocessing.cpu_count()
        try:
            p
        except NameError:
            p = Pool(processes=np.round(args.nr_of_cores / 4).astype(int))
                #Use only 1/4 of the max. number of cores
        print('Using %d/%d cores for %d distance iterations'%(args.nr_of_cores, multiprocessing.cpu_count(), len(dist_range_list)))
        for i in range(len(dist_range_list)):
            pickle_merged_fn = os.path.join(pickle_dir, 'PC_length_scale_merged_{}.pickle'.format(str(i).zfill(3)))
            if os.path.exists(pickle_merged_fn):
                print('skipping, file already exists: %s.'%os.path.basename(pickle_merged_fn))
                continue
            points_k_idx = pcl_xyzg_ckdtree.query_ball_point(pcl_xyzg_rstep_seed, r=dist_range_list[i], n_jobs=-1)
            total_nr_of_points = np.sum([len(x) for x in points_k_idx])
            pos_array = np.array(np.linspace(0, len(pcl_xyzg_rstep_seed),nr_of_processes), dtype=int) #This creates a position array so you can select from:to in each loop
    
            print('Processing %s total points from %s seed points with max. distance=%d m. At %%: '%("{:,}".format(total_nr_of_points), "{:,}".format(len(points_k_idx)), dist_range_list[i]), end='', flush=True)
            ts = time.time()
    # single-core processing is the following for loop
    # NOTE that the python functions within calc_pts_length_scale_multicore also have some multi-core functionality
    #        for ii in np.arange(0,len(pos_array)-1):
    #            from_pos = pos_array[ii] #Get start/end from position array
    #            to_pos = pos_array[ii+1]
    #            subarr = np.arange(from_pos, to_pos) #Slice the data into the selected part...
    #            calc_pts_length_scale_singlecore(ii, subarr)
    
    # multi-core processing (on in p defined cores available cores)
            for _ in p.imap_unordered(calc_pts_length_scale_multicore, np.arange(0,len(pos_array)-1)):
                pass    
            print('\ntotal time: %0.3fs or %0.3fm'%(time.time() - ts, (time.time() - ts)/60))        
    
            #combine pickle files
            pkls = glob.glob(os.path.join(pickle_dir, 'PC_length_scale_0*')) #Now get all the pickle files we made
            pkls.sort() #make sure they're sorted
            curv_scale_stats_pickle = np.empty((points_k_idx.shape[0], nr_of_curv_scale_stats)) 
            curv_scale_stats_pickle.fill(np.nan)
            count = 0
            for fid in pkls:
                curv_scale_stats_res = pickle.load(open(fid,'rb')) #Loop through and put each pickle into the right place in the output array
                if curv_scale_stats_res.shape[0] != pos_array[count+1] - pos_array[count]:
                    print('File %s, length of records do not match. file: %d vs pos_array: %d'%(fid, curv_scale_stats_res.shape[0], pos_array[count+1] - pos_array[count]), flush=True)
                    if curv_scale_stats_res.shape[0] < pos_array[count+1] - pos_array[count]:
                        curv_scale_stats_pickle[range(pos_array[count],pos_array[count+1]-1),:] = curv_scale_stats_res
                    elif curv_scale_stats_res.shape[0] > pos_array[count+1] - pos_array[count]:
                        curv_scale_stats_pickle[range(pos_array[count],pos_array[count+1]),:] = curv_scale_stats_res[:-1]
                else:
                    curv_scale_stats_pickle[range(pos_array[count],pos_array[count+1]),:] = curv_scale_stats_res
                count += 1
                del curv_scale_stats_res
                #remove pickle files
            for j in pkls:
                os.remove(j)
            del pkls
            #write merged pickle 
            pickle.dump((curv_scale_stats_pickle), open(pickle_merged_fn,'wb'))
            del curv_scale_stats_pickle
            del points_k_idx, total_nr_of_points, ts, from_pos, to_pos, subarr
        #final merge of pickle files from all distances:
        curv_scale_stats = np.empty((dist_range_list.shape[0], nr_of_curv_scale_stats, np.shape(pcl_xyzg_rstep_seed)[0],))
        curv_scale_stats.fill(np.nan)
        pkls = glob.glob(os.path.join(pickle_dir, 'PC_length_scale_merged_0*')) #Now get all the pickle files we made
        pkls.sort() #make sure they're sorted
        count = 0
        for fid in pkls:
            curv_scale_stats_res = pickle.load(open(fid,'rb')) #Loop through and put each pickle into the right place in the output array
            curv_scale_stats[count,:,:] = curv_scale_stats_res.T
            count += 1
            del curv_scale_stats_res
        #remove pickle files
        for j in pkls:
            os.remove(j)
        del pkls
        #write final pickle 
        pickle_fn = os.path.join(pickle_dir, 'PC_length_scale_dist_%02d_to_%02d.pickle'%(dist_range[0], dist_range[1]))
        if os.path.exists(pickle_fn) == False:
            pickle.dump((curv_scale_stats), open(pickle_fn,'wb'))
        #write as HDF file
        hdf_out = h5py.File(pc_length_scale_fn,'w')
        hdf_out.attrs['help'] = 'Array from pc_geomorph_roughness.py:calc_pts_length_scale raster size %0.2fm and sphere radius %0.2fm. Curvature and slope calculated over distances from %d to %d in steps of %d.'%(args.raster_m, args.sphere_radius_m, dist_range[0], dist_range[1], dist_range[2])
        pts_length_scales_fc = hdf_out.create_dataset('curv_scale_stats',data=curv_scale_stats, chunks=True, compression="gzip", compression_opts=9)
        pts_length_scales_fc.attrs['help'] = 'Seed-pt shape: %d x %d,  curv_scale_stats shape: %d x %d x %d, with :,x,: Centroid-X, Centroid-Y, Centroid-Z, nr of points, dmin, dmax, dmean, dstd, d25p, d50p, d75p, slope, curvature, dz-residual'%(pcl_xyzg_rstep_seed.shape[0], pcl_xyzg_rstep_seed.shape[1], curv_scale_stats.shape[0], curv_scale_stats.shape[1], curv_scale_stats.shape[2])
        hdf_out.close()
        
        #plot
### Initiate parallel run for slope/density/curvature calculations for every seed point from raw point cloud (no homogenous point, no subsampling)
#generate seed HDF filename and load data from HDF file if available
if args.raw_pt_cloud == 1:
    intensity_mean_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_intensity_mean.tif'%(args.raster_m,args.sphere_radius_m))
    intensity_stdev_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_intensity_stdev.tif'%(args.raster_m,args.sphere_radius_m))
    intensity_iqr_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_intensity_iqr.tif'%(args.raster_m,args.sphere_radius_m))
    nrlidar_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_nrlidar.tif'%(args.raster_m,args.sphere_radius_m))
    dz_std_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_stddev.tif'%(args.raster_m,args.sphere_radius_m))
    dz_range9010_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_dz_range9010.tif'%(args.raster_m,args.sphere_radius_m))
    dz_iqr_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_dz_iqr.tif'%(args.raster_m,args.sphere_radius_m))
    plane_slope_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_planeslope.tif'%(args.raster_m,args.sphere_radius_m))
    dz_max_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_dzmax.tif'%(args.raster_m,args.sphere_radius_m))
    dz_min_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_dzmin.tif'%(args.raster_m,args.sphere_radius_m))
    plane_curv_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_curv.tif'%(args.raster_m,args.sphere_radius_m))
    pc_density_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_density.tif'%(args.raster_m,args.sphere_radius_m))
    height_mean_tif_fn = os.path.join(geotif_dir, os.path.basename(args.inlas).split('.')[0] + '_raw_%0.2fm_rsphere%0.2fm_height_mean.tif'%(args.raster_m,args.sphere_radius_m))
    seed_pts_stats_hdf = '_raw_seed_pts_stats_raster_%0.2fm_radius_%0.2fm.h5'%(args.raster_m, args.sphere_radius_m)
    pcl_seed_pts_stats_hdf_fn = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_hdf)

    #Use all points, including vegetation points
    pcl_xyzig = pcl_xyzic[:,0:4]
    pcl_xyzg = pcl_xyzic[:,0:3]
    pcl_xyg = pcl_xyzic[:,0:2]
    nr_of_points = int(pcl_xyzig.shape[0]/2)

    print('\nGenerating XY-cKDTree for raw point cloud... ',end='', flush=True)
    pcl_xyg_ckdtree_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + '_xy_cKDTree.pickle')
    if os.path.exists(pcl_xyg_ckdtree_fn):
        pcl_xyg_ckdtree = pickle.load(open( pcl_xyg_ckdtree_fn, "rb" ))
    else:
        pcl_xyg_ckdtree = spatial.cKDTree(pcl_xyg, leafsize=32)
        pickle.dump(pcl_xyg_ckdtree, open(pcl_xyg_ckdtree_fn,'wb'))
    print('done.')
    
    print('Generating XYZ-cKDTree for raw point cloud... ',end='',flush=True)
    pcl_xyzg_ckdtree_fn = os.path.join(pickle_dir, os.path.basename(args.inlas).split('.')[0] + '_xyz_cKDTree.pickle')
    if os.path.exists(pcl_xyzg_ckdtree_fn):
        pcl_xyzg_ckdtree = pickle.load(open( pcl_xyzg_ckdtree_fn, "rb" ))
    else:
        pcl_xyzg_ckdtree = spatial.cKDTree(pcl_xyzg, leafsize=32)
        pickle.dump(pcl_xyzg_ckdtree,open(pcl_xyzg_ckdtree_fn,'wb'))
    print('done.')

    if args.dem_fname != '':
        try:
            src_ds = gdal.Open( args.dem_fname )
        except RuntimeError:
            print('Unable to open {}'.format(args.dem_fname))
            sys.exit(1)
    
        bandnr = 1
        try:
            src_band = src_ds.GetRasterBand(bandnr)
        except RuntimeError:
            print('Band ( {} ) not found in {}'.format(bandnr, args.dem_fname))
            sys.exit(1)
    
        #print("{}: [ MIN ] = {}, [ MAX ] = {}".format(os.path.basename(args.dem_fname), src_band.GetMinimum(), src_band.GetMaximum()))
        cols = src_band.XSize
        rows = src_band.YSize
        geo_transform = src_ds.GetGeoTransform()
        x_min = geo_transform[0] 
        x_max = geo_transform[0] + geo_transform[1] * cols
        y_min = geo_transform[3] 
        y_max = geo_transform[3] + geo_transform[5] * rows
    
        x_elements = len(np.arange(x_min, x_max, args.raster_m))
        if y_min > y_max:
            y_elements = len(np.arange(y_max, y_min, args.raster_m))
            y_coords = np.arange(y_max, y_min, args.raster_m) + args.raster_m / 2
        else:
            y_elements = len(np.arange(y_min, y_max, args.raster_m))
            y_coords = np.arange(y_min, y_max, args.raster_m) + args.raster_m / 2
        
        #get coordinate range and shift coordinates by half of the step size to make sure rater overlay is centered. 
        #This is not really necessary and only matters for very small point clouds with edge effects or for very large steps sizes:
        x_coords = np.arange(x_min, x_max, args.raster_m) + args.raster_m / 2
        
        #create combination of all coordinates (this is using lists and could be optimized)
        xy_coordinates = np.array([(x,y) for x in x_coords for y in y_coords])
    else:
        #no GeoTiff file given, using min/max coordinates to generate equally-spaced grid
            
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
    [pcl_xyg_ckdtree_distance, pcl_xyg_ckdtree_id] = pcl_xyg_ckdtree.query(xy_coordinates, k=1, n_jobs=-1)
    
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
    
    #Make sure to read in original point cloud
    pcl_xyzg_radius = pcl_xyzg_ckdtree.query_ball_point(pcl_xyzg_rstep_seed, r=args.sphere_radius_m, n_jobs=-1)
    pcl_xyzg_radius_nr = pcl_xyzg_radius.shape[0]
    dxyzn_max_nre = np.max([len(x) for x in pcl_xyzg_radius]) #extract list of neighboring points
    dxyzn_nre = np.sum([len(x) for x in pcl_xyzg_radius])
    #dxyzn_nre_pos_array = np.array(np.linspace(0, dxyzn_nre, nr_of_processes), dtype=int)
    
    # Find sphere points for slope and curvature calculation (if slope and curvature radius is different) from sphere_radius_m
    if args.slope_sphere_radius_m != args.sphere_radius_m:    
        if os.path.exists(pcl_xyzg_radius2_fn):
            pcl_xyzg_radius_slope = pcl_xyzg_ckdtree.query_ball_point(pcl_xyzg_rstep_seed,r=args.slope_sphere_radius_m, n_jobs=-1)
    elif args.slope_sphere_radius_m == args.sphere_radius_m:
        pcl_xyzg_radius_slope = pcl_xyzg_radius

    ts = time.time()
    print('\nCalculating seed point statistics from raw point cloud (not homogenized, not subsampled) for %s seed points from %s points'%("{:,}".format(pcl_xyzg_radius.shape[0]), "{:,}".format(pcl_xyzg_ckdtree.n)) )
    if os.path.exists(pcl_seed_pts_stats_hdf_fn) == False:
        total_nr_of_points = pcl_xyzg.shape[0]    
        pos_array = np.array(np.linspace(0, len(pcl_xyzg_rstep_seed), nr_of_processes), dtype=int) #This creates a position array so you can select from:to in each loop

        #calculating density for seed points
        _, _, pc_density_min, _ = \
        pc_density(pcl_xyzg_rstep_seed, pcl_xyzg_ckdtree, nn=int(dxyzn_max_nre) )
    
        if args.nr_of_cores != 0:
            p = Pool(processes=args.nr_of_cores)
        else:
            args.nr_of_cores = multiprocessing.cpu_count()
            p = Pool(processes=int(multiprocessing.cpu_count()/2))
        
        print('Using %d/%d cores '%(int(multiprocessing.cpu_count()/2), args.nr_of_cores))
        print('\nAt %d/%d [%%]: '%(1, 1), end='', flush=True)
        for _ in p.imap_unordered(calc_stats_for_seed_points_wrapper, np.arange(0,len(pos_array)-1)):
            pass    
    
        #Single core:
        #for i in np.arange(0,len(pos_array)-1):
        #    calc_stats_for_seed_points_wrapper(i)
            
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
        
        #add percentiles of density information stored in p_percentiles
        #where is pc_density_results
        #pts_seed_stats = np.c_[pts_seed_stats, pc_density_result]
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
        pc_density_min = np.array(hdf_in['pc_density_min'])        
        print('Statistics loaded from file: %s'%os.path.basename(pcl_seed_pts_stats_hdf_fn))
    print('done.')
    print('total time: %0.3fs or %0.3fm'%(time.time() - ts, (time.time() - ts)/60))    
    
    print('\nWriting seed points and statistics to HDF, CSV, and shapefiles... ', end='', flush=True)
    ### Write Seed point statistics to file
    if os.path.exists(pcl_seed_pts_stats_hdf_fn) == False:
        hdf_out = h5py.File(pcl_seed_pts_stats_hdf_fn,'w')
        hdf_out.attrs['help'] = 'Array from pc_dh_roughness.py with raster size %0.2fm and sphere radius %0.2fm'%(args.raster_m, args.sphere_radius_m)
        pts_seeds_stats_fc = hdf_out.create_dataset('pts_seed_stats',data=pts_seed_stats, chunks=True, compression="gzip", compression_opts=7)
        pts_seeds_stats_fc.attrs['help'] = '''Nr. of seed pts: %d,  (col: name) 
        0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  
        10: Dz-std.dev, 11: Dz-range, 12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: slope of fitted plane, 
        15: plane residuals, 16: variance dz, 17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals, 
        21:intensity mean, 22:intensity std., 23: intensity median, 24: 10, 25: 27, 26: 75, 27: 90th percentile intensity'''\
        %(pcl_xyzg_rstep_seed.shape[0])
        dxyzn_fc = hdf_out.create_dataset('dxyzn',data=dxyzn, chunks=True, compression="gzip", compression_opts=7)
        dxyzn_fc.attrs['help'] = 'Lidar points and their deviation from a plane with radius %0.2f'%args.sphere_radius_m
        pc_density_min_fc = hdf_out.create_dataset('pc_density_min',data=pc_density_min, chunks=True, compression="gzip", compression_opts=7)
        pc_density_min_fc.attrs['help'] = 'PC density for seed points with radius %0.2f'%args.sphere_radius_m
        hdf_out.close()
    
    #write csv
    header_str='1SeedX, 2SeedY, 3SeedZ, 4MeanX, 5MeanY, 6MeanZ, 7Z_min, 8Z_max, 9Dz_max, 10Dz_min, 11Dz_std, 12Dz_range, 13Dz_9010p, 14Dz_7525p, 15_Pl_slp, 16Pl_res, 17Pl_Var, 18Nr_lidar, 19SlopeLSQ, 20CurvLSQ, 21CurvRes, 22I_mean, 23I_std, 24I_med, 25I_p10, 26I_p25, 27I_p75, 28I_p90'
    seed_pts_stats_csv = '_raw_seed_pts_stats_raster_%0.2fm_radius_%0.2fm.csv'%(args.raster_m, args.sphere_radius_m)
    pcl_seed_pts_stats_csv_fn = os.path.join(args.outputdir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_csv)
    seed_pts_stats_vrt = '_raw_seed_pts_stats_raster_%0.2fm_radius_%0.2fm.vrt'%(args.raster_m, args.sphere_radius_m)
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
    vrt_f.write('\t\t\t<Field name="19SlopeLSQ" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="20CurvLSQ" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="21CurvRes" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="22I_mean" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="23I_std" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="24I_med" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="25I_p10" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="26I_p25" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="27I_p75" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t\t\t<Field name="28I_p90" type="Real" width="8" precision="7"/>\n')
    vrt_f.write('\t</OGRVRTLayer>\n')
    vrt_f.write('</OGRVRTDataSource>\n')
    vrt_f.close()
    
    # Generate shapefile from vrt
    if os.path.exists(pcl_seed_pts_stats_vrt_fn) == False:
        cwd=os.getcwd()
        os.chdir(args.outputdir)
        cmd = ['ogr2ogr', pcl_seed_pts_stats_vrt_fn, pcl_seed_pts_stats_vrt_fn]
        logfile_fname = os.path.join(args.outputdir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt'
        logfile_error_fname = os.path.join(args.outputdir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt'
        with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
            subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
            subprocess_p.wait()
        os.chdir(cwd)
    print('done.')

    ### Interpolate to equally-spaced grid and generate GeoTIFF output
    print('\nInterpolating seed points (mean-X, mean-Y) to geotiff rasters and writing geotiff raster... ',end='', flush=True)
    xx,yy = np.meshgrid(x_coords,y_coords)
    idx_nonan = np.where(np.isnan(pts_seed_stats_bootstrap_mean[:,3])==False)
    points = np.hstack((pts_seed_stats_bootstrap_mean[idx_nonan,3].T, pts_seed_stats_bootstrap_mean[idx_nonan,4].T))

    ncols=cols
    nrows=rows
    xres = args.raster_m
    yres = args.raster_m
    geotransform = (x_coords.min() - (args.raster_m / 2),xres, 0 , y_coords.min() - (args.raster_m / 2),0, yres) 
    
    #interpolate lidar intensity mean
    if os.path.exists(intensity_mean_tif_fn) == False:
        print('intensity_mean, ', end='', flush=True)
        intensity_mean = griddata_clip_geotif(intensity_mean_tif_fn, points, pts_seed_stats[idx_nonan,21][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(intensity_mean_tif_fn)
        intensity_mean = np.array(ds.GetRasterBand(1).ReadAsArray())
        intensity_mean[np.where(intensity_mean == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate lidar intensity IQR
    if os.path.exists(intensity_iqr_tif_fn) == False:
        print('intensity_IQR, ', end='', flush=True)
        intensity_iqr = griddata_clip_geotif(intensity_iqr_tif_fn, points, pts_seed_stats[idx_nonan,26][0]-pts_seed_stats[idx_nonan,25][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(intensity_iqr_tif_fn)
        intensity_iqr = np.array(ds.GetRasterBand(1).ReadAsArray())
        intensity_iqr[np.where(intensity_iqr == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate mean height
    if os.path.exists(height_mean_tif_fn) == False:
        print('mean height, ', end='', flush=True)
        height_mean = griddata_clip_geotif(height_mean_tif_fn, points, pts_seed_stats[idx_nonan,2][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(height_mean_tif_fn)
        height_mean = np.array(ds.GetRasterBand(1).ReadAsArray())
        height_mean[np.where(height_mean == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate nr_lidar_measurements
    if os.path.exists(nrlidar_tif_fn) == False:
        print('nr_lidar_measurements, ', end='', flush=True)
        nr_lidar = griddata_clip_geotif(nrlidar_tif_fn, points, pts_seed_stats[idx_nonan,17][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(nrlidar_tif_fn)
        nr_lidar = np.array(ds.GetRasterBand(1).ReadAsArray())
        nr_lidar[np.where(nr_lidar == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    if os.path.exists(dz_std_tif_fn) == False:
        print('Dz std. dev., ', end='', flush=True)
        dz_std = griddata_clip_geotif(dz_std_tif_fn, points, pts_seed_stats[idx_nonan,10][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_std_tif_fn)
        dz_std = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_std[np.where(dz_std == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz_range 90-10 percentile
    if os.path.exists(dz_range9010_tif_fn) == False:
        print('range (90-10th perc.), ', end='', flush=True)
        dz_range9010 = griddata_clip_geotif(dz_range9010_tif_fn, points, pts_seed_stats[idx_nonan,12][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_range9010_tif_fn)
        dz_range9010 = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_range9010[np.where(dz_range9010 == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz_range 75-25 percentile
    if os.path.exists(dz_iqr_tif_fn) == False:
        print('range (75-25th perc.), ', end='', flush=True)
        dz_iqr = griddata_clip_geotif(dz_iqr_tif_fn, points, pts_seed_stats[idx_nonan,13][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_iqr_tif_fn)
        dz_iqr = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_iqr[np.where(dz_iqr == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz-max
    if os.path.exists(dz_max_tif_fn) == False:
        print('max. dz, ', end='', flush=True)
        dz_max = griddata_clip_geotif(dz_max_tif_fn, points, pts_seed_stats[idx_nonan,8][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_max_tif_fn)
        dz_max = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_max[np.where(dz_max == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Dz-min
    if os.path.exists(dz_min_tif_fn) == False:
        print('min. dz, ', end='', flush=True)
        dz_min = griddata_clip_geotif(dz_min_tif_fn, points, pts_seed_stats[idx_nonan,9][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(dz_min_tif_fn)
        dz_min = np.array(ds.GetRasterBand(1).ReadAsArray())
        dz_min[np.where(dz_min == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate Plane_slope
    if os.path.exists(plane_slope_tif_fn) == False:
        print('plane slope, ', end='', flush=True)
        plane_slope = griddata_clip_geotif(plane_slope_tif_fn, points, pts_seed_stats[idx_nonan,14][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(plane_slope_tif_fn)
        plane_slope = np.array(ds.GetRasterBand(1).ReadAsArray())
        plane_slope[np.where(plane_slope == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate LST-Square curvate
    if os.path.exists(plane_curv_tif_fn) == False:
        print('LSTSQ curvature, ', end='', flush=True)
        plane_curv = griddata_clip_geotif(plane_curv_tif_fn, points, pts_seed_stats[idx_nonan,19][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(plane_curv_tif_fn)
        plane_curv = np.array(ds.GetRasterBand(1).ReadAsArray())
        plane_curv[np.where(plane_curv == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    #interpolate PC Density
    if os.path.exists(pc_density_tif_fn) == False:
        print('PC density, ', end='', flush=True)
        pc_density_grid = griddata_clip_geotif(pc_density_tif_fn, points, pc_density_min[idx_nonan], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    else:
        ds = gdal.Open(pc_density_tif_fn)
        pc_density_grid = np.array(ds.GetRasterBand(1).ReadAsArray())
        pc_density_grid[np.where(pc_density_grid == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    
    print('done.')
       
    # Could use gdal_grid to generate TIF from VRT/Shapefile:
    #gdal_grid -zfield "15Nr_lidar" -outsize 271 280 -a linear:radius=2.0:nodata=-9999 -of GTiff -ot Int16 Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_seed_pts_stats_raster_1.00m_radius_1.50m.vrt Pozo_USGS_UTM11_NAD83_all_color_cl2_SC12_seed_pts_stats_raster_1.00m_radius_1.50m_nrlidar2.tif --config GDAL_NUM_THREADS ALL_CPUS -co COMPRESS=DEFLATE -co ZLEVEL=9
    
    ### Plot output to figures
    seed_pts_stats_png = '_raw_seed_pts_overview1_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
    pcl_seed_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_png)
    print('\nGenerating overview figure 1 %s... '%os.path.basename(pcl_seed_pts_fig_overview_fn), end='', flush=True)
    if os.path.exists(pcl_seed_pts_fig_overview_fn) == False:
        fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
        #fig = plt.figure(figsize=(11.69,8.27), dpi=150)
        fig.clf()
        
        ax1 = fig.add_subplot(231)
        ax1.grid()
        cax1 = ax1.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,2], s=0.5, cmap=plt.get_cmap('terrain'), linewidth=0)
        ax1.set_title('Lidar seed point elevation with raster=%0.2fm'%args.raster_m,y=1.05)
        cbar = fig.colorbar(cax1)
        cbar.set_label('Mean elevation in sphere (m)')
        ax1.set_xlabel('UTM-X (m)')
        ax1.set_ylabel('UTM-Y (m)')
        ax1.axis('equal')
    
        ax2 = fig.add_subplot(232)
        ax2.grid()
        cax2 = ax2.imshow(nr_lidar,cmap=plt.get_cmap('gnuplot'))
#        ax2.scatter(pts_seed_stats[:,3], pts_seed_stats[:,4], c=pts_seed_stats[:,17], s=0.5, cmap=plt.get_cmap('gnuplot'), linewidth=0)    
        ax2.set_title('Nr. of lidar measurements for each seed point with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax2)
        cbar.set_label('nr. of lidar measurements')
        ax2.set_xlabel('UTM-X (m)')
        ax2.set_ylabel('UTM-Y (m)')
        ax2.axis('equal')
        
        ax3 = fig.add_subplot(233)
        ax3.grid()
        #cax3 = ax3.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,13], s=0.5, cmap=plt.get_cmap('seismic'), vmin=np.nanpercentile(pts_seed_stats[:,13], 10), vmax=np.nanpercentile(pts_seed_stats[:,13], 90), linewidth=0)
        cax3 = ax3.imshow(plane_slope, cmap=plt.get_cmap('seismic'), vmin=np.nanpercentile(plane_slope, 10), vmax=np.nanpercentile(plane_slope, 90))
        ax3.set_title('Slope of fitted plane for each sphere with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax3)
        cbar.set_label('Slope (m/m)')
        ax3.set_xlabel('UTM-X (m)')
        ax3.set_ylabel('UTM-Y (m)')
        ax3.axis('equal')
    
        ax5 = fig.add_subplot(234)
        ax5.grid()
        cax5 = ax5.imshow(plane_curv, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(plane_curv, 10), vmax=np.nanpercentile(plane_curv, 90))
        ax5.set_title('Curvature of sphere/disc with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax5)
        cbar.set_label('Curvature (1/m)')
        ax5.set_xlabel('UTM-X (m)')
        ax5.set_ylabel('UTM-Y (m)')
        ax5.axis('equal')
    
        ax4 = fig.add_subplot(235)
        ax4.grid()
        #cax4 = ax4.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,12], s=0.5, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(pts_seed_stats[:,12], 10), vmax=np.nanpercentile(pts_seed_stats[:,12], 90), linewidth=0)
        cax4 = ax4.imshow(dz_range9010, cmap=plt.get_cmap('Spectral'), vmin=np.nanpercentile(dz_range9010, 10), vmax=np.nanpercentile(dz_range9010, 90))
        ax4.set_title('Surface roughness: Range of offsets from linear plane (90-10th percentile) with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax4)
        cbar.set_label('Range (90-10th) of plane offsets (m)')
        ax4.set_xlabel('UTM-X (m)')
        ax4.set_ylabel('UTM-Y (m)')
        ax4.axis('equal')
    
    
        ax6 = fig.add_subplot(236)
        ax6.grid()
        #cax6 = ax6.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,10], s=0.5, cmap=plt.get_cmap('jet'), vmin=np.nanpercentile(pts_seed_stats[:,10], 10), vmax=np.nanpercentile(pts_seed_stats[:,10], 90), linewidth=0)
        cax6 = ax6.imshow(intensity_mean, cmap=plt.get_cmap('gray'), vmin=np.nanpercentile(intensity_mean, 10), vmax=np.nanpercentile(intensity_mean, 90))
        ax6.set_title('Point Cloud intensity for k=%d nearest neighbors and r=%0.2f'%(dxyzn_max_nre, args.sphere_radius_m),y=1.05)
        cbar = fig.colorbar(cax6)
        cbar.set_label('Intensity (mean)')
        ax6.set_xlabel('UTM-X (m)')
        ax6.set_ylabel('UTM-Y (m)')
        ax6.axis('equal')
        
        fig.savefig(pcl_seed_pts_fig_overview_fn, bbox_inches='tight')
        plt.close()
    print('done.')
    
    seed_pts_stats_png = '_raw_seed_pts_overview2_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
    pcl_seed_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + seed_pts_stats_png)
    print('\nGenerating overview figure 2 %s... '%os.path.basename(pcl_seed_pts_fig_overview_fn), end='', flush=True)
    if os.path.exists(pcl_seed_pts_fig_overview_fn) == False:
        fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
        #fig = plt.figure(figsize=(11.69,8.27), dpi=150)
        fig.clf()
        
        ax1 = fig.add_subplot(221)
        ax1.grid()
        cax1 = ax1.imshow(intensity_iqr, cmap=plt.get_cmap('jet'), vmin=np.nanpercentile(intensity_iqr, 10), vmax=np.nanpercentile(intensity_iqr, 90))
        ax1.set_title('Point Cloud intensity IQR for k=%d nearest neighbors and r=%0.2f'%(dxyzn_max_nre, args.sphere_radius_m),y=1.05)
        cbar = fig.colorbar(cax1)
        cbar.set_label('Intensity (IQR)')
        ax1.set_xlabel('UTM-X (m)')
        ax1.set_ylabel('UTM-Y (m)')
        ax1.axis('equal')
        
        ax2 = fig.add_subplot(222)
        ax2.grid()
        cax2 = ax2.imshow(pc_density_grid, cmap=plt.get_cmap('gnuplot'), vmin=np.nanpercentile(pc_density_grid, 10), vmax=np.nanpercentile(pc_density_grid, 90))
        ax2.set_title('Point Cloud density in pts/m^2 for r=%0.2f'%(args.sphere_radius_m),y=1.05)
        cbar = fig.colorbar(cax2)
        cbar.set_label('PC density (pts/m^2)')
        ax2.set_xlabel('UTM-X (m)')
        ax2.set_ylabel('UTM-Y (m)')
        ax2.axis('equal')

        ax4 = fig.add_subplot(223)
        ax4.grid()
        #cax4 = ax4.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,12], s=0.5, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(pts_seed_stats[:,12], 10), vmax=np.nanpercentile(pts_seed_stats[:,12], 90), linewidth=0)
        cax4 = ax4.imshow(dz_iqr, cmap=plt.get_cmap('Spectral'), vmin=np.nanpercentile(dz_iqr, 10), vmax=np.nanpercentile(dz_iqr, 90))
        ax4.set_title('Surface roughness: Range of offsets from linear plane (IQR) with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax4)
        cbar.set_label('IQR Range of plane offsets (m)')
        ax4.set_xlabel('UTM-X (m)')
        ax4.set_ylabel('UTM-Y (m)')
        ax4.axis('equal')    
  
        fig.savefig(pcl_seed_pts_fig_overview_fn, bbox_inches='tight')
        plt.close()
    print('done.')
    
    nrlidar_pts_stats_png = '_raw_nrlidar_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
    pcl_nrlidar_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + nrlidar_pts_stats_png)
    print('Generating figure for number of lidar measurement: %s... '%os.path.basename(pcl_nrlidar_pts_fig_overview_fn), end='', flush=True)
    if os.path.exists(pcl_seed_pts_fig_overview_fn) == False:
        fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
        fig.clf()
        
        ax2 = fig.add_subplot(111)
        ax2.grid()
        cax2 = ax2.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,15], s=3, cmap=plt.get_cmap('gnuplot'), linewidth=0)
        ax2.set_title('PC: All points - Nr. of lidar measurements for each seed point',y=1.05)
        cbar = fig.colorbar(cax2)
        cbar.set_label('#')
        ax2.set_xlabel('UTM-X (m)')
        ax2.set_ylabel('UTM-Y (m)')
        ax2.axis('equal')
        fig.savefig(pcl_nrlidar_pts_fig_overview_fn, bbox_inches='tight')
        plt.close()
    print('done.')
    
    slope_pts_stats_png = '_raw_slope_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
    pcl_slope_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + slope_pts_stats_png)
    print('Generating figure for slope of fitted plane: %s... '%os.path.basename(pcl_slope_pts_fig_overview_fn), end='', flush=True)
    if os.path.exists(pcl_slope_pts_fig_overview_fn) == False:
        fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
        fig.clf()
        
        ax3 = fig.add_subplot(111)
        ax3.grid()
        cax3 = ax3.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,14], s=3, cmap=plt.get_cmap('seismic'), vmin=np.nanpercentile(pts_seed_stats[:,14], 10), vmax=np.nanpercentile(pts_seed_stats[:,14], 90), linewidth=0)
        ax3.set_title('PC: All points - Slope of fitted plane for each sphere (r=%0.2f)'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax3)
        cbar.set_label('Slope (m/m)')
        ax3.set_xlabel('UTM-X (m)')
        ax3.set_ylabel('UTM-Y (m)')
        ax3.axis('equal')
        fig.savefig(pcl_slope_pts_fig_overview_fn, bbox_inches='tight')
        plt.close()
    print('done.')
    
    sroughness9010p_pts_stats_png = '_raw_sroughness9010p_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
    pcl_sroughness9010p_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + sroughness9010p_pts_stats_png)
    print('Generating figure for range off offsets from linear plane (90-10th p): %s... '%os.path.basename(pcl_sroughness9010p_pts_fig_overview_fn), end='', flush=True)
    if os.path.exists(pcl_sroughness9010p_pts_fig_overview_fn) == False:
        fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
        fig.clf()
        
        ax4 = fig.add_subplot(111)
        ax4.grid()
        cax4 = ax4.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,12], s=3, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(pts_seed_stats[:,12], 10), vmax=np.nanpercentile(pts_seed_stats[:,12], 90), linewidth=0)
        ax4.set_title('PC: All points - Surface roughness I: Range of offsets from linear plane (90-10th percentile) with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax4)
        cbar.set_label('Range (90-10th) of plane offsets (m)')
        ax4.set_xlabel('UTM-X (m)')
        ax4.set_ylabel('UTM-Y (m)')
        ax4.axis('equal')
        fig.savefig(pcl_sroughness9010p_pts_fig_overview_fn, bbox_inches='tight')
        plt.close()
    print('done.')
    
    curv_pts_stats_png = '_raw_curv_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
    pcl_curv_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + curv_pts_stats_png)
    print('Generating figure for range off offsets from linear plane (75-25th p): %s... '%os.path.basename(pcl_curv_pts_fig_overview_fn), end='', flush=True)
    if os.path.exists(pcl_curv_pts_fig_overview_fn) == False:
        fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
        fig.clf()
        
        ax4 = fig.add_subplot(111)
        ax4.grid()
        cax4 = ax4.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,18], s=3, cmap=plt.get_cmap('PiYG'), vmin=np.nanpercentile(pts_seed_stats[:,18], 10), vmax=np.nanpercentile(pts_seed_stats[:,18], 90), linewidth=0)
        ax4.set_title('PC: All points - Curvature of sphere/disc with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax4)
        cbar.set_label('Curvatre (1/m)')
        ax4.set_xlabel('UTM-X (m)')
        ax4.set_ylabel('UTM-Y (m)')
        ax4.axis('equal')
        fig.savefig(pcl_curv_pts_fig_overview_fn, bbox_inches='tight')
        plt.close()
    print('done.')
    
    stddev_pts_stats_png = '_raw_stddev_pts_overview_raster_%0.2fm_radius_%0.2fm.png'%(args.raster_m, args.sphere_radius_m)
    pcl_stddev_pts_fig_overview_fn = os.path.join(figure_dir, os.path.basename(args.inlas).split('.')[0] + stddev_pts_stats_png)
    print('Generating figure for std. deviation of points from fitted plane: %s... '%os.path.basename(pcl_stddev_pts_fig_overview_fn), end='', flush=True)
    if os.path.exists(pcl_stddev_pts_fig_overview_fn) == False:
        fig = plt.figure(figsize=(16.53*1.5,11.69*1.5), dpi=150)
        fig.clf()
        
        ax5 = fig.add_subplot(111)
        ax5.grid()
        cax5 = ax5.scatter(pts_seed_stats[:,0], pts_seed_stats[:,1], c=pts_seed_stats[:,10], s=3, cmap=plt.get_cmap('jet'), vmin=np.nanpercentile(pts_seed_stats[:,10], 10), vmax=np.nanpercentile(pts_seed_stats[:,10], 90), linewidth=0)
        ax5.set_title('PC: All points - Std. deviation of lidar points from plane with r=%0.2f'%args.sphere_radius_m,y=1.05)
        cbar = fig.colorbar(cax5)
        cbar.set_label('Std. deviation (m)')
        ax5.set_xlabel('UTM-X (m)')
        ax5.set_ylabel('UTM-Y (m)')
        ax5.axis('equal')
        fig.savefig(pcl_stddev_pts_fig_overview_fn, bbox_inches='tight')
        plt.close()
    print('done.')

### Write to LAS/LAZ file
if args.store_color == True:
    if os.path.exists(args.outlas) == False:
        print('\nWriting dz values to LAS file: %s... '%args.outlas, end='', flush=True)    
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

