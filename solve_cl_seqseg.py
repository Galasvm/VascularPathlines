
import os
from centerlines_seqseg import *
import time


time_start = time.time()

parent = os.path.dirname(__file__)

mesh_name = "/0131_0001_aorta_coarse"
mesh_dir = parent+ "/Meshes" + mesh_name + mesh_name
save_dir = parent + "/results/10292024/Numi_seqseg" + mesh_name
point_source = 133853



domain, facet = import_mesh(mesh_dir + ".xdmf",
                            mesh_dir + "_facet_markers.xdmf")

hmax,hmin,havg = h_max(domain)
initial_dis_threshold = hmax*0.8

dis = solve_eikonal(domain, 1, 1,1)
export_soln(save_dir + "/eikonal" + mesh_name + "_distance_field.xdmf", domain, dis)
if point_source == None:
    point_source = input("please enter the point source (open the distance field in ParaView and find the ID): ")
    
dtf_mod_speed = solve_eikonal(domain,2,1,point_source,dis)
dtf_extreme_nodes = solve_eikonal(domain,2,2,point_source,dis) # too many extreme nodes selected
extreme_nodes= gala_extreme_nodes(domain,dtf_extreme_nodes,havg*3)
cluster_graph = discritize_dtf(dtf_mod_speed,domain,60)
spatial_cluster_graph, extreme = separate_clusters(domain, cluster_graph, initial_dis_threshold)
rescale_dis, rescale_dis_array = rescale_distance_map(domain, spatial_cluster_graph, dis)

point_dtf_map = solve_eikonal(domain, 2, 3, point_source, rescale_dis)
export_vtk(save_dir + "/eikonal" + mesh_name + "_destination_time.vtu", point_dtf_map)
print(f"there are {len(extreme)} extreme nodes: {extreme}")


export_soln(save_dir + "/eikonal" + mesh_name + "_distance_field.xdmf", domain, dis)
export_soln(save_dir + "/eikonal" + mesh_name + "_dtf_moderate_speed.xdmf", domain, dtf_mod_speed)
export_soln(save_dir + "/eikonal" + mesh_name + "_rescale_distance_field.xdmf", domain, rescale_dis)
# export_soln(save_dir + "/eikonal" + mesh_name + "_dtf_high_speed.xdmf", domain, dft_high_speed)

export_soln(save_dir + "/cluster" + mesh_name + "_clustermap.xdmf", domain, cluster_graph)
export_soln(save_dir + "/cluster" + mesh_name + "_spatial_clustermap.xdmf", domain, spatial_cluster_graph)

centerline_polydata = combinie_cl(domain, extreme_nodes, spatial_cluster_graph, save_dir + "/eikonal" + mesh_name + "_destination_time_p0_000000.vtu")

# centerline_polydata = combining_cl(domain, spatial_cluster_graph, extreme, rescale_dis, save_dir + "/eikonal" + mesh_name + "_destination_time_p0_000000.vtu")
save_centerline_vtk(centerline_polydata, save_dir + "/centerlines" + mesh_name + "_centerline.vtp")
centerline_merged = merge_centerline_segments(centerline_polydata)
_, dict_cell = create_dict(centerline_merged)
tolerance = get_subdivided_cl_spacing(dict_cell)
print(tolerance)
smooth_centerline_polydata,_,_,_ = combine_cls_into_one_polydata(dict_cell, tolerance/2)
save_centerline_vtk(smooth_centerline_polydata, save_dir + "/centerlines" + mesh_name + "smooth_centerline.vtp")
