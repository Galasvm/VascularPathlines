import yaml
import os
from centerline_extraction import *
import time

time_start = time.time()

yaml_file = "0100_A_AO_COA.yaml"

# Determine parent folder
parent = os.path.dirname(__file__)

# Read yaml file located in simvascular_mesh/yaml_files
with open(parent + "/yaml_files/" + yaml_file, "rb") as f:
    params = yaml.safe_load(f)

yaml_file_name = params["file_name"]
save_dir = parent + params["saving_dir"] + yaml_file_name
mesh_dir = parent + "/Meshes" + yaml_file_name + yaml_file_name
 
domain, facet = import_mesh(mesh_dir + ".xdmf",
                            mesh_dir + "_facet_markers.xdmf")
# hmax,hmin,havg = h_max(domain)
# print(f"hmax: {hmax}, hmin: {hmin}, havg: {havg}")
edgemax,edgemin,edgeavg = edge_max(domain)
print(f"edgemax: {edgemax}, edgemin: {edgemin}, edgeavg: {edgeavg}")


initial_dis_threshold = edgemax*0.8

dis = solve_eikonal(domain, 1, 1, 1)
export_vtk(save_dir + "/eikonal" + yaml_file_name + "_dis_map.vtu", dis)
# export_soln(save_dir + "/eikonal" + yaml_file_name + "_dis_map.xdmf", domain, dis)
point_index = params["manual_ps"]
dtf_mod_speed = solve_eikonal(domain,2,1,point_index,dis)
# export_soln(save_dir + "/eikonal" + yaml_file_name + "_dtf_mod_speed_map.xdmf", domain, dtf_mod_speed)

geometry_type=params["type"]
# geometry_type=input("please enter the geometry type (aorta, pulm, cere, or coro): ")


cluster_graph = discritize_dtf(dtf_mod_speed, domain, geometry_type)
cluster_separate, extreme = separate_clusters(domain, cluster_graph, initial_dis_threshold, 30)
export_soln(save_dir + "/cluster" + yaml_file_name + "_cluster_map.xdmf", domain, cluster_separate)
rescale_dis, rescale_dis_array = rescale_distance_map(domain, cluster_separate, dis)
export_soln(save_dir + "/eikonal" + yaml_file_name + "_rescale_dis_map.xdmf", domain, rescale_dis)

################### subclustering
# second_dis_threshold = hmax*0.5
# subcluster_map = subclusters(domain, cluster_separate, rescale_dis,dtf_mod_speed)
# subclsuter_separate, extreme_clusters_ = separate_clusters(domain, subcluster_map,second_dis_threshold,30)

##################

# dtf_map_extreme_nodes = solve_eikonal(domain, 2, 2,point_index, dis) # GALA CHANGED THIS
# export_soln(save_dir + "/eikonal" + yaml_file_name + "_dtf_inverse_speed_map.xdmf", domain, dtf_map_extreme_nodes)
extreme_nodes, surrounding_dofs = gala_extreme_nodes(domain,dtf_mod_speed,edgeavg*4) # CHANGEDDDDDDDDGALA CHANGED THIS
#############################################

# final_dis_map = solve_eikonal(domain,3,1,extreme_nodes,surrounding_dofs )
# export_soln(save_dir + "/eikonal" + yaml_file_name + "_visual_selected_extreme_nodes.xdmf", domain, final_dis_map)
# rescale_dis_second, _ = rescale_distance_map(domain, cluster_separate, final_dis_map)
# export_soln(save_dir + "/eikonal" + yaml_file_name + "_final_rescale_dis_map.xdmf", domain, rescale_dis_second)

point_dtf_map = solve_eikonal(domain, 2, 3, point_index,rescale_dis)
# export_soln(save_dir + "/eikonal" + yaml_file_name + "_destination_time.xdmf", domain, point_dtf_map)
export_vtk(save_dir + "/eikonal" + yaml_file_name + "_destination_time.vtu", point_dtf_map)


# ############ subcluster cl
# centerline_subcluster = combining_cl(domain, cluster_separate, extreme, rescale_dis,save_dir + "/eikonal" + yaml_file_name + "_destination_time_p0_000000.vtu",save_dir + "/eikonal" + yaml_file_name + "_dis_map_p0_000000.vtu")
# save_centerline_vtk(centerline_subcluster, save_dir + "/centerlines" + yaml_file_name + "_centerline_subcluster.vtp")
# centerline_merged_ = merge_centerline_segments(centerline_subcluster)
# _, dict_cell_ = create_dict(centerline_merged_)
# tolerance = get_subdivided_cl_spacing(dict_cell_)
# print(tolerance)
# smooth_centerline_polydata_subcluster,_,_,_ = combine_cls_into_one_polydata(dict_cell_, tolerance/2) #CHANGED THIS GALA 10172024
# save_centerline_vtk(smooth_centerline_polydata_subcluster, save_dir + "/centerlines" + yaml_file_name + "_subcluster_smooth_centerline.vtp")
#############################

centerline_polydata = combine_cl(domain, extreme_nodes, cluster_separate, save_dir + "/eikonal" + yaml_file_name + "_destination_time_p0_000000.vtu",save_dir + "/eikonal" + yaml_file_name + "_dis_map_p0_000000.vtu",geometry_type)

save_centerline_vtk(centerline_polydata, save_dir + "/centerlines" + yaml_file_name + "_centerline.vtp")
centerline_merged = merge_centerline_segments(centerline_polydata)
_, dict_cell = create_dict(centerline_merged)
tolerance = get_subdivided_cl_spacing(dict_cell)
print(tolerance)
smooth_centerline_polydata,_,_,_ = combine_cls_into_one_polydata(dict_cell, tolerance/2) #CHANGED THIS GALA 10172024
save_centerline_vtk(smooth_centerline_polydata, save_dir + "/centerlines" + yaml_file_name + "smooth_centerline.vtp")
####################################
# with open(save_dir + "/cluster" + yaml_file_name + "_extreme_clusters.txt", "w") as file:
#     file.write(f"there are {len(extreme_clusters_)} extreme nodes: {extreme_clusters_}.")
#     file.close()

with open(save_dir + "/cluster" + yaml_file_name + "_extreme_nodes.txt", "w") as file:
    file.write(f"there are {len(extreme_nodes)} extreme nodes: {extreme_nodes}.")
    file.close()

with open(save_dir + yaml_file_name + "_execution_time.txt", "w") as file:
    file.write(f"Time to run the entire code: {time.time() - time_start:0.2f}")
    file.close()





