import yaml
import os
from centerlines import *
import time

time_start = time.time()

yaml_file = "0168_H_PULMFON_SVD.yaml"

# Determine parent folder
parent = os.path.dirname(__file__)

# Read yaml file located in simvascular_mesh/yaml_files
with open(parent + "/yaml_files/" + yaml_file, "rb") as f:
    params = yaml.safe_load(f)

yaml_file_name = params["file_name_fine"]
save_dir = parent + params["saving_dir"] + yaml_file_name
mesh_dir = parent + "/Fine_Meshes" + yaml_file_name + yaml_file_name

domain, facet = import_mesh(mesh_dir + ".xdmf",
                            mesh_dir + "_facet_markers.xdmf")
hmax,hmin,havg = h_max(domain)


#nodes_center = True
# Checking if the mesh has multiple walls with separate mesh tags
if params["multiple_wall_tag"] is True:
    face_tags = list(range(params["wall_face_tag"], params["wall_face_tag_2"]))
    dis = solve_eikonal(domain, facet, True, 1, *face_tags)
    export_vtk(save_dir + "/eikonal" + yaml_file_name + "_dis_map.vtu", dis)


else:
    dis = solve_eikonal(domain, facet, True, 1, params["wall_face_tag"])
    export_vtk(save_dir + "/eikonal" + yaml_file_name + "_dis_map.vtu", dis)


# if params["cluster"] is True:
#     initial_dis_threshold = hmax*0.8
#     second_dis_threshold = hmax*0.5
#     clustering_destination_time_map = solve_eikonal(domain, facet, True, 1, 0, params["inlet_face_tag"])
#     cluster_graph = discritize_dtf(clustering_destination_time_map, domain, params["num_clusters"])
#     cluster_separate, extreme = separate_clusters(domain, cluster_graph,initial_dis_threshold,30)
#     subcluster_map = smaller_clusters(domain, cluster_separate,dis,clustering_destination_time_map)
#     cluster_separate_final, extreme_final = separate_clusters(domain,subcluster_map, second_dis_threshold, 30)
#     print(f"there are {len(extreme)} extreme nodes: {extreme}")
#     #print(f"there are {len(extreme_actual)} extreme nodes: {extreme_actual}")
#     print(f"there are {len(extreme_final)} extreme nodes: {extreme_final}")
#     if params["just_distance"] is False:
#         rescale_dis, rescale_dis_array = rescale_distance_map(domain, cluster_separate_final, dis)
#         ######################CHANGE THIS JUST EXPERIMENTS#################################
#         #rescale_dis_NOSUBCLUSTER, rescale_dis_array_NOSUBCLUSTER = rescale_distance_map(domain, cluster_separate, dis)



# if params["just_distance"] is False:
#     point_dtf_map = solve_eikonal(domain, facet, False, rescale_dis,
#                              params["inlet_face_tag"])
#     gradient_point_dtf, gradient_point_dtf_array = gradient_field(domain, point_dtf_map)
#     gradient_dis, gradient_dis_array = gradient_field(domain, dis)


#     #####################CHANGE THIS JUST EXPERIMENTS#################################
#     point_dtf_map_NOSUBCLUSTER = solve_eikonal(domain, facet, False, rescale_dis,
#                              params["inlet_face_tag"])
#     gradient_point_dtf_NOSUBCLUSTER, gradient_point_dtf_array_NOSUBCLUSTER = gradient_field(domain, point_dtf_map_NOSUBCLUSTER)
    

#     if params["centerlines"] is True:
#         centerline_polydata = centerlines_with_tags(domain, cluster_separate_final, extreme_final, rescale_dis, gradient_point_dtf_array, params["centerline_node"])
#         _, dict_cell = create_dict(centerline_polydata)
#         tolerance = get_subdivided_cl_spacing(dict_cell)
#         print(tolerance)
#         smooth_centerline_polydata,_,_,_ = combine_cls_into_one_polydata(dict_cell, tolerance)
    
#         ######################CHANGE THIS JUST EXPERIMENTS#################################
#         # centerline_NOSUBCLUSTER = centerlines_with_tags(domain, cluster_separate, extreme, rescale_dis_NOSUBCLUSTER, gradient_point_dtf_array_NOSUBCLUSTER, params["centerline_node"])
#         # smooth_centerline_polydata_NOSUBCLUSTER = post_process_centerline(centerline_polydata,hmax/28)



# if params["cluster"] is True:
#     export_soln(save_dir + "/cluster" + yaml_file_name + "_1_dtf_clustermap.xdmf", domain, cluster_graph)
#     export_soln(save_dir + "/cluster" + yaml_file_name + "_2_separate_clustermap.xdmf", domain, cluster_separate)
#     export_soln(save_dir + "/cluster" + yaml_file_name + "_3_sub_clustermap.xdmf", domain, subcluster_map)
#     export_soln(save_dir + "/cluster" + yaml_file_name + "_4_final_separate_clustermap.xdmf", domain, cluster_separate_final)


# if params["save_eikonal"] is True:
export_soln(save_dir + "/eikonal" + yaml_file_name + "_distance_field.xdmf", domain, dis)
#     if params["cluster"] is True:
#         #export_soln(save_dir + "/eikonal" + yaml_file_name + "_rescaled_distance_field.xdmf", domain, rescale_dis)
#         export_soln(save_dir + "/eikonal" + yaml_file_name + "_clustering_destination_time.xdmf", domain, clustering_destination_time_map)

#     if params["just_distance"] is False:
#         export_soln(save_dir + "/eikonal" + yaml_file_name + "_destination_time.xdmf", domain, point_dtf_map)
#         export_vtk(save_dir + "/eikonal" + yaml_file_name + "_destination_time.vtu",point_dtf_map)
#         export_soln(save_dir + "/eikonal" + yaml_file_name + "_grad_soln.xdmf", domain, gradient_point_dtf) 
#         export_soln(save_dir + "/eikonal" + yaml_file_name + "_grad_rescaled_dis.xdmf", domain, gradient_dis) 

# if params["just_distance"] is False:
#     if params["centerlines"] is True:
        
#         if params["centerline_node"] is True:
#             # this one is for nodes
#             save_centerline_vtk(centerline_polydata, save_dir + "/centerlines" + yaml_file_name + "_nodes_centerline.vtp")
#             save_centerline_vtk(smooth_centerline_polydata, save_dir + "/centerlines" + yaml_file_name + "_nodes_smooth.vtp")
#             # save_centerline_vtk(centerline_NOSUBCLUSTER, save_dir + "/centerlines" + yaml_file_name + "_NOSUBCLUSTER_nodes_centerline.vtp")
#             # save_centerline_vtk(smooth_centerline_polydata_NOSUBCLUSTER, save_dir + "/centerlines" + yaml_file_name + "_NOSUBCLUSTER_nodes_smooth.vtp")

            
#         else:
#             # this one is for not nodes
#             save_centerline_vtk(centerline_polydata, save_dir + "/centerlines" + yaml_file_name + "_centerline.vtp")
#             save_centerline_vtk(smooth_centerline_polydata, save_dir + "/centerlines" + yaml_file_name + "_smooth.vtp")
#             # save_centerline_vtk(centerline_NOSUBCLUSTER, save_dir + "/centerlines" + yaml_file_name + "_NOSUBCLUSTER_centerline.vtp")
#             # save_centerline_vtk(smooth_centerline_polydata_NOSUBCLUSTER, save_dir + "/centerlines" + yaml_file_name + "_NOSUBCLUSTER_smooth.vtp")




# if params["cluster"] is True:
#     with open(save_dir + "/cluster" + yaml_file_name + "_extreme_nodes.txt", "w") as file:
#         file.write(f"there are {len(extreme)} extreme nodes: {extreme}.\nthere are {len(extreme_final)} extreme nodes: {extreme_final}")
#         file.close()

# with open(save_dir + yaml_file_name + "_geometry_summary.txt", "w") as file:

#     file.write(f"Total number of points: {len(clustering_destination_time_map.x.array)}.\nPoints per cluster(initially):{len(clustering_destination_time_map.x.array)/params["num_clusters"]}.\nMax edge size: {hmax:0.2f}\nMin edge size: {hmin:0.2f}\nAvg edge size: {havg:0.2f}")
#     file.write(f"\nDistance threshold initial separation is h_max*{initial_dis_threshold/hmax:0.2f}\nDistance threshold second separation is h_max*{second_dis_threshold/hmax:0.2f}")
#     file.write(f"\nalpha is 20")
#     file.close()
    
# print(f"Time to run the entire code: {time.time() - time_start:0.2f}")
# if params["centerline_node"] is True:
#     with open(save_dir + yaml_file_name + "_nodes_execution_time.txt", "w") as file:
#         file.write(f"Time to run the entire code: {time.time() - time_start:0.2f}")
#         file.close()
# else:
#     with open(save_dir + yaml_file_name + "_execution_time.txt", "w") as file:
#         file.write(f"Time to run the entire code: {time.time() - time_start:0.2f}")
#         file.close()




