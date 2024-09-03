import yaml
import os
from centerlines import *
import time


time_start = time.time()

yaml_file = "cylinder.yaml"

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

#nodes_center = True
# Checking if the mesh has multiple walls with separate mesh tags
if params["multiple_wall_tag"] is True:
    face_tags = list(range(params["wall_face_tag"], params["wall_face_tag_2"]))
    dis = solve_eikonal(domain, facet, True, 1, *face_tags)

else:
    dis = solve_eikonal(domain, facet, True, 1, params["wall_face_tag"])

clustering_destination_time_map = solve_eikonal(domain, facet, True, 1, 0, params["inlet_face_tag"])
cluster_graph = cluster_map(clustering_destination_time_map, domain)
cluster_separate, extreme, cluster_separate_array = separate_clusters(domain, cluster_graph)
print(f"there are {len(extreme)} extreme nodes: {extreme}")

rescale_dis, rescale_dis_array = rescale_distance_map(domain, cluster_separate, dis)

if params["just_distance"] is False:
    solution = solve_eikonal(domain, facet, False, rescale_dis,
                             params["inlet_face_tag"])

gradient_soln, gradient_array = gradient_distance(domain, solution)

centerline_polydata, centerline_all_points = centerlines(domain, cluster_separate, extreme, rescale_dis, gradient_array)



# Checking if the solution should be saved
if params["save_eikonal"] is True:
    export_soln(save_dir + "/eikonal" + yaml_file_name + "_distance_field.xdmf", domain, dis)
    export_soln(save_dir + "/eikonal" + yaml_file_name + "_rescaled_distance_field.xdmf", domain, rescale_dis)
    if params["just_distance"] is False:
        export_soln(save_dir + "/eikonal" + yaml_file_name + "_destination_time.xdmf", domain, solution)
        export_soln(save_dir + "/eikonal" + yaml_file_name + "_grad_soln.xdmf", domain, gradient_soln)

if params["save_clustermap"] is True:
    export_soln(save_dir + "/cluster" + yaml_file_name + "_cluster_map.xdmf", domain, cluster_graph)
    export_soln(save_dir + "/cluster" + yaml_file_name + "_cluster_separate.xdmf", domain, cluster_separate)

if params["save_centerline"] is True:
    if nodes_center is True:
        save_centerline_vtk(centerline_polydata, save_dir + "/cluster/centerline" + yaml_file_name + "_nodeextraction_centerline.vtp")
        #save_centerline_vtk(smooth_centerline, save_dir + "/cluster/centerline" + yaml_file_name + "_nodeextraction_smooth_centerline.vtp")
    else:
        save_centerline_vtk(centerline_polydata, save_dir + "/cluster/centerline" + yaml_file_name + "_centerline.vtp")
        #save_centerline_vtk(smooth_centerline, save_dir + "/cluster/centerline" + yaml_file_name + "_smooth_centerline.vtp")       

with open(save_dir + "/cluster" + yaml_file_name + "_extreme_nodes.txt", "w") as file:
    file.write(f"there are {len(extreme)} extreme nodes: {extreme}")
    file.close()

print(f"Time to run the entire code: {time.time() - time_start:0.2f}")

with open(save_dir + yaml_file_name + "_execution_time.txt", "w") as file:
    file.write(f"Time to run the entire code: {time.time() - time_start:0.2f}")
    file.close()
