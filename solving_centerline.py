
import yaml
import os
from centerlines import *

yaml_file = "0100_A_AO_COA_finer.yaml"

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


# Checking if the mesh has multiple walls with separate mesh tags
if params["multiple_wall_tag"] is True:
    face_tags = list(range(params["wall_face_tag"], params["wall_face_tag_2"]))
    dis = solve_eikonal(domain, facet, True, 1, 0, *face_tags)

else:
    dis = solve_eikonal(domain, facet, True, 1, 0, params["wall_face_tag"])

# Checking if only solving for the distance
if params["just_distance"] is False:
    solution = solve_eikonal(domain, facet, False, dis,
                             params["outlet_face_tag"],
                             params["inlet_face_tag"])

# Checking if the solution should be saved
if params["save_eikonal"] is True:
    export_soln(save_dir + "/eikonal" + yaml_file_name + "_distance_field.xdmf", domain, dis)
    if params["just_distance"] is False:
        export_soln(save_dir + "/eikonal" + yaml_file_name + "_final_soln.xdmf", domain, solution)

cluster_graph = cluster_map(solution, domain, params["num_clusters"])

cluster_separate, extreme, final_clusters = separate_clusters(domain, cluster_graph)
print(f"there are {len(extreme)} extreme nodes: {extreme}")


if params["save_clustermap"] is True:
    export_soln(save_dir + "/cluster" + yaml_file_name + "_cluster_map.xdmf", domain, cluster_graph)
    export_soln(save_dir + "/cluster" + yaml_file_name + "_cluster_separate.xdmf", domain, cluster_separate)



with open(save_dir + "/cluster" + yaml_file_name + "_extreme_nodes.txt", "w") as file:
    file.write(f"there are {len(extreme)} extreme nodes: {extreme}")
    file.close()



gradient_soln, gradient_array = gradient_distance(domain, solution)




export_soln(save_dir + "/eikonal" + yaml_file_name + "_grad_soln.xdmf", domain, gradient_soln)

centerline_polydata = centerlines(domain, cluster_separate, extreme, dis, gradient_array)

save_centerline_vtk(centerline_polydata, save_dir + "/cluster" + yaml_file_name + "_centerline.vtp")

with open(save_dir + "/cluster" + yaml_file_name + "_extreme_nodes.txt", "w") as file:
    file.write(f"there are {len(extreme)} extreme nodes: {extreme}")
    file.close()
