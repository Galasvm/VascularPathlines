import numpy as np
import dolfinx
from dolfinx import fem, default_scalar_type, log
from dolfinx.fem import functionspace
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from dolfinx.io import XDMFFile
import ufl
import yaml
import os

# Call the yaml file you want to use
yaml_file = "demomesh.yaml"

# Determine parent folder
parent = os.path.dirname(__file__)

# Read yaml file located in simvascular_mesh/yaml_files
with open(parent + "/yaml_files/" + yaml_file, "rb") as f:
    params = yaml.safe_load(f)

yaml_file_name = params["file_name"]
save_dir = parent + params["saving_dir"] + yaml_file_name + yaml_file_name
mesh_dir = parent + "/Meshes" + yaml_file_name + yaml_file_name


def solve_eikonal(domain, ft, distance: bool, c=1,
                  outlet_face_tag=0, *face_tag: int):
    V = functionspace(domain, ("Lagrange", 1))

    # putting all the dofs of the faces we want to select for the boundary
    # condition
    face_tags = list(face_tag)
    count_face_tags = len(face_tags)

    # seeing if there is more than one face selected
    if count_face_tags > 1:
        if distance is False:
            print("error: Should only select one face" +
                  "when solving solving with a seedpoint")
            exit()

    all = np.array([])
    for tag in face_tags:
        face = ft.find(tag)
        all = np.append(all, face)

    # Preparing parameters based on what field is being solved
    if distance is True:

        # setting the wall of the vessel as the boundary condition
        domain.topology.create_connectivity(domain.topology.dim - 1,
                                            domain.topology.dim)
        wall_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                all)
        bc = fem.dirichletbc(default_scalar_type(0), wall_dofs, V)

        # defining the speed as 1 since we are looking for the distance field
        f = fem.Constant(domain, default_scalar_type(1))

    else:

        domain.topology.create_connectivity(domain.topology.dim - 1,
                                            domain.topology.dim)
        inlet_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                 all)
        
        # reescaling the minimum distance field

        # now we need to find the dof at the center of the facet to select as
        # the source point (the point with the greater distance from the walls)
        c_values = c.vector.array  # Get the solution values as a numpy array
        # Find the index of the maximum value in the selected values
        selected_values = c_values[inlet_dofs]
        max_selected_index = np.argmax(selected_values)

        # Find the corresponding dof (global index)
        max_dof = inlet_dofs[max_selected_index]

        # make it into an array
        inlet_dof = np.array([max_dof])

        # define the dirichlet boundary condition at one point
        bc = fem.dirichletbc(default_scalar_type(0), inlet_dof, V)

        # CALCULATING ALPHA
        # now we need to calculate alpha for the wave speed, which is related
        # to a small vessel diameter in the geometry (a face outlet
        # selected by the user)

        # first we find the dofs of the outlet selected
        outlet = ft.find(outlet_face_tag)
        outlet_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                  outlet)
        # then we find the node with the highest minimum distance from the
        # walls (the center) for the outlet. We also print out the highest
        # minimum distance from the walls for the inlet
        selected_values_out = c_values[outlet_dofs]
        max_distance_outlet = selected_values_out.max()
        max_distance_inlet = selected_values.max()
        print(f"max_distance_outlet is: {max_distance_outlet}")
        print(f"max_distance_inlet is: {max_distance_inlet}")

        # alpha is related to the smallest vessel diameter

        alpha = 1/max_distance_outlet
        print(f"alpha: {alpha}")

        # now we set the wave speed proportional to the minimum distance field
        f = 1/(2**(alpha*c))

    # CALCULATING EPS: We need to make eps related to the mesh size
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    domain = dolfinx.cpp.mesh.Mesh_float64(domain.comm, domain.topology,
                                           domain.geometry)
    h = dolfinx.cpp.mesh.h(domain, domain.topology.dim, cells)
    # finding the max cell size now
    hmax = max(h)
    print(f"max cell size: {hmax}")
    eps = hmax/2
    print(f"eps: {eps}")

    # set up the problem
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    uh = ufl.TrialFunction(V)

    # Initialize the solution to avoid convergence issues
    with u.vector.localForm() as loc:
        loc.set(1.0)

    # Initialization problem to get good initial guess
    a = ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx
    L = f*v*ufl.dx
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc],
                                      petsc_options={"ksp_type": "preonly",
                                                     "pc_type": "lu"})
    u = problem.solve()

    F = ufl.sqrt(ufl.inner(ufl.grad(u), ufl.grad(u)))*v*ufl.dx - f*v*ufl.dx
    F += eps * ufl.inner(ufl.grad(abs(u)), ufl.grad(v)) * ufl.dx
    # Jacobian of F
    J = ufl.derivative(F, u, ufl.TrialFunction(V))

    # Create nonlinear problem and solver
    problem = fem.petsc.NonlinearProblem(F, u, bcs=[bc], J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    # Set solver options
    solver.rtol = 1e-6
    solver.report = True
    # can turn this off
    log.set_log_level(log.LogLevel.INFO)

    solver.solve(u)

    if distance is True:
        # here we want to reescale the distance field
        dis_values = u.x.array
        min_val = dis_values.min()
        max_val = dis_values.max()
        new_min = 0.01
        new_max = 1.0
        rescaled_values = new_min + (dis_values - min_val) * (new_max - new_min) / (max_val - min_val)
        u = fem.Function(V)
        u.x.array[:] = rescaled_values
    return u


def import_mesh(path_mesh, path_facets):

    with XDMFFile(MPI.COMM_WORLD, path_mesh, "r") as xdmf:
        domain_ = xdmf.read_mesh(name="Grid")
    domain_.topology.create_connectivity(domain_.topology.dim,
                                         domain_.topology.dim - 1)

    with XDMFFile(MPI.COMM_WORLD, path_facets, "r") as xdmf:
        facet_ = xdmf.read_meshtags(domain_, name="Grid")
    return domain_, facet_


def export_soln(path_export, mesh_, function_):

    # Save the solution in XDMF format for visualization
    with XDMFFile(MPI.COMM_WORLD, path_export, "w") as file:
        file.write_mesh(mesh_)
        file.write_function(function_)


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
    export_soln(save_dir + "_distance_field.xdmf", domain, dis)
    if params["just_distance"] is False:
        export_soln(save_dir + "_final_soln.xdmf", domain, solution)


def cluster_map(seg, num_points_per_cluster=2000):
    values = seg.x.array
    sorted_indices = np.argsort(values)
    # Determine the number of clusters needed
    num_clusters = len(values) // num_points_per_cluster

    # Initialize the clustered values
    clustered_values = np.zeros_like(values, dtype=int)

    # Assign cluster labels
    for i in range(num_clusters):
        start_idx = i * num_points_per_cluster
        end_idx = start_idx + num_points_per_cluster

        # Ensure we don't exceed the array bounds
        if i == num_clusters - 1:
            end_idx = len(values)

        clustered_values[sorted_indices[start_idx:end_idx]] = i

    # If there are leftover points that don't fit into a full cluster, assign them to the last cluster
    if end_idx < len(values):
        clustered_values[sorted_indices[end_idx:]] = num_clusters

    V = functionspace(domain, ("Lagrange", 1))
    u = fem.Function(V)

    # Assign the clustered values back to the array
    u.x.array[:] = clustered_values
    return u


from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix



def separate_clusters(mesh, clustersmap):
    coordinates = mesh.geometry.x
    num_clusters = int(np.max(clustersmap.x.array)) + 1

    new_clusters = clustersmap.x.array.copy()
    next_cluster_id = num_clusters

    for cluster_id in range(num_clusters):
        # Get the indices of nodes in the current cluster
        cluster_indices = np.where(clustersmap.x.array == cluster_id)[0]
        cluster_coords = coordinates[cluster_indices]

        if len(cluster_indices) < 2:  # Skip clusters too small to split
            continue

        # Build a k-d tree for the cluster nodes to find close neighbors
        tree = cKDTree(cluster_coords)

        # Define a distance threshold to determine connected nodes
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        cells = np.arange(num_cells, dtype=np.int32)
        domain = dolfinx.cpp.mesh.Mesh_float64(mesh.comm, mesh.topology,
                                               mesh.geometry)
        h = dolfinx.cpp.mesh.h(domain, domain.topology.dim, cells)
        distance_threshold = max(h) * 10

        # Find pairs of nodes that are close to each other
        pairs = tree.query_pairs(r=distance_threshold)

        # Create a graph where each node is connected to its neighbors
        num_nodes = len(cluster_indices)
        graph = csr_matrix((np.ones(len(pairs)), (list(zip(*pairs))[0], list(zip(*pairs))[1])),
                           shape=(num_nodes, num_nodes))

        # Find connected components in the graph
        n_components, labels = connected_components(csgraph=graph, directed=False)

        # If there is more than one connected component, separate them into new clusters
        if n_components > 1:
            for i in range(n_components):
                component_indices = cluster_indices[labels == i]
                if i == 0:
                    # Keep the first component in the original cluster
                    new_clusters[component_indices] = cluster_id
                else:
                    # Assign a new cluster ID to the second component
                    new_clusters[component_indices] = next_cluster_id
                    next_cluster_id += 1

    # Create a new dolfinx function to hold the separated clusters
    V = functionspace(mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.x.array[:] = new_clusters
    u.x.scatter_forward()

    return u


cluster_graph = cluster_map(solution)
cluster_separate = separate_clusters(domain, cluster_graph)
export_soln(save_dir + "_cluster_map.xdmf", domain, cluster_graph)
export_soln(save_dir + "_cluster_separate.xdmf", domain, cluster_separate)
