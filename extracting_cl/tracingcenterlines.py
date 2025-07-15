'''
Contributed by Gala Sanchez Van Moer (galasvm@berkeley.edu)

'''

try:
    import vtk
except ImportError:
    raise ImportError("Could not find vtk, please install using pip or conda")

try:
    import dolfinx
except ImportError:
    raise ImportError("Could not find dolfinx, please install")


import numpy as np
from dolfinx import fem, default_scalar_type, log
from dolfinx.fem import functionspace
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.io import XDMFFile, VTKFile
import ufl
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import os
import argparse
import time
import pyvista as pv
from vtk.util import numpy_support
from model_to_mesh import *


def import_mesh(path_name):
    """
    Import a mesh and its corresponding facet tags from XDMF files.

    Args:
        path_mesh (str): Path to the XDMF file containing the mesh.
        path_facets (str): Path to the XDMF file containing the facet tags.

    Returns:
        dolfinx.cpp.mesh.Mesh: The imported mesh
    """

    # Read the mesh from the XDMF file
    with XDMFFile(MPI.COMM_WORLD, path_name, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    # Create connectivity between the mesh elements and their facets
    domain.topology.create_connectivity(domain.topology.dim, 
                                        domain.topology.dim - 1)

    return domain


def export_xdmf(path_export, mesh, function):
    """
    Export a mesh and its associated fenicsx function to an XDMF file.

    Args:
        path_export (str): Path to the XDMF file where the mesh and function
        will be saved.
        mesh: The mesh to be exported.
        function: The function associated with the mesh to be exported.

    """

    # Save the mesh and function in XDMF format for visualization
    with XDMFFile(MPI.COMM_WORLD, path_export, "w") as file:
        file.write_mesh(mesh)
        file.write_function(function)


def export_vtk(path_export, function):
    """
    Export a function to a VTK file.

    Args:
        path_export (str): Path to the VTK file where the function will be
        saved.
        function: The function to be exported.

    """

    # Create a VTK file for exporting the function
    vtkfile = VTKFile(MPI.COMM_WORLD, path_export, "w")

    # Write the function to the VTK file
    vtkfile.write_function(function)


def save_centerline_vtk(centerline_polydata, filename):
    """
    Save the centerline points and lines to a VTK file.

    Parameters:
    - centerline_polydata: vtkPolyData containing the centerline points and
    lines.
    - filename: The name of the output VTK file.
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(centerline_polydata)
    writer.Write()


def automatic_pointsource(distance_map):
    """
    Automatically determine the point source based on the distance map.

    Args:
        distance_map (fem.Function): The distance map.

    Returns:
        int: The index of the point source.
    """

    # Get the distance map values
    distance_values = distance_map.x.array

    # Find the index of the point source (the maximum distance value)
    point_source = np.argmax(distance_values)

    return point_source


def edge_max(domain):
    """
    Calculate the maximum, minimum, and average edge lengths of the cells in the mesh.
    Args:
        domain: The mesh domain (xdmf file).
    Returns:
        tuple: (max edge length, min edge length, avg edge length)
    """
    # Ensure edge-to-cell connectivity exists
    domain.topology.create_connectivity(domain.topology.dim - 2, domain.topology.dim)

    num_cells = domain.topology.index_map(domain.topology.dim).size_global
    cells = np.arange(num_cells, dtype=np.int32)
    edge = dolfinx.cpp.mesh.h(domain._cpp_object, domain.topology.dim-2, cells)
    edge_avg = np.mean(edge)
    edge_max = max(edge)
    edge_min = min(edge)
    return edge_max, edge_min, edge_avg


def solve_eikonal(domain, boundary_type, f_type, ps_index=1, distance=1):
    """
    Solve the eikonal equation on the given domain with specified boundary
    conditions and field type.

    Args:
        domain (xdmf file): The mesh domain.
        boundary_type (int): The type of boundary condition
                            (1 for walls, 2 for point source).
        f_type (int): The type of field
                        (1 for steady speed, 2 for high-speed wave).
        ps_index (int, optional): The index of the point source. Default is 1.
        distance (float, fem.Function): The distance field.
        If boundary_type is 2, this is just 1. if its for the
        destination time field then it is the distance field (fem.Function)

    Returns:
        fem.Function: The solution to the eikonal equation.
    """

    # Create a function space on the domain
    V = functionspace(domain, ("Lagrange", 1))

    # Create connectivity between the mesh elements and their facets
    domain.topology.create_connectivity(domain.topology.dim - 1,
                                        domain.topology.dim)

    # Get boundary facets and corresponding DOFs
    boundary_facets = dolfinx.cpp.mesh.exterior_facet_indices(domain.topology._cpp_object)
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                boundary_facets)
    hmax, _, _ = edge_max(domain)

    # Set boundary conditions based on the boundary type
    if boundary_type == 1:  # Setting 0 at the walls (for distance field)
        bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)

    # Setting 0 at a point source (for destination time field)
    elif boundary_type == 2:
        ps_index_array = np.array([ps_index], dtype=np.int32)

        # Define the Dirichlet boundary condition at one point
        bc = fem.dirichletbc(default_scalar_type(0), ps_index_array, V)

    else:
        print("error: no other boundary types")
        exit()

    # Prepare parameters based on the field type
    if f_type == 1:  # Steady speed
        f = fem.Constant(domain, default_scalar_type(1))
    elif f_type == 2:  # High-speed wave (proportional to distance field)
        f = 1 / (np.e ** (7 * distance))
    elif f_type == 3:
        f = distance+0.01

    # Calculate epsilon (eps) related to the mesh size
    eps = hmax / 3

    print(f"eps: {eps}")

    # Solve the problem using the set_problem function
    u = set_problem(V, bc, eps, f)

    return u


def set_problem(funcspace, DirichletBC, epsilon, f):
    """
    Set up and solve a nonlinear PDE problem using the finite element method.

    Args:
        funcspace: The function space for the problem.
        DirichletBC: Dirichlet boundary conditions.
        epsilon: Regularization parameter.
        f: Source term.

    Returns:
        u: The solution to the PDE problem.
    """

    # Create a function to hold the solution
    u = fem.Function(funcspace)
    v = ufl.TestFunction(funcspace)
    uh = ufl.TrialFunction(funcspace)

    # Set up an initial linear problem to get a good initial guess
    a = ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    problem = fem.petsc.LinearProblem(a, L, bcs=[DirichletBC],
                                      petsc_options={"ksp_type": "preonly",
                                                     "pc_type": "lu"})
    u = problem.solve()

    # Define the nonlinear form F
    F = ufl.sqrt(ufl.inner(ufl.grad(u), ufl.grad(u))) * v * ufl.dx - f * v * ufl.dx
    F += epsilon * ufl.inner(ufl.grad(abs(u)), ufl.grad(v)) * ufl.dx

    # Compute the Jacobian of F
    J = ufl.derivative(F, u, ufl.TrialFunction(funcspace))

    # Create a nonlinear problem and solver
    problem = fem.petsc.NonlinearProblem(F, u, bcs=[DirichletBC], J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    solver.rtol = 1e-8 # GALA CHANGED 02202025

    # log.set_log_level(log.LogLevel.INFO)

    # Solve the nonlinear problem
    solver.solve(u)

    return u


def gala_extreme_nodes(domain, dtf_map, distance_threshold):
    """
    Finds the local extremum nodes in the mesh based on the destination time
    field map. This code looks at all the nodes on the boundary and finds the
    nodes with the highest value out of all its neighboring nodes. Each vessel
    should have a local extremum, so this is used to identify all the outlets 
    in the geometry

    Args:
        domain(xdmf file): The mesh domain .
        dtf_map(fem.Function): The destination time field map.
        distance_threshold: The distance threshold for filtering nodes.

    Returns:
        tuple: A tuple containing the extreme DOFs and the surrounding DOFs 
        array.
    """

    # Create a function space on the domain
    V = functionspace(domain, ("Lagrange", 1))
    domain.topology.create_connectivity(domain.topology.dim - 1,
                                        domain.topology.dim)

    # Get boundary facets (2D triangles) and corresponding DOFs
    boundary_facets_indices = dolfinx.cpp.mesh.exterior_facet_indices(domain.topology._cpp_object)

    # Dictionary to store facet -> DOFs mapping
    facet_to_dofs = {}

    # this loop iterates through each facet on the boundary, finds the DOFs 
    # associated with it
    # and stores the mapping in the facet_to_dofs dictionary
    for facet in boundary_facets_indices.T:
        facet_ = np.array([facet])
        # Get DOFs associated with this facet
        facet_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                 facet_)
        facet_to_dofs[facet] = facet_dofs  # Save facet and associated DOFs

    # Now it finds all the DOFs, and finds all the facets that contain that
    # DOF and stores the mapping in dof_to_facets
    dof_to_facets = {}
    for facet, dofs in facet_to_dofs.items():
        for dof in dofs:
            if dof not in dof_to_facets:
                dof_to_facets[dof] = []
            dof_to_facets[dof].append(facet)

    # destination time field values (the higher the value the further away
    # from the inlet) and mesh coordinates
    dtf_values = dtf_map.x.array
    coordinates = domain.geometry.x  # Get mesh node coordinates

    # initilize array to store DOFs with the highest value and their
    # surrounding facets
    highest_value_dofs = []
    surrounding_dofs_array = set()  # Use set to avoid duplicates

    # Iterate through all DOFs
    for dof, facets in dof_to_facets.items():
        # Collect all the DOFs in the surrounding facets
        surrounding_dofs = set()  # Use set to avoid duplicates
        for facet in facets:
            surrounding_dofs.update(facet_to_dofs[facet])

        # Get the dtf_map values for these surrounding DOFs
        surrounding_values = {d: dtf_values[d] for d in surrounding_dofs}

        # Check if the current DOF has the highest value in the surrounding
        # facets
        if dtf_values[dof] == max(surrounding_values.values()):
            highest_value_dofs.append(dof)

    # Now filter highest_value_dofs based on pairwise distances
    final_dofs = []
    while highest_value_dofs:
        current_dof = highest_value_dofs.pop(0)
        current_coords = coordinates[current_dof]
        current_dtf = dtf_values[current_dof]

        keep_current = True

        for kept_dof in final_dofs:
            kept_coords = coordinates[kept_dof]
            distance = np.linalg.norm(current_coords - kept_coords)

            # If the nodes are close, keep the one with the higher dtf value
            if distance < distance_threshold:
                kept_dtf = dtf_values[kept_dof]
                if current_dtf <= kept_dtf:
                    keep_current = False
                    break
                else:
                    # Replace the kept node with the current node
                    final_dofs.remove(kept_dof)
                    break

        if keep_current:
            final_dofs.append(current_dof)
    coords_final_dofs = []

    # Collect coordinates of final DOFs
    for dof in final_dofs:
        coords_final_dofs.append(coordinates[dof])

    # Convert to numpy array
    coords_final_dofs = np.array(coords_final_dofs)

    surrounding_dofs_array = set()  # Use set to avoid duplicates
    for dof in final_dofs:
        facets = dof_to_facets[dof]
        for facet in facets:
            surrounding_dofs_array.update(facet_to_dofs[facet])

    surrounding_dofs_array_ = np.array(list(surrounding_dofs_array))
    return final_dofs, coords_final_dofs, surrounding_dofs_array_


def cluster_map_dtf(values, num_clusters=25):
    """
    Cluster the domain based on the destination time field values.


    Args:
        values (array): The values to be clustered.
        num_clusters (int): The number of clusters to create. Default is 25.

    Returns:
        array-like: An array of cluster labels corresponding to the input
        values.
    """

    # Sort the indices of the values in ascending order
    sorted_indices = np.argsort(values)

    # Determine the number of points per cluster
    num_points_per_cluster = len(values) // num_clusters
    print(f"number of points per clusters: {num_points_per_cluster}")

    # Initialize the array to store cluster labels
    clustered_values = np.zeros_like(values, dtype=int)

    # Assign cluster labels
    for i in range(num_clusters):
        start_idx = i * num_points_per_cluster
        end_idx = start_idx + num_points_per_cluster

        # Ensure we don't exceed the array bounds for the last cluster
        if i == num_clusters - 1:
            end_idx = len(values)

        clustered_values[sorted_indices[start_idx:end_idx]] = i

    # If there are leftover points that don't fit into a full cluster, assign
    # them to the last cluster
    if end_idx < len(values):
        clustered_values[sorted_indices[end_idx:]] = num_clusters

    return clustered_values


def discritize_dtf(dtf_map, mesh):
    """
    Discretize the destination time field (dtf) map into clusters based on the
    specified geometry type.

    Args:
        dtf_map (fem.Function): The destination time field map.
        mesh (xdmf file): The mesh domain.

    Returns:
        fem.Function: A function with the clustered destination time field 
        values.
    """

    # Extract dtf values from the map
    dtf_values = dtf_map.x.array

    nodes_per_cluster = 1500
    # Calculate the number of clusters
    num_clusters = len(dtf_values) // nodes_per_cluster
    print(num_clusters)

    # Cluster the dtf values
    clustered_values = cluster_map_dtf(dtf_values, num_clusters)

    # Create a function space on the mesh
    V = functionspace(mesh, ("Lagrange", 1))
    u = fem.Function(V)

    # Assign the clustered values back to the function array
    u.x.array[:] = clustered_values

    return u


def spatial_clustering(mesh, clustersmap):
    """
    Based on the destination time field clustered map, this code separates the
    nodes in each cluster that are on different vessels, and adds them as a
    separate cluster.


    Args:
        mesh(xdmf file): The mesh domain.
        clustersmap(fem.Function): The destination time field cluster map.

    Returns:
        array: An array of new cluster labels after spatial clustering.
    """

    # Get the coordinates of the mesh nodes
    coordinates = mesh.geometry.x

    # Determine the number of initial clusters
    num_clusters = int(np.max(clustersmap.x.array)) + 1

    # Calculate the maximum edge length in the mesh
    dis_max, _, _ = edge_max(mesh)

    # Set the distance threshold for clustering
    distance_threshold = dis_max * 0.8

    # Copy the initial clusters map to create a new clusters map
    new_clusters = clustersmap.x.array.copy()
    next_cluster_id = num_clusters
    components = {}

    for cluster_id in range(num_clusters):
        # Get the indices of nodes in the current cluster
        cluster_indices = np.where(clustersmap.x.array == cluster_id)[0]
        cluster_coords = coordinates[cluster_indices]

        # Skip clusters that are too small to split
        if len(cluster_indices) < 2:
            continue

        # Build a k-d tree for the cluster nodes to find close neighbors
        tree = cKDTree(cluster_coords)

        # Find pairs of nodes that are close to each other
        pairs = tree.query_pairs(r=distance_threshold)

        # Create a graph where each node is connected to its neighbors
        num_nodes = len(cluster_indices)
        graph = csr_matrix((np.ones(len(pairs)), (list(zip(*pairs))[0],
                                                  list(zip(*pairs))[1])),
                           shape=(num_nodes, num_nodes))

        # Find connected components in the graph
        n_components, labels = connected_components(csgraph=graph,
                                                    directed=False)
        components[cluster_id] = n_components

        # If there is more than one connected component, separate them into
        # new clusters
        if n_components > 1:
            for i in range(n_components):
                component_indices = cluster_indices[labels == i]

                if i == 0:
                    # Keep the first component in the original cluster
                    new_clusters[component_indices] = cluster_id
                else:
                    # Assign a new cluster ID to the subsequent components
                    new_clusters[component_indices] = next_cluster_id
                    next_cluster_id += 1

    return new_clusters


def merging_small_clusters(mesh, separate_clusters, cluster_threshold: int):
    """
    Merge small clusters into nearby larger clusters based on a size threshold.

    Args:
        mesh(xdmf file): The mesh domain.
        separate_clusters(array): The array of cluster labels.
        cluster_threshold (int): The minimum size of a cluster.
        Clusters smaller than this will be merged.

    Returns:
        array: An array of cluster labels after merging small clusters.
    """

    # Get the coordinates of the mesh nodes
    coordinates = mesh.geometry.x

    # Determine the number of clusters
    separated_num_clusters = int(np.max(separate_clusters)) + 1

    # Build a global k-d tree for the mesh nodes
    tree = cKDTree(coordinates)

    for cluster_id in range(separated_num_clusters):
        # Get the indices of nodes in the current cluster
        cluster_indices = np.where(separate_clusters == cluster_id)[0]

        # If the cluster is too small, merge it
        if len(cluster_indices) < cluster_threshold:
            cluster_coords = coordinates[cluster_indices]

            # Find the nearest neighbors for the nodes in the small cluster
            _, nearest_neighbor_indices = tree.query(cluster_coords, k=2)

            # Get the cluster IDs of these nearest neighbors
            nearest_cluster_ids = separate_clusters[
                nearest_neighbor_indices[:, 1]
            ]
            nearest_cluster_ids = nearest_cluster_ids.astype(int)
            nearest_cluster_ids = nearest_cluster_ids[
                nearest_cluster_ids != cluster_id
            ]

            if len(nearest_cluster_ids) > 0:
                # Find the most common nearest cluster ID to merge into
                merge_target = np.bincount(nearest_cluster_ids).argmax()

                # Merge the small cluster into the selected nearby cluster
                separate_clusters[cluster_indices] = merge_target

    return separate_clusters


def noempty_clusters(mesh, new_clusters):
    """
    Ensure that there are no empty clusters by reassigning nodes from empty
    clusters to nearby non-empty clusters.

    Args:
        mesh(xdmf file): The mesh domain.
        new_clusters(array): The array of cluster labels.

    Returns:
        array: An array of cluster labels after removing empty clusters.
    """

    # Get the coordinates of the mesh nodes
    coordinates = mesh.geometry.x

    # Build a k-d tree for the mesh nodes
    tree = cKDTree(coordinates)

    # Find unique cluster labels and their counts
    unique, counts = np.unique(new_clusters, return_counts=True)

    # Identify empty clusters
    empty_clusters = unique[counts == 0]

    for empty_cluster in empty_clusters:
        # Find nodes that were assigned to the empty cluster
        empty_nodes = np.where(new_clusters == empty_cluster)[0]

        if len(empty_nodes) > 0:
            # Reassign these nodes to a nearby non-empty cluster
            _, nearest_indices = tree.query(coordinates[empty_nodes], k=1)
            nearest_cluster_ids = new_clusters[nearest_indices]
            nearest_cluster_ids = nearest_cluster_ids[
                nearest_cluster_ids != empty_cluster
                ]

            if len(nearest_cluster_ids) > 0:
                # Find the most common nearby cluster ID to merge into
                merge_target = np.bincount(
                    nearest_cluster_ids.astype(int)).argmax()
                new_clusters[empty_nodes] = merge_target

    # Reassign cluster IDs to ensure consecutive numbering
    unique_clusters = np.unique(new_clusters)
    cluster_mapping = {old_id: new_id for new_id, old_id in
                       enumerate(unique_clusters)}

    new_clusters = np.array([cluster_mapping[old_id] for old_id in
                             new_clusters])

    return new_clusters


def separate_clusters(mesh, clustersmap, cluster_threshold=30):
    """
    Separate clusters in the mesh by performing spatial clustering,
    merging small clusters, and ensuring no empty clusters.

    Args:
        mesh(xdmf file): The mesh domain.
        clustersmap(fem.Function): The initial clusters map.
        cluster_threshold (int): The minimum size of a cluster. Clusters
        smaller than this will be merged.

    Returns:
        fem.Function: A function with the final cluster labels.
    """

    # Perform spatial clustering on the mesh
    separate_clusters = spatial_clustering(mesh, clustersmap)

    # Merge small clusters into nearby larger clusters
    new_clusters = merging_small_clusters(mesh, separate_clusters,
                                          cluster_threshold)

    # Ensure there are no empty clusters
    new_clusters_final = noempty_clusters(mesh, new_clusters)

    # Create a new dolfinx function to hold the final clusters
    V = functionspace(mesh, ("Lagrange", 1))
    u = fem.Function(V)

    # Assign the final cluster labels to the function array
    u.x.array[:] = new_clusters_final
    u.x.scatter_forward()

    return u


def rescale_distance_map(mesh, cluster_map, distance_map):
    """
    Here we are rescaling the distance map based on the "radius" of
    each cluster. This is done to ensure the center of the vessel is ~1
    throughout the geometry regardless of the diameter of the vessel. This
    helps ensure the centerlines are treaced as accurately as possible

    Args:
        mesh(xdmf file): The mesh domain.
        final_cluster_map(fem.Function): The separated cluster map.
        distance_map: The original distance map.

    Returns:
        fem.Function: A function containing the rescaled distance map function
    """

    # Determine the number of clusters
    num_clusters = int(np.max(cluster_map.x.array)) + 1

    # Extract the distance map values
    distance_map_array = distance_map.x.array

    # Initialize the array to store rescaled distance map values
    rescaled_distance_map_array = np.zeros_like(distance_map_array)

    for cluster_id in range(num_clusters):
        # Get the indices of nodes in the current cluster
        cluster_indices = np.where(cluster_map.x.array == cluster_id)[0]

        # Get the distance values for the current cluster
        distance_values = distance_map_array[cluster_indices]

        # Find the maximum distance value in the current cluster (the radius)
        max_radii = distance_values.max()

        # Avoid division by zero by setting a minimum value for max_radii
        if max_radii == 0:
            max_radii = 0.01

        # Rescale the distance values to be between 0 and 1
        rescaled_values = distance_values / max_radii
        rescaled_distance_map_array[cluster_indices] = rescaled_values

    # Create a function space on the mesh
    V = functionspace(mesh, ("Lagrange", 1))
    rescaled_distance_map = fem.Function(V)

    # Assign the rescaled distance map values to the function array
    rescaled_distance_map.x.array[:] = rescaled_distance_map_array

    return rescaled_distance_map


def compute_gradient_vtk(path_file, path_export):
    """
    Compute the gradient of a scalar field in a VTK file and save the result
    to a new VTK file.

    Args:
        path_file (str): Path to the input VTK file
        save_path (str): Path to save the output VTK file with the gradient.

    """

    # Read the input VTK file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path_file)
    reader.Update()

    # Get the mesh data from the reader
    data = reader.GetOutput()

    # Compute the gradient of the scalar field
    gradient_filter = vtk.vtkGradientFilter()
    gradient_filter.SetInputData(data)
    gradient_filter.SetInputScalars(
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 0)
    # Name for the gradient array
    gradient_filter.SetResultArrayName("Gradient")
    gradient_filter.Update()

    # Get the output mesh with the computed gradient
    gradient_data = gradient_filter.GetOutput()

    # Save the result to a new .vtu file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(path_export)  # Output file name
    writer.SetInputData(gradient_data)
    writer.Write()


def finding_longest_streamline(all_streamlines):
    """
    Find the longest streamline based on the number of points.

    Args:
        all_streamlines (vtk.vtkPolyData): The vtkPolyData object containing all streamlines.

    Returns:
        int: The index of the longest streamline, or None if no valid streamline is found.
    """

    # Find the number of streamlines (cells)
    num_cells = all_streamlines.GetNumberOfCells()

    # Initialize variables to store the index of the longest streamline and the max number of points
    max_points = -1
    longest_streamline_index = -1  # To store the index of the longest streamline

    # Iterate through all the streamlines
    for i in range(num_cells):
        selected_line = all_streamlines.GetCell(i)

        # Check if the streamline has points
        num_point_ids = selected_line.GetNumberOfPoints()

        if num_point_ids == 0:
            # Skip this streamline if it has no points
            print(f"Streamline {i} has no points, skipping.")
            continue

        # If this streamline has the most points, update the max points and store the current index
        if num_point_ids > max_points:
            max_points = num_point_ids
            longest_streamline_index = i

    # Return the index of the longest streamline
    if longest_streamline_index != -1:
        return longest_streamline_index
    else:
        print("No valid streamline found.")
        return None



def finding_best_streamline(all_streamlines, dis_map_path):
    """
    Find the best streamline traced from each outlet based on the distance map
    (choose the streamline with the highest distance).

    Args:
        all_streamlines(vtp file): The vtp file containing all streamlines.
        dis_map_path (str): Path to the VTK file containing the distance map.

    Returns:
        int: The index of the best streamline, or None if no valid streamline
        is found.
    """

    # Find the number of streamlines (cells)
    num_cells = all_streamlines.GetNumberOfCells()

    # Read the distance map from the VTK file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(dis_map_path)
    reader.Update()

    # Get the unstructured grid (mesh) from the reader
    mesh = reader.GetOutput()

    # Create a probe filter to interpolate the solution at arbitrary points
    probe = vtk.vtkProbeFilter()
    points_polydata = vtk.vtkPolyData()

    # Create a vtkPoints object for specific points
    input_points = vtk.vtkPoints()

    # Initialize variables to store the index of the best streamline and the
    # max distance
    max_distance = -float("inf")
    max_points = -1

    for i in range(num_cells):
        selected_line = all_streamlines.GetCell(i)

        # Check if the streamline has points
        num_point_ids = selected_line.GetNumberOfPoints()

        if num_point_ids == 0:
            # Skip this streamline if it has no points
            print(f"Streamline {i} has no points, skipping.")
            continue

        # If this streamline has the most points, update the max points and store the current index
        if num_point_ids > max_points:
            max_points = num_point_ids
            best_streamline_index = i


    # Iterate through all the streamlines
    for i in range(num_cells):
        selected_line = all_streamlines.GetCell(i)

        # Check if the streamline has points
        num_point_ids = selected_line.GetNumberOfPoints()
        point_id = selected_line.GetPointId(0)

        if num_point_ids == 0:
            # Skip this streamline if it has no points

            continue

        # Get the first point of the streamline
        point = all_streamlines.GetPoint(point_id)
        input_points.InsertNextPoint(point)

        # Interpolate the distance field for the current point
        points_polydata.SetPoints(input_points)
        probe.SetInputData(points_polydata)
        probe.SetSourceData(mesh)
        probe.Update()
        interpolated_data = probe.GetOutput()

        # Access the interpolated distance field using the correct name 'f'
        distance_array = interpolated_data.GetPointData().GetArray("f")

        if distance_array is None:
            print(f"Error: Could not find the 'f' array for streamline {i}.")
            continue

        # Get the distance value at this point
        distance_value = distance_array.GetValue(0)

        # If this distance is the highest, update the max distance and store
        # the current index
        if distance_value > max_distance:
            max_distance = distance_value
   
        if distance_value == max_distance and num_point_ids > (max_points*0.8):
            best_streamline_index = i  # Store the index of the current

        # Clear points for the next iteration
        input_points.Reset()

    # Return the index of the best streamline and its associated max distance
    if best_streamline_index != -1:
        return best_streamline_index
    else:
        print("No valid streamline found.")
        return None


def trace_centerline_vtk(grad_dtf_path, start_point, dis_map_path):
    """
    Trace the centerline using a gradient destination time field (DTF) map and
    a starting point.

    Args:
        grad_dtf_path (str): Path to the VTK file containing the gradient DTF
        map.
        start_point (tuple): The starting point for the stream tracer.

    Returns:
        polydata: the selected best streamline.
    """

    # Load the .vtu file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(grad_dtf_path)
    reader.Update()

    dataset = reader.GetOutput()

    # Set the name of the gradient array
    gradient_array_name = "Gradient"

    # Set up the stream tracer and use the gradient as the vector field
    streamTracer = vtk.vtkStreamTracer()
    streamTracer.SetInputData(dataset)

    # Set the active vector field to the gradient array
    streamTracer.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        gradient_array_name
    )

    # Set seed points for the stream tracer
    seed_points = vtk.vtkPointSource()
    seed_points.SetRadius(0.25)
    seed_points.SetNumberOfPoints(100)  # number of seed points
    seed_points.SetCenter(start_point)  # Center at start_point
    seed_points.Update()

    streamTracer.SetSourceConnection(seed_points.GetOutputPort())

    # Set integration parameters
    streamTracer.SetIntegratorTypeToRungeKutta4()
    streamTracer.SetInterpolatorTypeToDataSetPointLocator()
    streamTracer.SetMaximumPropagation(50000)
    streamTracer.SetInitialIntegrationStep(0.1)

    streamTracer.SetIntegrationDirectionToBackward()

    # Update and get streamlines
    streamTracer.Update()
    streamlines = streamTracer.GetOutput()

    # Get number of lines (cells) in the streamline output
    num_cells = streamlines.GetNumberOfCells()

    if num_cells == 0:
        print("No streamlines were generated.")
        return None
    elif num_cells == 1:
        return streamlines  # Return the only streamline

    # Find the best streamline based on the distance map

    # id_best_streamline = finding_longest_streamline(streamlines)
    id_best_streamline = finding_best_streamline(streamlines, dis_map_path)

    selected_cell = streamlines.GetCell(id_best_streamline)

    # Check if the selected cell has more than 15 points
    if selected_cell.GetNumberOfPoints() <= 15:
        print("The selected streamline is too short (15 or fewer points).")
        return None

    # Extract the selected line and its points
    selected_polydata = vtk.vtkPolyData()
    selected_points = vtk.vtkPoints()
    selected_lines = vtk.vtkCellArray()

    # Create a new cell and add its points to `selected_points`, skipping the first 15 points
    selected_line = vtk.vtkPolyLine()
    selected_line.GetPointIds().SetNumberOfIds(selected_cell.GetNumberOfPoints() - 15)

    for i in range(15, selected_cell.GetNumberOfPoints()):
        point_id = selected_cell.GetPointId(i)
        point = streamlines.GetPoint(point_id)
        selected_points.InsertNextPoint(point)
        selected_line.GetPointIds().SetId(i - 15, i - 15)

    # Add the line to the cell array
    selected_lines.InsertNextCell(selected_line)

    # Set the points and lines in the new polydata
    selected_polydata.SetPoints(selected_points)
    selected_polydata.SetLines(selected_lines)

    # Use vtkCleanPolyData to remove duplicate points
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(selected_polydata)
    cleaner.SetToleranceIsAbsolute(True)  # Use absolute tolerance
    cleaner.SetAbsoluteTolerance(0.02)   # Set the minimum distance
    cleaner.Update()

    # The cleaned polydata
    cleaned_polydata = cleaner.GetOutput()

    # Return both all streamlines and the selected one
    return cleaned_polydata


def merge_centerline_segments(centerline):
    """
    Merge all small lines for each tagged individual centerline into one long
    line.

    Parameters
    ----------
    centerline : vtkPolyData
        Combined centerline of the geometry with individual centerlines tagged
        by "CenterlineID".

    Returns
    -------
    merged_centerline : vtkPolyData
        The centerline with each individual tagged centerline merged into a
        single line.
    """
    # Use vtkStripper to merge connected line segments into one long line for
    # each centerline
    stripper = vtk.vtkStripper()
    stripper.SetInputData(centerline)
    stripper.JoinContiguousSegmentsOn()  # Ensure connected segments are joined
    stripper.Update()

    # Get the merged centerline
    merged_centerline = stripper.GetOutput()

    return merged_centerline


def combine_cl(mesh, extreme_nodes, cluster_map, dtf_sol_path, dis_map_path,):
    """
    Combine the centerlines traced from each extreme node (outlet of geometry)
    into a single polydata

    Args:
        mesh(xdmf file): The mesh domain.
        extreme_nodes: The extreme nodes from which to trace the centerlines.
        cluster_map(fem.Function): The cluster map of the mesh.
        dtf_sol_path(str): Path to the distance-to-feature solution file.
        dis_map_path(stf): Path to the distance map file.
        geometry_type (str): The type of geometry
        (e.g., "cere", "pulm", "coro", "aorta").

    Returns:
        vtkPolyData: The combined centerline polydata.
    """

    # Find the coordinates of the mesh nodes
    coordinates = mesh.geometry.x

    # Find the indices for the first cluster (where we shouldn't have any
    # extreme nodes)
    cluster_indices = np.where(cluster_map.x.array == 0)[0]

    # Filter out extreme nodes that are in the first cluster
    extreme_nodes_filtered = np.array(
        list(set(extreme_nodes) - set(cluster_indices))
        )

    # Initialize vtkPoints and vtkCellArray for the combined centerline
    combined_centerline_points = vtk.vtkPoints()
    combined_centerline_lines = vtk.vtkCellArray()

    # Offset to keep track of the point IDs across different centerlines
    point_offset = 0

    # Compute the gradient of the distance-to-feature solution
    compute_gradient_vtk(dtf_sol_path, dtf_sol_path + "grad_dtf.vtu")

    # coords_final_dofs = []
    for j in range(len(extreme_nodes_filtered)):
        # Get the coordinates of the current extreme node
        start_pt = coordinates[extreme_nodes_filtered[j]]

        # Trace the centerline from the current extreme node
        polydata = trace_centerline_vtk(dtf_sol_path + "grad_dtf.vtu", start_pt, dis_map_path)

        if polydata is None:
            continue

        # Append the vessel points to the combined centerline
        for i in range(polydata.GetNumberOfPoints()):
            point = polydata.GetPoint(i)
            combined_centerline_points.InsertNextPoint(point)

        # Create a line from the polydata and add it to combined centerline
        # lines
        if polydata.GetNumberOfCells() > 0:  # Check if polydata has cells
            line = vtk.vtkCellArray()
            line.DeepCopy(polydata.GetLines())
            for cell_id in range(line.GetNumberOfCells()):
                id_list = vtk.vtkIdList()
                line.GetCell(cell_id, id_list)
                for idx in range(id_list.GetNumberOfIds()):
                    id_list.SetId(idx, id_list.GetId(idx) + point_offset)
                combined_centerline_lines.InsertNextCell(id_list)

        # Update the point offset for the next centerline
        point_offset += polydata.GetNumberOfPoints()

    # Create vtkPolyData to hold the combined points and lines
    combined_centerline_polydata = vtk.vtkPolyData()
    combined_centerline_polydata.SetPoints(combined_centerline_points)
    combined_centerline_polydata.SetLines(combined_centerline_lines)

    return combined_centerline_polydata


"""
This next function is modified from Bryan's (BryannGan) code to combine
centerlines into one polydata
"""


def create_dict(pd):
    # key = cell number; value = [vtkPolyLine, [pt1, pt2, ...]]
    dict_cell = {}
    # [distance, vtkPolyLine, [pt1, pt2, ...]]
    distance_lst = [[0, 0, 0] for _ in range(pd.GetNumberOfCells())]

    for i in range(pd.GetNumberOfCells()):

        cell = pd.GetCell(i)
        num_points = cell.GetNumberOfPoints()

        # Get the coordinates of each point in the cell
        points = pd.GetPoints()
        init_pt_array = [0]*num_points

        for j in range(num_points):
            point_id = cell.GetPointId(j)
            point = points.GetPoint(point_id)
            init_pt_array[j] = point
            # calculated cumulative distance for each line

            if j > 0:
                distance_lst[i][0] += np.linalg.norm(np.array(
                    init_pt_array[j]) - np.array(init_pt_array[j-1]))

        # reverse the coordinate ordering due to Eikonal equation
        distance_lst[i][1] = cell
        distance_lst[i][2] = init_pt_array

    # sort the distance_lst based on the first index
    # re-order dict_cell content base on distance (from longest to shortest)
    distance_lst = sorted(distance_lst, key=lambda x: x[0], reverse=True)

    # dict_cell[0] has the longest distance
    for i in range(len(distance_lst)):
        # save the cell [vtp cell, pt coordinates]
        dict_cell[i] = []
        dict_cell[i].append(distance_lst[i][1])
        dict_cell[i].append(distance_lst[i][2])

    return pd, dict_cell

"""
This next functions is modified from Bryan's (BryannGan) code to combine
centerlines into one polydata
"""


def create_polydata_from_edges(edges, points):

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    edges_array = np.array(edges)

    lines = vtk.vtkCellArray()
    for edge in edges_array:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge[0])
        line.GetPointIds().SetId(1, edge[1])
        lines.InsertNextCell(line)

    polydata.SetLines(lines)

    return polydata


def create_vtkpoints_from_coordinates(coords):
    # Create a vtkPoints object and insert the coordinates
    points = vtk.vtkPoints()
    for coord in coords:
        points.InsertNextPoint(coord)
    return points


def findclosestpoint(polydata, refpoint):
    """Returns coordinate of point on polydata closest to refpoint."""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()

    id = locator.FindClosestPoint(refpoint)
    coord = polydata.GetPoint(id)
    return coord


def write_polydata(polydata, filename):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


# calculate the average of centerline spacing to determine tolerance
def get_subdivided_cl_spacing(dict_cell):
    """
    get the average spacing of the centerline
    """
    spacing = []
    for i in range(len(dict_cell)):
        pts = dict_cell[i][1]
        for j in range(len(pts)-1):
            spacing.append(np.linalg.norm(np.array(pts[j]) -
                                          np.array(pts[j+1])))
    return np.mean(spacing)


# modified Bryan's function (no radii) and changed the direction of cl


def combine_cls_into_one_polydata(dict_cell, tolerance=0.05, clean=False):
    """
    combine all centerlines into one polydata with the right
    connectivity/direction
    IMPORTANT: if subdivide factor change, tolerance should be adjusted!!
    """
    # dict_cell: {0: [polydata,pts], 1: [polydata,pts], ...}
    # first take the longest centerline and create a vtpPoint
    # then, in the next centerline, start from the endpoint/target
    # and go back to existing line and stop when it reaches a tolerance
    def create_edges(starting_num, ending_num):
        """
        create a list of edges from starting_num to ending_num
        eg, starting_num = 0, ending_num = 10
        edges = [[0,1],[1,2],...,[9,10]]
        """
        edges = []
        for i in range(starting_num, ending_num):
            edges.append([i, i+1])
        return edges

    def is_close(coord1, coord2, tol=0.1):
        """Helper function to compare coordinates with a tolerance."""
        return np.linalg.norm(np.array(coord1) - np.array(coord2)) < tol

    def find_point_coords(master_coords, coord, tol=0.01): # changed name
        """Find the index of a coordinate in master_coords, considering
        tolerance."""
        for idx, master_coord in enumerate(master_coords):
            if is_close(master_coord, coord, tol):
                return idx
        return None

    print('*** Combining centerlines ***')
    master_coords = []
    master_edges = []

    temp_pd = vtk.vtkPolyData()
    new_dict_cell_pd = [0]*len(dict_cell)
    new_dict_cell_points = [0]*len(dict_cell)
    new_dict_cell_edges = [0]*len(dict_cell)

    for i in range(len(dict_cell)):
        if i == 0:  # using the first cl as ground to grow
            length = len(master_coords)
            master_coords.extend(dict_cell[i][1])
            addition = len(master_coords) - length
            edges = create_edges(length, length+addition-1)
            master_edges.extend(edges)
            temp_pd = create_polydata_from_edges(master_edges,
                    create_vtkpoints_from_coordinates(master_coords))

            # prepare to update dict_cell
            new_dict_cell_pd[i] = temp_pd
            new_dict_cell_points[i] = dict_cell[i][1]
            new_dict_cell_edges[i] = edges

            print(f"Done with centerline {i}")

        else:
            # GALA CHANGE: mine are not backwards
            backward_cl = dict_cell[i][1][::1]
            coords_to_add = []

            count_addition = 0
            found_close_point = False  # Flag to track if any point is within tolerance

            for j in range(len(backward_cl)):
                coord = backward_cl[j]

                closest_point = findclosestpoint(temp_pd, coord)

                # when the other line gets close to the existing line, stop
                if np.linalg.norm(np.array(coord) - np.array(closest_point)) < tolerance:
                    # find the index of the closest point
                    coords_to_add.append(coord)
                    count_addition += 1
                    index_for_connecting_pt = find_point_coords(master_coords, closest_point)
                    print(f"index_for_connecting_pt: {index_for_connecting_pt}, therefore this edge is [{index_for_connecting_pt},{len(master_coords)}]")
                    if index_for_connecting_pt is None:
                        print("warning: closest point not found. something must be wrong")
                    found_close_point = True  # Set the flag to True
                    break
                else:
                    coords_to_add.append(coord)
                    count_addition += 1

            # If no close point was found, skip adding this line
            if clean and not found_close_point:
                # if not found_close_point:
                print(f"No close point found for centerline {i}, skipping this line.")
                continue

            # flip coords_to_add so that it follows flow direction
            coords_to_add = coords_to_add[::-1]

            # create edges: first coords is the bifurcation
            edges = create_edges(len(master_coords), len(master_coords)+count_addition-1)

            # add to master_coords
            master_coords.extend(coords_to_add)
            master_edges.extend(edges)
            temp_pd = create_polydata_from_edges(master_edges, create_vtkpoints_from_coordinates(master_coords))

            print(f"Done with centerline {i}")

    points = vtk.vtkPoints()
    for coord in master_coords:
        points.InsertNextPoint(coord)

    pd_pre = create_polydata_from_edges(master_edges, points)
    pd = merge_centerline_segments(pd_pre) # GALA ADDED THIS TO MAKE EACH CELL A SINGLE CENTERLINE

    return pd, master_coords, master_edges


def remove_lines_from_vtp(input_vtp_path, output_vtp_path, cells_to_remove):
    """
    Remove specified lines (cells) from a VTP file and save the result to a new VTP file.

    Args:
        input_vtp_path (str): Path to the input VTP file.
        output_vtp_path (str): Path to save the output VTP file.
        cells_to_remove (list of int): List of cell IDs to remove from the VTP file.

    """

    # Read the input VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Use vtkIdFilter to add original cell IDs to the polydata
    id_filter = vtk.vtkIdFilter()
    id_filter.SetInputData(polydata)
    id_filter.SetCellIdsArrayName("vtkOriginalCellIds")
    id_filter.Update()

    # Convert cells_to_remove to a set for quick lookup
    cells_to_remove_set = set(cells_to_remove)

    # Create a new vtkPolyData to store the cells to keep
    cells_to_keep = vtk.vtkPolyData()
    cells_to_keep.DeepCopy(polydata)

    # Iterate over cells and keep those not in cells_to_remove
    cell_ids = vtk.vtkIdTypeArray()
    cell_ids.SetName("vtkOriginalCellIds")

    for i in range(cells_to_keep.GetNumberOfCells()):
        if i not in cells_to_remove_set:
            cell_ids.InsertNextValue(i)

    # Extract only the cells with IDs in cell_ids
    selection_node = vtk.vtkSelectionNode()
    selection_node.SetFieldType(vtk.vtkSelectionNode.CELL)
    selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
    selection_node.SetSelectionList(cell_ids)

    selection = vtk.vtkSelection()
    selection.AddNode(selection_node)

    extract_selection = vtk.vtkExtractSelection()
    extract_selection.SetInputData(0, cells_to_keep)
    extract_selection.SetInputData(1, selection)
    extract_selection.Update()

    # Convert the output to vtkPolyData
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(extract_selection.GetOutput())
    geometry_filter.Update()

    # Write the result to a new VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtp_path)
    writer.SetInputData(geometry_filter.GetOutput())
    writer.Write()


def cl_distance_field(cl, eps, dis_field_map, output_vtp_file):
    """
    Add the distance field values to the centerline points and write the updated file

    Args:
        cl (vtp): centerline vtp file
        dis_field_map (path): distance field path to vtu file
        output_vtp_file (vtp): output centerline vtp file with distance field values added

    """
    # Use vtkProbeFilter to interpolate distance values at centerline points
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(dis_field_map)
    reader.Update()
    dis_field_map = reader.GetOutput()

    # Use vtkProbeFilter to interpolate distance values at centerline points
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInputData(cl)
    probe_filter.SetSourceData(dis_field_map)
    probe_filter.Update()

    # Get the distance array
    distance_array = probe_filter.GetOutput().GetPointData().GetArray("f")

    # Convert the distance array to a NumPy array for manipulation
    distance_values = numpy_support.vtk_to_numpy(distance_array)

    # Apply the transformation to each value
    transformed_values = 1.224 * distance_values + 1.037 * eps + 0.012

    # Convert the transformed values back to a VTK array
    transformed_array = numpy_support.numpy_to_vtk(transformed_values)
    transformed_array.SetName("MaximumInscribedSphereRadius")

    # Add the transformed array to the centerline's point data
    cl.GetPointData().AddArray(transformed_array)

    # Write the updated centerline to a VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtp_file)
    writer.SetInputData(cl)
    writer.Write()

    return transformed_array


def user_pick_pointsource(mesh_path):
    """
    Opens a VTK window displaying the mesh surface.
    - Left-click to pick a point (previous marker, if any, is removed automatically).
    - Press 'q' to finish and close.
    Returns the (x,y,z) of the picked point.
    """
    # Read XDMF
    reader = vtk.vtkXdmf3Reader()
    reader.SetFileName(mesh_path)
    reader.Update()
    ug = reader.GetOutput()

    # Extract surface
    surf_filter = vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetInputData(ug)
    surf_filter.Update()
    surface = surf_filter.GetOutput()

    # Setup rendering
    renderer = vtk.vtkRenderer()
    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renwin)

    # Surface actor (fully opaque to prevent picking through)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1.0)  # opaque surface
    actor.GetProperty().EdgeVisibilityOn()
    renderer.AddActor(actor)
    renderer.SetBackground(1,1,1)

    # Precompute sphere radius
    b = surface.GetBounds()
    diag = np.linalg.norm([b[1]-b[0], b[3]-b[2], b[5]-b[4]])
    sphere_radius = diag * 0.01

    picked = None     # store single pick
    marker = None     # current sphere actor

    class PickerStyle(vtk.vtkInteractorStyleTrackballCamera):
        def __init__(self):
            super().__init__()
            self.AddObserver('RightButtonPressEvent', self.on_right_click)
            self.AddObserver('KeyPressEvent', self.on_key_press)

        def on_right_click(self, obj, event):
            nonlocal picked, marker
            # Remove previous marker if exists
            if marker is not None:
                renderer.RemoveActor(marker)
                marker = None
            # Get click position
            x,y = iren.GetEventPosition()
            picker = vtk.vtkPointPicker()
            picker.Pick(x, y, 0, renderer)
            pt = picker.GetPickPosition()
            picked = pt
            # Create new sphere marker
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(pt)
            sphere.SetRadius(sphere_radius)
            sphere.SetThetaResolution(16)
            sphere.SetPhiResolution(16)
            sphere.Update()
            mmapper = vtk.vtkPolyDataMapper()
            mmapper.SetInputData(sphere.GetOutput())
            mactor = vtk.vtkActor()
            mactor.SetMapper(mmapper)
            mactor.GetProperty().SetColor(1,0,0)
            renderer.AddActor(mactor)
            marker = mactor
            renwin.Render()

        def on_key_press(self, obj, event):
            key = self.GetInteractor().GetKeySym().lower()
            if key == 'q':
                iren.TerminateApp()

    # Apply custom style
    style = PickerStyle()
    iren.SetInteractorStyle(style)

    # Start interaction
    renwin.Render()
    iren.Initialize()
    renwin.SetWindowName("Left-click: pick; 'q': finish")
    iren.Start()

    if picked is None:
        print("No point was picked.")
    return picked


def find_point_index_in_vtu(vtu_path, point_xyz):
    """
    Given a VTU file and a 3D coordinate, find the index of the closest mesh point.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    ug = reader.GetOutput()

    vtk_pts = ug.GetPoints().GetData()
    np_pts = numpy_support.vtk_to_numpy(vtk_pts)

    tree = cKDTree(np_pts)
    _, idx = tree.query(point_xyz)
    return int(idx)


def user_pick_lines(vtp_path, vessel_xdmf_path=None):
    """
    Display a VTP of centerlines, let the user right-click to
    toggle-select cells (lines). Press 'q' to finish and close.
    Optionally overlay vessel geometry from XDMF for context.
    Returns a list of selected cell IDs.
    """
    # Read the centerline polydata
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    pd = reader.GetOutput()

    # Rendering setup
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renwin)

    # 1) Optional vessel geometry overlay
    if vessel_xdmf_path:
        geo_reader = vtk.vtkXdmf3Reader()
        geo_reader.SetFileName(vessel_xdmf_path)
        geo_reader.Update()
        ug_geo = geo_reader.GetOutput()
        surf_filter = vtk.vtkDataSetSurfaceFilter()
        surf_filter.SetInputData(ug_geo)
        surf_filter.Update()
        vessel_surface = surf_filter.GetOutput()

        vmapper = vtk.vtkPolyDataMapper()
        vmapper.SetInputData(vessel_surface)
        vactor = vtk.vtkActor()
        vactor.SetMapper(vmapper)
        vactor.GetProperty().SetOpacity(0.2)         # translucent
        vactor.GetProperty().SetColor(1, 1, 1)  # white
        vactor.SetPickable(False)                    # ignore picks
        renderer.AddActor(vactor)

    # 2) Centerlines actor (opaque, thick)
    cl_mapper = vtk.vtkPolyDataMapper()
    cl_mapper.SetInputData(pd)
    cl_actor = vtk.vtkActor()
    cl_actor.SetMapper(cl_mapper)
    cl_actor.GetProperty().SetColor(0.2, 0.2, 0.8)
    cl_actor.GetProperty().SetLineWidth(6)  # thicker base lines
    renderer.AddActor(cl_actor)

    # storage for selections
    selected_ids = []
    selected_actors = {}

    # Custom interactor style
    class CellStyle(vtk.vtkInteractorStyleTrackballCamera):
        def __init__(self):
            super().__init__()
            self.AddObserver('RightButtonPressEvent', self.on_right)
            self.AddObserver('KeyPressEvent', self.on_key)

        def on_right(self, obj, event):
            x, y = iren.GetEventPosition()
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            picker.Pick(x, y, 0, renderer)
            cid = picker.GetCellId()
            if cid < 0:
                return

            # toggle selection
            if cid in selected_actors:
                # remove existing highlight
                renderer.RemoveActor(selected_actors[cid])
                selected_ids.remove(cid)
                del selected_actors[cid]
            else:
                # extract that single cell
                ids = vtk.vtkIdList()
                ids.InsertNextId(cid)
                extract = vtk.vtkExtractCells()
                extract.SetInputData(pd)
                extract.SetCellList(ids)
                extract.Update()
                ug_sel = extract.GetOutput()

                # convert to PolyData so mapper can handle it
                geom = vtk.vtkGeometryFilter()
                geom.SetInputData(ug_sel)
                geom.Update()
                sel_poly = geom.GetOutput()

                # highlight it in red and thicker
                sel_mapper = vtk.vtkPolyDataMapper()
                sel_mapper.SetInputData(sel_poly)
                sel_actor = vtk.vtkActor()
                sel_actor.SetMapper(sel_mapper)
                sel_actor.GetProperty().SetColor(1, 0, 0)
                sel_actor.GetProperty().SetLineWidth(8)  # even thicker
                renderer.AddActor(sel_actor)

                selected_actors[cid] = sel_actor
                selected_ids.append(cid)

            renwin.Render()

        def on_key(self, obj, event):
            key = self.GetInteractor().GetKeySym().lower()
            if key == 'q':
                iren.TerminateApp()

    # Launch
    style = CellStyle()
    iren.SetInteractorStyle(style)
    renwin.Render()
    renwin.SetWindowName("Right-click: toggle select lines; 'q': finish")
    iren.Initialize()
    iren.Start()

    return selected_ids

def main_(model_path, save_dir, pointsource=None, remove_extra_centerlines=False):

    if model_path.endswith(".xdmf"):
        domain = import_mesh(model_path)
        mesh_time = None

    elif model_path.endswith(".vtp") or model_path.endswith(".stl"):
        print("Converting surface mesh to XDMF format...")
        start_time = time.time()
        main(model_path, save_dir)
        end_time = time.time()
        mesh_time = end_time - start_time
        domain = import_mesh(save_dir+".xdmf")

    start_timer = time.time()
    edgemax, _, edgeavg = edge_max(domain)

    # Solve for the distance field
    dis = solve_eikonal(domain, 1, 1, 1)
    export_vtk(save_dir + "/eikonal/dis_map.vtu", dis) # this must be saved for the next step

    with open(save_dir + "/eikonal/eps.txt", "w") as file:
        file.write(f"eps: {edgemax/3}")
        file.close()

    # Handle point source input. If no point source was selected, 
    # ask the user to input one. If the user still doesn't want to
    # provide one, then it will be automatically selected.

    first_stop_timer = time.time()
    if pointsource is None:
        print("No point source provided. Please select a point source on the mesh (window should open). Left click to select a point and press 'q' to finish.")
        if model_path.endswith(".xdmf"):
            point_coord = user_pick_pointsource(model_path)
        else:
            point_coord = user_pick_pointsource(save_dir + ".xdmf")
        print(f"User-selected point source: node {point_coord}")
        if point_coord is None:
            print("No point source selected. Automatically selecting point source.")
            point_index = automatic_pointsource(dis)
        else:
            point_index = find_point_index_in_vtu(save_dir + "/eikonal/dis_map_p0_000000.vtu", point_coord)
    elif pointsource == "auto":
        point_index = automatic_pointsource(dis)
        print("Automatically selecting point source.")
    else:
        point_index = pointsource
    
    # Now propagate a moderate speed wave. This is the first step to identify
    # the extreme nodes (or outlets of the geometry)
    second_start_timer = time.time()
    dtf_mod_speed = solve_eikonal(domain, 2, 3, point_index, dis)
    # dtf_mod_speed does not need to be saved
    # export_xdmf(save_dir + "/eikonal/dft_mod_speed.xdmf", domain, dtf_mod_speed)

    # Perform clustering based on the destination time field
    cluster_graph = discritize_dtf(dtf_mod_speed, domain)
    cluster_separate = separate_clusters(domain, cluster_graph, 30)
    # cluster map doesn't need to be saved
    # export_xdmf(save_dir + "/cluster/cluster_map.xdmf", domain, cluster_separate) 

    # Rescale the distance map based on the cluster radii. This will ensure that
    # the center of the vessel is ~1 throughout the geometry regardless of the
    # diameter of the vessel. This helps ensure the centerlines are traced as
    # accurately as possible.
    rescale_dis = rescale_distance_map(domain, cluster_separate, dis)
    # rescaled distance field doesn't need to be saved
    # export_xdmf(save_dir + "/eikonal/rescale_dis_map.xdmf", domain, rescale_dis) 

    # Find extreme nodes by looking through the nodes of the surface of the mesh
    # and finding all the nodes that have highest destination time values
    # out of their immediate neighbors
    extreme_nodes, extreme_node_coord, _ = gala_extreme_nodes(domain, dtf_mod_speed, edgeavg * 4)
    point_cloud = pv.PolyData(extreme_node_coord)
    # point_cloud does not need to be saved
    point_cloud.save(save_dir+"/extreme_points.vtp")

    # Solve the eikonal equation for the destination time field
    point_dtf_map = solve_eikonal(domain, 2, 2, point_index, rescale_dis)
    
    export_vtk(save_dir + "/eikonal/destination_time.vtu", point_dtf_map) # destination time field NEEDS to be saved

    # Combine all individual centerlines from the extreme nodes (outlets) into a 
    # single polydata
    centerline_polydata = combine_cl(domain, extreme_nodes, cluster_separate, save_dir + "/eikonal/destination_time_p0_000000.vtu",save_dir + "/eikonal/dis_map_p0_000000.vtu")
    # np.save(save_dir+".npy", coord_extreme_nodes)
    # Don't need to save the un-merged, un-smooth centerline
    save_centerline_vtk(centerline_polydata, save_dir + "/centerlines/centerline.vtp")

    # These are the steps for merging individual centerlines that overlap
    _, dict_cell = create_dict(centerline_polydata)
    tolerance = get_subdivided_cl_spacing(dict_cell)
    print("#############Tolerance: ##################", tolerance)

    print("edge max: ", edgemax)

    if tolerance < (edgemax/10):
        tolerance = edgemax/10
        print("############# New Tolerance: ##################", tolerance)

    # smooth_centerline_polydata, _, _ = combine_cls_into_one_polydata(dict_cell, tolerance / 2)
    smooth_clean_centerline_polydata, _, _ = combine_cls_into_one_polydata(dict_cell, tolerance, True) # Gala changed tolerance!!!!

    # This is the final centerline (before deletion of extra centerlines according to user)
    # save_centerline_vtk(smooth_clean_centerline_polydata, save_dir + "/centerlines/smooth_clean_centerline.vtp")
    cl_distance_field(smooth_clean_centerline_polydata, edgemax/3, save_dir + "/eikonal/dis_map_p0_000000.vtu", save_dir + "/centerlines/smooth_centerline.vtp")
    end_timer = time.time()
    run_time = (end_timer - second_start_timer) + (first_stop_timer-start_timer)
    # Save extreme nodes information. This is not necessary
    with open(save_dir + "/extreme_nodes.txt", "w") as file:
        file.write(f"There are {len(extreme_nodes)} extreme nodes: {extreme_nodes}.")
        file.close()

    with open(save_dir + "/time.txt", "w") as file:
        if mesh_time is not None:
            file.write(f"Meshing time: {mesh_time} seconds\n")
        file.write(f"Total run time: {run_time} seconds")
        file.close()

    if remove_extra_centerlines:
        while True:
            print("Please visualize the centerline. If there are any extra centerlines you would like to remove, left-click on them to select them. Press 'q' to finish.")
            if model_path.endswith(".xdmf"):
                ids = user_pick_lines(save_dir + "/centerlines/smooth_centerline.vtp", model_path)
            else:
                ids = user_pick_lines(save_dir + "/centerlines/smooth_centerline.vtp", save_dir + ".xdmf")
            print(f"Selected lines to remove: {ids}")
            if not ids:
                print("No lines to remove. Ending the program.")
                break
            else:
                output_vtp_path = save_dir + "/centerlines/smooth_centerline.vtp"
                remove_lines_from_vtp(save_dir + "/centerlines/smooth_centerline.vtp", output_vtp_path, ids)
                break


import glob

def process_all_models_in_directory(directory, base_save_dir, pointsource="auto", remove_extra_centerlines=False):
    # Find all model files in the specified directory
    vtp_files = glob.glob(os.path.join(directory, "*.vtp"))
    xdmf_files = glob.glob(os.path.join(directory, "*.xdmf"))
    stl_files = glob.glob(os.path.join(directory, "*.stl"))

    model_files = vtp_files + xdmf_files + stl_files

    for model_file in model_files:
        # Extract the model name without the extension
        model_name = os.path.splitext(os.path.basename(model_file))[0]

        # Create a subdirectory for each model
        model_save_dir = base_save_dir+"/"+model_name+"/"+model_name
        os.makedirs(model_save_dir, exist_ok=True)

        print(f"Processing model: {model_file}")
        main_(model_file, model_save_dir, pointsource, remove_extra_centerlines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process centerline tracing.")
    parser.add_argument("directory", type=str, help="Directory containing the model files.")
    parser.add_argument("save_dir", type=str, help="Directory to save the results.")
    parser.add_argument("--pointsource", type=str, help="Index of the point source.", default="auto")
    parser.add_argument("--remove_extra_centerlines", action="store_true", help="Flag to remove extra centerlines.")

    args = parser.parse_args()
    process_all_models_in_directory(args.directory, args.save_dir, args.pointsource, args.remove_extra_centerlines)
