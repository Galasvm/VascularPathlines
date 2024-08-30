import numpy as np
import dolfinx
from dolfinx import fem, default_scalar_type, log
from dolfinx.fem import functionspace
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from dolfinx.io import XDMFFile
import ufl
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import vtk


def solve_eikonal(domain, ft, distance: bool, c=1,
                  outlet_face_tag=0, *face_tag: int):
    V = functionspace(domain, ("Lagrange", 1))
    domain.topology.create_connectivity(domain.topology.dim - 1,
                                        domain.topology.dim)
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

        wall_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                all)
        bc = fem.dirichletbc(default_scalar_type(0), wall_dofs, V)

        # defining the speed as 1 since we are looking for the distance field
        f = fem.Constant(domain, default_scalar_type(1))

    else:

        inlet_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                 all)

        # now we need to find the dof at the center of the facet to select as
        # the source point (the point with the greater distance from the walls)

        inlet_dof = find_ps(c, inlet_dofs)
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
        c_values = c.vector.array
        selected_values_out = c_values[outlet_dofs]
        selected_values_in = c_values[inlet_dofs]
        max_distance_outlet = selected_values_out.max()
        max_distance_inlet = selected_values_in.max()
        print(f"max_distance_outlet is: {max_distance_outlet}")
        print(f"max_distance_inlet is: {max_distance_inlet}")

        # alpha is related to the smallest vessel diameter

        alpha = 1/max_distance_outlet
        print(f"alpha: {alpha}")

        # now we set the wave speed proportional to the minimum distance field
        f = 1/(2**(alpha*c))

    # CALCULATING EPS: We need to make eps related to the mesh size
    # finding the max cell size now
    hmax, hmin = h_max(domain)
    print(f"max cell size: {hmax}")
    eps = hmax/2
    print(f"eps: {eps}")

    u = set_problem(V, bc, eps, f)

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


def find_ps(distance_field, tag_dofs):
    # thid function finds point source for selected face tag
    values = distance_field.vector.array
    inlet_values = values[tag_dofs]
    max_inlet_index = np.argmax(inlet_values)
    inlet_dof = np.array([tag_dofs[max_inlet_index]])

    return inlet_dof


def h_max(domain):
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    domain = dolfinx.cpp.mesh.Mesh_float64(domain.comm, domain.topology,
                                           domain.geometry)
    h = dolfinx.cpp.mesh.h(domain, domain.topology.dim, cells)
    # finding the max cell size now
    h_max = max(h)
    h_min = min(h)
    return h_max, h_min


def set_problem(funcspace, boundary, epsilon, f):
    u = fem.Function(funcspace)
    v = ufl.TestFunction(funcspace)
    uh = ufl.TrialFunction(funcspace)

    # Initialize the solution to avoid convergence issues
    with u.vector.localForm() as loc:
        loc.set(1.0)

    # Initialization problem to get good initial guess
    a = ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx
    L = f*v*ufl.dx
    problem = fem.petsc.LinearProblem(a, L, bcs=[boundary],
                                      petsc_options={"ksp_type": "preonly",
                                                     "pc_type": "lu"})
    u = problem.solve()

    F = ufl.sqrt(ufl.inner(ufl.grad(u), ufl.grad(u)))*v*ufl.dx - f*v*ufl.dx
    F += epsilon * ufl.inner(ufl.grad(abs(u)), ufl.grad(v)) * ufl.dx
    # Jacobian of F
    J = ufl.derivative(F, u, ufl.TrialFunction(funcspace))

    # Create nonlinear problem and solver
    problem = fem.petsc.NonlinearProblem(F, u, bcs=[boundary], J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    # Set solver options
    solver.rtol = 1e-6
    solver.report = True
    # can turn this off
    #log.set_log_level(log.LogLevel.INFO)

    solver.solve(u)

    return u


def gradient_distance(mesh, time_distance_map):

    # Compute the symbolic gradient of the solution u
    grad_u = ufl.grad(time_distance_map)

    # Define a vector function space for the gradient (e.g., Lagrange element of degree 1)
    W = functionspace(mesh, ("Lagrange", 1, (3, )))

    # Define the trial and test functions for the projection problem
    w = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)

    # Set up the variational problem to project the gradient
    a = ufl.inner(w, v) * ufl.dx
    L = ufl.inner(grad_u, v) * ufl.dx

    # Solve the projection problem
    w = fem.Function(W)
    problem = fem.petsc.LinearProblem(a, L, u=w,
                                      petsc_options={"ksp_type": "preonly",
                                                     "pc_type": "lu"})
    problem.solve()

    x_comp = w.x.array.reshape(-1, 3)[:, 0]  # x component
    y_comp = w.x.array.reshape(-1, 3)[:, 1]  # y component
    z_comp = w.x.array.reshape(-1, 3)[:, 2]

    # Combine into a single NumPy array
    gradient_array = np.vstack((x_comp, y_comp, z_comp)).T

    return w, gradient_array


def import_mesh(path_mesh, path_facets):

    with XDMFFile(MPI.COMM_WORLD, path_mesh, "r") as xdmf:
        domain_ = xdmf.read_mesh(name="Grid")
    domain_.topology.create_connectivity(domain_.topology.dim,
                                         domain_.topology.dim - 1)

    with XDMFFile(MPI.COMM_WORLD, path_facets, "r") as xdmf:
        facet_ = xdmf.read_meshtags(domain_, name="Grid")
    return domain_, facet_


def export_soln(path_export, mesh, function):

    # Save the solution in XDMF format for visualization
    with XDMFFile(MPI.COMM_WORLD, path_export, "w") as file:
        file.write_mesh(mesh)
        file.write_function(function)


def cluster_map(seg, mesh, num_clusters=25):
    values = seg.x.array
    print(f"this is how many nodes there are: {len(values)}")
    sorted_indices = np.argsort(values)
    # Determine the number of clusters needed
    num_points_per_cluster = len(values) // num_clusters
    print(f"number of points per clusters: {num_points_per_cluster}")

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

    V = functionspace(mesh, ("Lagrange", 1))
    u = fem.Function(V)

    # Assign the clustered values back to the array
    u.x.array[:] = clustered_values
    return u


def separate_clusters(mesh, clustersmap):
    coordinates = mesh.geometry.x
    num_clusters = int(np.max(clustersmap.x.array)) + 1

    dis_max, dis_min = h_max(mesh)
    distance_threshold = dis_max * 0.8
    print(f"distance threshold: {distance_threshold}")
    print(f"min distance: {dis_min} and max distance: {dis_max}")

    new_clusters = clustersmap.x.array.copy()
    next_cluster_id = num_clusters
    components = {}

    # Step 1: Separate clusters
    for cluster_id in range(num_clusters):
        # Get the indices of nodes in the current cluster
        cluster_indices = np.where(clustersmap.x.array == cluster_id)[0]
        cluster_coords = coordinates[cluster_indices]

        if len(cluster_indices) < 2:  # Skip clusters too small to split
            continue

        # Build a k-d tree for the cluster nodes to find close neighbors
        tree = cKDTree(cluster_coords)

        # Find pairs of nodes that are close to each other
        pairs = tree.query_pairs(r=distance_threshold)

        # Create a graph where each node is connected to its neighbors
        num_nodes = len(cluster_indices)
        graph = csr_matrix((np.ones(len(pairs)), (list(zip(*pairs))[0], list(zip(*pairs))[1])),
                           shape=(num_nodes, num_nodes))

        # Find connected components in the graph
        n_components, labels = connected_components(csgraph=graph, directed=False)
        components[cluster_id] = n_components

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
        
    print(f"connected components list: {components}")
    # Step 2: Handle small clusters by merging into nearby larger clusters
    separated_num_clusters = int(np.max(new_clusters)) + 1
    tree = cKDTree(coordinates)  # Global k-d tree

    for cluster_id in range(separated_num_clusters):
        # Get the indices of nodes in the current cluster
        cluster_indices = np.where(new_clusters == cluster_id)[0]

        if len(cluster_indices) < 50:  # If the cluster is too small, merge it
            cluster_coords = coordinates[cluster_indices]
            _, nearest_neighbor_indices = tree.query(cluster_coords, k=2)

            # Get the cluster IDs of these nearest neighbors
            nearest_cluster_ids = new_clusters[nearest_neighbor_indices[:, 1]]
            nearest_cluster_ids = nearest_cluster_ids.astype(int)
            nearest_cluster_ids = nearest_cluster_ids[nearest_cluster_ids != cluster_id]

            if len(nearest_cluster_ids) > 0:
                # Find the most common nearest cluster ID to merge into
                merge_target = np.bincount(nearest_cluster_ids).argmax()
                # Merge the small cluster into the selected nearby cluster
                new_clusters[cluster_indices] = merge_target

    # Step 3: Identify and remove empty clusters
    unique, counts = np.unique(new_clusters, return_counts=True)
    empty_clusters = unique[counts == 0]

    for empty_cluster in empty_clusters:
        # Find nodes that were assigned to the empty cluster
        empty_nodes = np.where(new_clusters == empty_cluster)[0]

        if len(empty_nodes) > 0:
            # Reassign these nodes to a nearby non-empty cluster
            _, nearest_indices = tree.query(coordinates[empty_nodes], k=1)
            nearest_cluster_ids = new_clusters[nearest_indices]
            nearest_cluster_ids = nearest_cluster_ids[nearest_cluster_ids != empty_cluster]

            if len(nearest_cluster_ids) > 0:
                merge_target = np.bincount(nearest_cluster_ids.astype(int)).argmax()
                new_clusters[empty_nodes] = merge_target

    # Step 4: Reassign cluster IDs to ensure consecutive numbering
    unique_clusters = np.unique(new_clusters)
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}

    new_clusters = np.array([cluster_mapping[old_id] for old_id in new_clusters])

    # Step 5: Identify end clusters based on the updated cluster information
    separated_num_clusters = int(np.max(new_clusters)) + 1
    cluster_neighbors = {cluster_id: set() for cluster_id in range(separated_num_clusters)}

    for cluster_id in range(separated_num_clusters):
        cluster_indices = np.where(new_clusters == cluster_id)[0]
        cluster_coords = coordinates[cluster_indices]

        if len(cluster_indices) < 2:  # Skip clusters too small to analyze
            continue

        for idx in cluster_indices:
            neighbors = tree.query_ball_point(coordinates[idx], distance_threshold)
            for neighbor_idx in neighbors:
                neighbor_cluster_id = new_clusters[neighbor_idx]
                if neighbor_cluster_id != cluster_id:
                    cluster_neighbors[cluster_id].add(neighbor_cluster_id)

    end_clusters = [cluster_id for cluster_id, neighbors in cluster_neighbors.items() if len(neighbors) < 2]

    # Create a new dolfinx function to hold the final clusters
    V = functionspace(mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.x.array[:] = new_clusters
    u.x.scatter_forward()

    return u, end_clusters, new_clusters


def find_nearest_node(point, coordinates):
    """
    Find the index of the nearest node to a given point.

    Parameters:
    - point: The point to search for (as a numpy array).
    - coordinates: The coordinates of all nodes (as a numpy array).

    Returns:
    - The index of the nearest node.
    """
    distances = np.linalg.norm(coordinates - point, axis=1)
    return np.argmin(distances)


def trace_centerline_vtk(mesh, gradient_array, start_point, end_point, step_size=0.05, tolerance=0.1):
    """
    Trace a centerline from start_point to end_point by following the gradient of gradient_array,
    and save the points in VTK format.

    Parameters:
    - grad_u_array: Array of gradient vectors at each node (shape should be (num_nodes, dim)).
    - start_point: The starting point of the centerline (as a numpy array).
    - end_point: The ending point of the centerline (as a numpy array).
    - coordinates: The coordinates of all nodes (as a numpy array).
    - step_size: The size of the step taken along the gradient.
    - tolerance: The tolerance for stopping the trace when close to the end_point.

    Returns:
    - centerline_polydata: vtkPolyData containing the centerline points and lines.
    """
    # Initialize the current point to the starting point
    coordinates = mesh.geometry.x
    current_point = np.array(start_point)

    # Initialize vtk points and lines for the centerline
    centerline_points = vtk.vtkPoints()
    centerline_lines = vtk.vtkCellArray()

    # Insert the starting point
    previous_point_id = centerline_points.InsertNextPoint(current_point)

    while np.linalg.norm(current_point - end_point) > tolerance:
        # Find the nearest node to the current point
        nearest_node_index = find_nearest_node(current_point, coordinates)

        # Get the gradient at the nearest node
        grad_at_point = gradient_array[nearest_node_index]

        # Normalize the gradient to get the direction
        direction = -grad_at_point / np.linalg.norm(grad_at_point)

        # Take a step in the direction of the gradient
        next_point = current_point + step_size * direction

        # Insert the new point
        point_id = centerline_points.InsertNextPoint(next_point)

        # Create a line between the previous and current point
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, previous_point_id)
        line.GetPointIds().SetId(1, point_id)
        centerline_lines.InsertNextCell(line)

        # Update the current point and previous_point_id
        current_point = next_point
        previous_point_id = point_id   
    # Create vtkPolyData to hold the centerline points and lines

    return centerline_points, centerline_lines


def center_point_cluster(mesh, new_clustermap, cluster_number, distance_map):
    """
    Find the middle point for the first or last cluster for the centerline
    extraction

    Parameters:
    - mesh: the entire mesh.
    - new_clustermap: the fenicsx function of the separated clustermap.
    - cluster_numer: the number of the cluster you want to find the middle point
    - distance map: this is a funciton with the distance of each node to the
    boundaries of the mesh (the first time the eikonal equation is solved)

    Returns:
    - end_point: array of the coordinates of the end point found
    """
    coordinates = mesh.geometry.x
    cluster_ind = np.where(new_clustermap.x.array == cluster_number)[0]
    max_dist_ind = cluster_ind[np.argmax(distance_map.x.array[cluster_ind])]
    max_dist_point = coordinates[max_dist_ind]

    return max_dist_point


def centerlines(mesh, new_clustermap, extreme_nodes, distance_map, gradient_array):
    """
    Trace centerlines from each extreme node to a common end point and combine them into one vtkPolyData.

    Parameters:
    - mesh: The mesh of the geometry.
    - new_clustermap: The function that maps the nodes to clusters.
    - extreme_nodes: The list of extreme nodes to start the centerlines from.
    - distance_map: The function with the distance of each node to the boundaries.
    - gradient_array: Array of gradient vectors at each node.

    Returns:
    - combined_centerline_polydata: vtkPolyData containing all the centerlines.
    """
    # Initialize vtkPoints and vtkCellArray for the combined centerline
    combined_centerline_points = vtk.vtkPoints()
    combined_centerline_lines = vtk.vtkCellArray()

    point_offset = 0  # Offset to keep track of the point IDs across different centerlines

    # Find the common end point
    end_pt = center_point_cluster(mesh, new_clustermap, 0, distance_map)
    print(end_pt)

    for j in range(len(extreme_nodes)):
        current_cluster = extreme_nodes[j]
        if current_cluster != 0:
            start_pt = center_point_cluster(mesh, new_clustermap, current_cluster, distance_map)
            vessel_points, vessel_lines = trace_centerline_vtk(mesh, gradient_array, start_pt, end_pt)

            # Append the vessel points and lines to the combined centerline
            for i in range(vessel_points.GetNumberOfPoints()):
                combined_centerline_points.InsertNextPoint(vessel_points.GetPoint(i))

            vessel_lines.InitTraversal()
            id_list = vtk.vtkIdList()
            while vessel_lines.GetNextCell(id_list):
                # Create a new line with updated point IDs
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, id_list.GetId(0) + point_offset)
                line.GetPointIds().SetId(1, id_list.GetId(1) + point_offset)
                combined_centerline_lines.InsertNextCell(line)
            
            # Update the point offset for the next centerline
            point_offset += vessel_points.GetNumberOfPoints()

    # Create vtkPolyData to hold the combined points and lines
    combined_centerline_polydata = vtk.vtkPolyData()
    combined_centerline_polydata.SetPoints(combined_centerline_points)
    combined_centerline_polydata.SetLines(combined_centerline_lines)

    return combined_centerline_polydata


def save_centerline_vtk(centerline_polydata, filename):
    """
    Save the centerline points and lines to a VTK file.

    Parameters:
    - centerline_polydata: vtkPolyData containing the centerline points and lines.
    - filename: The name of the output VTK file.
    """
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(centerline_polydata)
    writer.Write()
