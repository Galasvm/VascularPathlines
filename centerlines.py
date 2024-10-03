import numpy as np
import dolfinx
from dolfinx import fem, default_scalar_type, log
from dolfinx.fem import functionspace
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from dolfinx.io import XDMFFile
import ufl
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import vtk
import os


def solve_eikonal(domain, ft, distance: bool, c=1, *face_tag: int):
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

        # now we set the wave speed proportional to the minimum distance field
        f = 1/(2**(20*c))

    # CALCULATING EPS: We need to make eps related to the mesh size
    # finding the max cell size now
    hmax, hmin,_ = h_max(domain)
    print(f"min edge size: {hmin} and max edge size: {hmax}")
    eps = hmax/2
    print(f"eps: {eps}")

    u = set_problem(V, bc, eps, f)

    return u#, hmax


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
    h_avg = np.mean(h)
    h_max = max(h)
    h_min = min(h)
    return h_max, h_min, h_avg


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
    log.set_log_level(log.LogLevel.INFO)

    solver.solve(u)

    return u


def gradient_field(mesh, time_distance_map):

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


def cluster_map_dtf(values, num_clusters=25):

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

    return clustered_values


def discritize_dtf(dtf_map,mesh,num_clusters=25):
    dtf_values = dtf_map.x.array
    print(f"this is how many nodes there are: {len(dtf_values)}")
    clustered_values = cluster_map_dtf(dtf_values,num_clusters)

    V = functionspace(mesh, ("Lagrange", 1))
    u = fem.Function(V)

    # Assign the clustered values back to the array
    u.x.array[:] = clustered_values    
    return u


def spatial_clustering(mesh,clustersmap):

    coordinates = mesh.geometry.x
    num_clusters = int(np.max(clustersmap.x.array)) + 1
    dis_max, dis_min,_ = h_max(mesh)
    distance_threshold = dis_max * 0.8
    print(f"distance threshold: {distance_threshold}")
    print(f"min distance: {dis_min} and max distance: {dis_max}")

    new_clusters = clustersmap.x.array.copy()
    next_cluster_id = num_clusters
    components = {}

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

    return new_clusters

def merging_small_clusters(mesh, separate_clusters, cluster_threshold: int):
    coordinates = mesh.geometry.x
    separated_num_clusters = int(np.max(separate_clusters)) + 1
    tree = cKDTree(coordinates)  # Global k-d tree

    for cluster_id in range(separated_num_clusters):
        # Get the indices of nodes in the current cluster
        cluster_indices = np.where(separate_clusters == cluster_id)[0]

        if len(cluster_indices) < cluster_threshold:  # If the cluster is too small, merge it
            cluster_coords = coordinates[cluster_indices]
            _, nearest_neighbor_indices = tree.query(cluster_coords, k=2)

            # Get the cluster IDs of these nearest neighbors
            nearest_cluster_ids = separate_clusters[nearest_neighbor_indices[:, 1]]
            nearest_cluster_ids = nearest_cluster_ids.astype(int)
            nearest_cluster_ids = nearest_cluster_ids[nearest_cluster_ids != cluster_id]

            if len(nearest_cluster_ids) > 0:
                # Find the most common nearest cluster ID to merge into
                merge_target = np.bincount(nearest_cluster_ids).argmax()
                # Merge the small cluster into the selected nearby cluster
                separate_clusters[cluster_indices] = merge_target
        
    return separate_clusters

def noempty_clusters(mesh, new_clusters):
    coordinates = mesh.geometry.x
    tree = cKDTree(coordinates)
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

    return new_clusters

def extreme_clusters(mesh, cluster_map, distance_threshold):
    coordinates = mesh.geometry.x
    tree = cKDTree(coordinates)
    separated_num_clusters = int(np.max(cluster_map)) + 1
    cluster_neighbors = {cluster_id: set() for cluster_id in range(separated_num_clusters)}

    for cluster_id in range(separated_num_clusters):
        cluster_indices = np.where(cluster_map == cluster_id)[0]
        cluster_coords = coordinates[cluster_indices]

        if len(cluster_indices) < 2:  # Skip clusters too small to analyze
            continue

        for idx in cluster_indices:
            neighbors = tree.query_ball_point(coordinates[idx], distance_threshold)
            for neighbor_idx in neighbors:
                neighbor_cluster_id = cluster_map[neighbor_idx]
                if neighbor_cluster_id != cluster_id:
                    cluster_neighbors[cluster_id].add(neighbor_cluster_id)

    end_clusters = [cluster_id for cluster_id, neighbors in cluster_neighbors.items() if len(neighbors) < 2]

    return end_clusters


def separate_clusters(mesh, clustersmap, distance_threshold, cluster_threshold=20):

    separate_clusters = spatial_clustering(mesh,clustersmap)
    new_clusters = merging_small_clusters(mesh, separate_clusters, cluster_threshold)
    new_clusters_final = noempty_clusters(mesh,new_clusters)
    end_clusters = extreme_clusters(mesh, new_clusters_final,distance_threshold)

    # Create a new dolfinx function to hold the final clusters
    V = functionspace(mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.x.array[:] = new_clusters_final
    u.x.scatter_forward()

    return u, end_clusters


def h_max_cluster(mesh, cluster_indices):
    """
    Calculate the average edge size for the given cluster (using node indices).
    """
    # Get the topology and geometry of the mesh
    topology = mesh.topology
    geometry = mesh.geometry
    num_cells = topology.index_map(topology.dim).size_local
    cells = np.arange(num_cells, dtype=np.int32)

    # Create a sub-mesh for the current cluster based on its cells
    mesh_float = dolfinx.cpp.mesh.Mesh_float64(mesh.comm, mesh.topology, mesh.geometry)
    
    # Get the edge lengths for the cells in the current cluster
    h_cluster = dolfinx.cpp.mesh.h(mesh_float, topology.dim, cells[cluster_indices])
    
    # Calculate the average edge size
    hmax_cluster = max(h_cluster)
    hmin_cluster = min(h_cluster)
    havg_cluster = np.mean(h_cluster)
    
    return hmax_cluster, hmin_cluster, havg_cluster


def smaller_clusters(mesh, separate_cluster_map, distance_map, dtf_map):

    num_clusters = int(np.max(separate_cluster_map.x.array)) + 1
    distance_map_array = distance_map.x.array
    dtf_map_array = dtf_map.x.array
    
    # Get the average edge size
    hmax,hmin,havg = h_max(mesh)


    cluster_to_radii_array = np.zeros((num_clusters, 3))
    new_clusters = separate_cluster_map.x.array.copy()
    next_cluster_id = num_clusters
    
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(separate_cluster_map.x.array == cluster_id)[0]
        distance_values = distance_map_array[cluster_indices]
        max_radii = distance_values.max()
        hmax_c, hmin_c, havg_c = h_max_cluster(mesh, cluster_indices)
        cluster_to_radii_array[cluster_id,:] = [cluster_id,max_radii,len(cluster_indices)]
        l = havg_c/2
        number_pts_cluster = len(cluster_indices)
        pts_per_subcluster = (np.pi*max_radii**3*6*np.sqrt(2))//((l**3)*3.215)

        print(f"for cluster: {cluster_id}, hmax: {hmax_c} hmin: {hmin_c} and havg: {havg_c}")
        print(f"for cluster: {cluster_id}, the max radii is: {max_radii} and the number of pts in the cluster is: {number_pts_cluster} and the number of points per subcluster should be: {pts_per_subcluster}")
        if pts_per_subcluster<30:
            pts_per_subcluster+=40
        
        if 2*pts_per_subcluster<number_pts_cluster and number_pts_cluster>30:
   
            dtf_values = dtf_map_array[cluster_indices]
            num_subclusters = int(number_pts_cluster//pts_per_subcluster)

            new_cluster_indices = cluster_map_dtf(dtf_values, num_subclusters)
            for i in range(num_subclusters):
                # Find local component indices that belong to this subcluster
                component_local_indices = np.where(new_cluster_indices == i)[0]
                
                # Map local component indices to global indices
                component_global_indices = cluster_indices[component_local_indices]

                if i == 0:
                    # Keep the first component in the original cluster
                    new_clusters[component_global_indices] = cluster_id
                else:
                    # Assign a new cluster ID to the other components
                    new_clusters[component_global_indices] = next_cluster_id
                    next_cluster_id += 1
        else:
            new_clusters[cluster_indices]=cluster_id

    new_clusters = noempty_clusters(mesh,new_clusters)
    #end_clusters = extreme_clusters(mesh, new_clusters, hmax*0.5)

    V = functionspace(mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.x.array[:] = new_clusters
    u.x.scatter_forward()

    return u#, end_clusters


def rescale_distance_map(mesh, final_cluster_map, distance_map):
    num_clusters = int(np.max(final_cluster_map.x.array)) + 1
    distance_map_array = distance_map.x.array
    rescaled_distance_map_array = np.zeros_like(distance_map_array)

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(final_cluster_map.x.array == cluster_id)[0]
        distance_values = distance_map_array[cluster_indices]
        max_radii = distance_values.max()
        rescaled_values = distance_values/max_radii
        rescaled_distance_map_array[cluster_indices] = rescaled_values

    V = functionspace(mesh, ("Lagrange", 1))
    rescaled_distance_map = fem.Function(V)
    rescaled_distance_map.x.array[:] = rescaled_distance_map_array

    return rescaled_distance_map, rescaled_distance_map_array


def find_nearest_node(point, coordinates):
    distances = np.linalg.norm(coordinates - point, axis=1)
    return np.argmin(distances)


def interpolate_gradient(gradient_array, point, coordinates):
    """
    Interpolate the gradient at a given point using barycentric interpolation.

    Parameters:
    - gradient_array: Array of gradient vectors at each node (shape should be (num_nodes, dim)).
    - point: The point at which to interpolate the gradient (as a numpy array).
    - coordinates: The coordinates of all nodes (as a numpy array).

    Returns:
    - interpolated_gradient: The interpolated gradient at the given point.
    """
    # Find the Delaunay triangulation of the coordinates
    tri = Delaunay(coordinates)

    # Find the simplex (triangle/tetrahedron) that contains the point
    simplex = tri.find_simplex(point)

    # Get the vertices of the simplex
    vertices = tri.simplices[simplex]

    # Get the coordinates and gradients of the vertices
    #vertex_coords = coordinates[vertices]
    vertex_grads = gradient_array[vertices]

    # Compute the barycentric coordinates of the point within the simplex
    bary_coords = tri.transform[simplex, :3].dot(point - tri.transform[simplex, 3])
    bary_coords = np.append(bary_coords, 1 - bary_coords.sum())

    # Interpolate the gradient using barycentric coordinates
    interpolated_gradient = np.dot(bary_coords, vertex_grads)
    print(interpolated_gradient)

    return interpolated_gradient


def is_point_within_mesh(point, mesh_coordinates):
    """
    Check if the point is within the bounding box of the mesh.

    Parameters:
    - point: The point to check (as a numpy array).
    - mesh_coordinates: The coordinates of all nodes in the mesh (as a numpy array).

    Returns:
    - within_bounds: Boolean indicating if the point is within the mesh bounds.
    """
    min_bounds = np.min(mesh_coordinates, axis=0)
    max_bounds = np.max(mesh_coordinates, axis=0)

    return np.all(point >= min_bounds) and np.all(point <= max_bounds)


def trace_centerline_vtk(mesh, gradient_array, start_point, end_point, step_size=0.5, tolerance=1):

    """
    Trace a centerline from start_point to end_point by following the gradient of gradient_array,
    and save the points in VTK format.

    Parameters:
    - gradient_array: Array of gradient vectors at each node (shape should be (num_nodes, dim)).
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
        # Interpolate the gradient at the current point

        grad_at_point = interpolate_gradient(gradient_array, current_point, coordinates)
        # Normalize the gradient to get the direction
        direction = -grad_at_point / np.linalg.norm(grad_at_point)

        # Take a step in the direction of the gradient
        next_point = current_point + step_size * direction
       
        if not is_point_within_mesh(next_point, coordinates):
            print("Next point is outside the mesh bounds, stopping trace.")
            break
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
    centerline_polydata = vtk.vtkPolyData()
    centerline_polydata.SetPoints(centerline_points)
    centerline_polydata.SetLines(centerline_lines)

    return centerline_polydata#centerline_points, centerline_lines#centerline_polydata, centerline_lines


def trace_centerline_vtk_nodes(mesh, gradient_array, start_point, end_point, step_size=0.05, tolerance=0.5):
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
    centerline_polydata = vtk.vtkPolyData()
    centerline_polydata.SetPoints(centerline_points)
    centerline_polydata.SetLines(centerline_lines)
    return centerline_polydata#centerline_points, centerline_lines


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


def resample_centerline(polydata, subdivision_length):
    # Create a spline filter to resample the lines
    spline_filter = vtk.vtkSplineFilter()
    spline_filter.SetInputData(polydata)

    # Set the resolution (distance between points along the spline)
    spline_filter.SetSubdivideToLength()
    spline_filter.SetLength(subdivision_length)  # Set desired length between points

    # Update the spline filter
    spline_filter.Update()

    # Get the resampled polydata with points
    resampled_polydata = spline_filter.GetOutput()

    # Get the points from the resampled data
    resampled_points = resampled_polydata.GetPoints()

    # Create a new vtkPolyData to hold the resampled points and new lines
    new_polydata = vtk.vtkPolyData()
    new_polydata.SetPoints(resampled_points)

    # Create a vtkCellArray to store the new lines
    new_lines = vtk.vtkCellArray()

    # Manually connect the points by creating lines between consecutive points
    for i in range(resampled_points.GetNumberOfPoints() - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        new_lines.InsertNextCell(line)

    # Set the new lines to the new polydata
    new_polydata.SetLines(new_lines)

    return new_polydata


# Processing centerlines to merge
def merge_centerline_segments(centerline):
    """
    Merge all small lines for each tagged individual centerline into one long line.

    Parameters
    ----------
    centerline : vtkPolyData
        Combined centerline of the geometry with individual centerlines tagged by "CenterlineID".

    Returns
    -------
    merged_centerline : vtkPolyData
        The centerline with each individual tagged centerline merged into a single line.
    """
    # Use vtkStripper to merge connected line segments into one long line for each centerline
    stripper = vtk.vtkStripper()
    stripper.SetInputData(centerline)
    stripper.JoinContiguousSegmentsOn()  # Ensure connected segments are joined
    stripper.Update()

    # Get the merged centerline
    merged_centerline = stripper.GetOutput()

    return merged_centerline


def centerlines_with_tags(mesh, new_clustermap, extreme_nodes, distance_map, gradient_array, centerline_nodes=bool):
    # Initialize vtkPoints, vtkCellArray, and tag array for the combined centerline
    combined_centerline_points = vtk.vtkPoints()
    combined_centerline_lines = vtk.vtkCellArray()
    tags_array = vtk.vtkIntArray()  # This array will hold the tags for each line
    tags_array.SetName("LineTags")  # Name the array to identify it later

    point_offset = 0  # Offset to keep track of the point IDs across different centerlines


    # Find the common end point
    end_pt = center_point_cluster(mesh, new_clustermap, 0, distance_map)
    hmax, _ ,_= h_max(mesh)

    for j in range(len(extreme_nodes)):
        current_cluster = extreme_nodes[j]
        if current_cluster != 0:
            start_pt = center_point_cluster(mesh, new_clustermap, current_cluster, distance_map)
            if centerline_nodes is False:
                print("Centerline extraction using gradient")
                polydata_pre = trace_centerline_vtk(mesh, gradient_array, start_pt, end_pt, hmax, hmax*3)
                polydata = resample_centerline(polydata_pre,hmax/5)
                vessel_lines = polydata.GetLines()
            else:
                #vessel_points, vessel_lines = trace_centerline_vtk_nodes(mesh, gradient_array, start_pt, end_pt, hmax, hmax*3)
                polydata_pre = trace_centerline_vtk_nodes(mesh, gradient_array, start_pt, end_pt, hmax, hmax*3)
                polydata = resample_centerline(polydata_pre,hmax/5)
                vessel_lines = polydata.GetLines()
                print("Extracting centerline based on nodes")

            # Append the vessel points and lines to the combined centerline
            for i in range(polydata.GetNumberOfPoints()):
                point = polydata.GetPoint(i)
                combined_centerline_points.InsertNextPoint(point)


            
            vessel_lines.InitTraversal()
            id_list = vtk.vtkIdList()

            # Tag lines with a specific value (e.g., cluster number)
            line_tag = j  # Tag lines with th

            while vessel_lines.GetNextCell(id_list):
                # Create a new line with updated point IDs
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, id_list.GetId(0) + point_offset)
                line.GetPointIds().SetId(1, id_list.GetId(1) + point_offset)
                combined_centerline_lines.InsertNextCell(line)

                # Add the tag (e.g., cluster number) to the tag array for this line
                tags_array.InsertNextValue(line_tag)

            # Update the point offset for the next centerline
            point_offset += polydata.GetNumberOfPoints()

    # Create vtkPolyData to hold the combined points and lines
    combined_centerline_polydata = vtk.vtkPolyData()
    combined_centerline_polydata.SetPoints(combined_centerline_points)
    combined_centerline_polydata.SetLines(combined_centerline_lines)

    # Attach the tags array to the cell data (for lines)
    combined_centerline_polydata.GetCellData().AddArray(tags_array)
    final_pd = merge_centerline_segments(combined_centerline_polydata)

    return final_pd



# creating dictionary for Bryan's merging code. This is Bryan's modified function
def create_dict(pd):
    dict_cell = {}  # key = cell number; value = [vtkPolyLine, [pt1, pt2, ...]]
    distance_lst = [[0, 0, 0] for _ in range(pd.GetNumberOfCells())] # [distance, vtkPolyLine, [pt1, pt2, ...]]
    
    for i in range(pd.GetNumberOfCells()):
        # key = i; value = 
        cell = pd.GetCell(i)
        #print(cell)
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
                distance_lst[i][0] += np.linalg.norm(np.array(init_pt_array[j]) - np.array(init_pt_array[j-1]))

        # reverse the coordinate ordering due to Eikonal equation
        
        distance_lst[i][1] = cell
        distance_lst[i][2] = init_pt_array
        # if len(distance_lst[i][2]) != len(distance_lst[i][3]):
        #     print('error, the length of the coordinates and max_inscribed_radii are not the same')
        # #print(len(distance_lst[i][2]))
        # #print(len(distance_lst[i][3]))
    
    # sort the distance_lst based on the first index
    # re-order dict_cell content base on distance (from longest to shortest)
    distance_lst = sorted(distance_lst, key=lambda x: x[0], reverse=True)
    
    # dict_cell[0] has the longest distance
    for i in range(len(distance_lst)): 
        # save the cell [vtp cell, pt coordinates]
        dict_cell[i] = []
        dict_cell[i].append(distance_lst[i][1])
        
        #dict_cell[i].append(points)
        dict_cell[i].append(distance_lst[i][2])
        #print(len(distance_lst[i][2]))
        # dict_cell[i].append(distance_lst[i][3])
        #print(len(distance_lst[i][3]))
    
    return pd, dict_cell

# helper functions for bryans combining cls into one polydata function. Also Bryan's

def create_polydata_from_edges(edges, points):
    #print(edges)
    #print(points)
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
    #"""Returns coordinate of point on polydata closest to refpoint."""
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
            spacing.append(np.linalg.norm(np.array(pts[j]) - np.array(pts[j+1])))
    return np.mean(spacing)


#modified Bryan's function (no radii) and changed the direction of cl

def combine_cls_into_one_polydata(dict_cell, tolerance=0.05):
    """
    combine all centerlines into one polydata with the right connectivity/direction
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
            edges.append([i,i+1])
        return edges
    def is_close(coord1, coord2, tol=0.1):
        """Helper function to compare coordinates with a tolerance."""
        return np.linalg.norm(np.array(coord1) - np.array(coord2)) < tol

    def find_point_index_in_master_coords(master_coords, coord, tol=0.01):
        """Find the index of a coordinate in master_coords, considering tolerance."""
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
        if i == 0: # using the first cl as ground to grow
            length = len(master_coords)
            master_coords.extend(dict_cell[i][1])
            addition = len(master_coords) - length
            edges = create_edges(length,length+addition-1)
            master_edges.extend(edges)
            temp_pd = create_polydata_from_edges(master_edges,create_vtkpoints_from_coordinates(master_coords))
            # prepare to update dict_cell
            new_dict_cell_pd[i] = temp_pd
            new_dict_cell_points[i] = dict_cell[i][1]
            new_dict_cell_edges[i] = edges

            print(f"Done with centerline {i}")
            #write_polydata(temp_pd, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\cl0.vtp')
        else:
            # GALA CHANGE: mine are not backwards
            #backward_cl = dict_cell[i][1][::-1]
            # if i >= 1 or i <=8:
            #     backward_cl = dict_cell[i][1][::1]

            # else:
            #     backward_cl = dict_cell[i][1][::-1]
            # find the closest point in the master_coords
            backward_cl = dict_cell[i][1][::1]
            coords_to_add = []

            count_addition = 0
            for j in range(len(backward_cl)):
                coord = backward_cl[j]

                closest_point = findclosestpoint(temp_pd,coord)
                # print(f'number is {j}')
                # print(f"coord: {coord}")
                # print(f"closest_point on polydata: {closest_point}")
                # print(f"distance: {np.linalg.norm(np.array(coord) - np.array(closest_point))}")
                
                # when the other line gets close to the existing line, stop
                if np.linalg.norm(np.array(coord) - np.array(closest_point)) < tolerance:
                    # find the index of the closest point
                    coords_to_add.append(coord)
                    count_addition += 1
                    index_for_connecting_pt = find_point_index_in_master_coords(master_coords,closest_point)
                    print(f"index_for_connecting_pt: {index_for_connecting_pt}, therefore this edge is [{index_for_connecting_pt},{len(master_coords)}]")
                    if index_for_connecting_pt is None:
                        print("warning: closest point not found. something must be wrong")
                    break
                else:
                    coords_to_add.append(coord)
                    count_addition += 1
            # flip coords_to_add so that it follows flow direction
            coords_to_add = coords_to_add[::-1]
            
            # create edges: first coords is the bifurcation
            edges = create_edges(len(master_coords),len(master_coords)+count_addition-1)
            edges.insert(0,[index_for_connecting_pt,len(master_coords)])
            
            # add to master_coords
            master_coords.extend(coords_to_add)
            master_edges.extend(edges)
            temp_pd = create_polydata_from_edges(master_edges,create_vtkpoints_from_coordinates(master_coords))
            #write_polydata(temp_pd, f'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\cl{i}.vtp')
            print(f"Done with centerline {i}")
            #pdb.set_trace()
            # save to new dict_cell 
            for k in range(len(new_dict_cell_points)):
                if master_coords[index_for_connecting_pt] in new_dict_cell_points[k]:
                    

                    new_dict_cell_points[i] = new_dict_cell_points[k][0:new_dict_cell_points[k].index(master_coords[index_for_connecting_pt])+1]+coords_to_add
                    new_dict_cell_edges[i] = new_dict_cell_edges[k][0:new_dict_cell_points[k].index(master_coords[index_for_connecting_pt])]+edges
                    new_dict_cell_pd[i] = create_polydata_from_edges(new_dict_cell_edges[i],create_vtkpoints_from_coordinates(new_dict_cell_points[i]))
                    
                    break
    
    # pdb.set_trace()
    points = vtk.vtkPoints()
    for coord in master_coords:
        points.InsertNextPoint(coord)
    
    pd = create_polydata_from_edges(master_edges,points)
    #write_polydata(pd, save_dir+'edge_created_combined_cl.vtp')
    # recreate dict_cell from what we have
    dict_cell = {}
    for i in range(len(new_dict_cell_pd)):
        dict_cell[i] = [new_dict_cell_pd[i],new_dict_cell_points[i]]

    # pdb.set_trace()
    
    return pd, master_coords, master_edges, dict_cell


def save_centerline_vtk(centerline_polydata, filename):
    """
    Save the centerline points and lines to a VTK file.

    Parameters:
    - centerline_polydata: vtkPolyData containing the centerline points and lines.
    - filename: The name of the output VTK file.
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(centerline_polydata)
    writer.Write()

