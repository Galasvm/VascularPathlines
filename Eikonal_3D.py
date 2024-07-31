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
yaml_file = "0103_H_PULM_H_coarse.yaml"

# Determine parent folder
parent = os.path.dirname(__file__)

# Read yaml file located in simvascular_mesh/yaml_files
with open(parent + "/yaml_files/" + yaml_file, "rb") as f:
    params = yaml.safe_load(f)

yaml_file_name = params["file_name"]
save_dir = parent + params["saving_dir"] + yaml_file_name + yaml_file_name
mesh_dir = parent + "/Meshes" + yaml_file_name + yaml_file_name


def solve_eikonal(domain, ft, distance: bool, c=1, *face_tag: int):
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

    if distance is True:
        # defining the speed as 1 since we are looking for the distance field
        f = fem.Constant(domain, default_scalar_type(1))
        # setting the wall of the vessel as the boundary condition
        domain.topology.create_connectivity(domain.topology.dim - 1,
                                            domain.topology.dim)
        wall_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                all)
        bc = fem.dirichletbc(default_scalar_type(0), wall_dofs, V)

    else:
        # defining f as 1 over the distance of each node
        f = 1/(2**(100*c))
        domain.topology.create_connectivity(domain.topology.dim - 1,
                                            domain.topology.dim)
        inlet_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                 all)
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

    # We need to make eps related to the mesh size
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    domain = dolfinx.cpp.mesh.Mesh_float64(domain.comm, domain.topology,
                                           domain.geometry)
    h = dolfinx.cpp.mesh.h(domain, tdim, cells)
    hmax = max(h)
    print(f"max cell size: {hmax}")
    eps = hmax/2
    print(f"eps: {eps}")

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
    dis = solve_eikonal(domain, facet, True, 1, *face_tags)

else:
    dis = solve_eikonal(domain, facet, True, 1, params["wall_face_tag"])

# Checking if only solving for the distance
if params["just_distance"] is False:
    solution = solve_eikonal(domain, facet, False, dis,
                             params["inlet_face_tag"])

# Checking if the solution should be saved
if params["save_eikonal"] is True:
    export_soln(save_dir + "_distance_field.xdmf", domain, dis)
    if params["just_distance"] is False:
        export_soln(save_dir + "_final_soln.xdmf", domain, solution)
