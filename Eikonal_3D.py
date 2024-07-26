import numpy as np
from dolfinx import  fem, default_scalar_type
from dolfinx.fem import functionspace
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from dolfinx.io import XDMFFile
import ufl

def solve_eikonal(domain, ft, distance: bool, c = 1 ,*face_tag: int):

   
    V = functionspace(domain, ("Lagrange", 1))

    # putting all the dofs of the faces we want to select for the boundary condition
    face_tags = list(face_tag)
    count_face_tags = len(face_tags)

    # seeing if there is more than one face selected
    if count_face_tags > 1:
        if distance == False:
            print("error: Should only select one face when solving solving with a seedpoint")
            exit()


    all = np.array([])
    for tag in face_tags:
        face = ft.find(tag)
        all = np.append(all,face)


    if distance == True:
        # defining the speed as 1 since we are looking for the distance field
        f = fem.Constant(domain, default_scalar_type(1))
        # setting the wall of the vessel as the boundary condition 
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        wall_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, all)
        bc = fem.dirichletbc(default_scalar_type(0), wall_dofs, V)

    else:
        # defining f as 1 over the distance of each node
        f = 1/(1+c)
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        inlet_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, all)
        # now we need to find the dof at the center of the facet to select as the source point (the point with the greater distance from the walls)

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


    # Define the unknown function and test function
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    uh = ufl.TrialFunction(V)

    # Initialize the solution to avoid convergence issues
    with u.vector.localForm() as loc:
        loc.set(1.0)

    # Initialization problem to get good initial guess
    a = ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx
    L = f*v*ufl.dx
    problem = fem.petsc.LinearProblem(a, L, bcs = [bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u = problem.solve()

    eps = 0.5
    F = ufl.sqrt(ufl.inner(ufl.grad(u), ufl.grad(u)))*v*ufl.dx - f*v*ufl.dx + eps * ufl.inner(ufl.grad(abs(u)), ufl.grad(v)) * ufl.dx

    # Jacobian of F
    J = ufl.derivative(F, u, ufl.TrialFunction(V))

    # Create nonlinear problem and solver
    problem = fem.petsc.NonlinearProblem(F, u, bcs = [bc], J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    # Set solver options
    solver.rtol = 1e-6
    solver.max_it = 80 # Increase the max iterations

    # Solve the problem
    solver.solve(u)

    return u


def import_mesh(path_mesh, path_facets):

    with XDMFFile(MPI.COMM_WORLD, path_mesh, "r") as xdmf:
        domain_ = xdmf.read_mesh(name="Grid")
        #ct = xdmf.read_meshtags(domain_, name="Grid")

    domain_.topology.create_connectivity(domain_.topology.dim, domain_.topology.dim - 1)

    with XDMFFile(MPI.COMM_WORLD, path_facets, "r") as xdmf:
        facet_ = xdmf.read_meshtags(domain_, name="Grid")
    
    return domain_, facet_


def export_soln(path_export, mesh_, function_):

    # Save the solution in XDMF format for visualization
    with XDMFFile(MPI.COMM_WORLD, path_export, "w") as file:
        file.write_mesh(mesh_)
        file.write_function(function_)


# FOR 0103_H_PULM_H_coarse
face_tags_list = list(range(1, 67))
domain, facet = import_mesh("Meshes/0103_H_PULM_H_coarse/0103_H_PULM_H_coarse.xdmf","Meshes/0103_H_PULM_H_coarse/0103_H_PULM_H_coarse_facet_markers.xdmf")
dis = solve_eikonal(domain, facet, True, 1 ,*face_tags_list)
solution = solve_eikonal(domain, facet, False, dis , 87)

export_soln("results/Eikonal_3D/newcode_0725/0103_H_PULM_H_coarse/distance_field.xdmf",domain,dis)
#exporting solution
export_soln("results/Eikonal_3D/newcode_0725/0103_H_PULM_H_coarse/final_soln.xdmf",domain,solution)

