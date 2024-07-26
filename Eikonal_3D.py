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

'''
# FOR 0100_A_AO_COA

domain, facet = import_mesh("Meshes/0100_A_AO_COA_coarse/0100_A_AO_COA_coarse.xdmf","Meshes/0100_A_AO_COA_coarse/0100_A_AO_COA_coarse_facet_markers.xdmf")
dis = solve_eikonal(domain, facet, True, 1 , 1)
solution = solve_eikonal(domain, facet, False, dis , 2)

export_soln("results/Eikonal_3D/newcode_0725/0100_A_AO_COA/distance_field.xdmf",domain,dis)
#exporting solution
export_soln("results/Eikonal_3D/newcode_0725/0100_A_AO_COA/final_soln.xdmf",domain,solution)
'''
'''

# FOR CYLINDER

domain, facet = import_mesh("Meshes/cylinder/cylinder.xdmf","Meshes/cylinder/cylinder_facet_markers.xdmf")
dis = solve_eikonal(domain, facet, True, 1 , 1, 2)
solution = solve_eikonal(domain, facet, False, dis , 2,1)

export_soln("results/Eikonal_3D/newcode_0725/cylinder/distance_field.xdmf",domain,dis)
#exporting solution
export_soln("results/Eikonal_3D/newcode_0725/cylinder/final_soln.xdmf",domain,solution)
'''

'''
def compute_distance(domain, ft, wall_tag): # first input the mesh, the facet tags, and the wall tag

    V = functionspace(domain, ("Lagrange", 1))

    # finding the wall facet to set a dirichlet BC to 0
    wall = ft.find(wall_tag)

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    wall_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, wall)
    bc_w = fem.dirichletbc(default_scalar_type(0), wall_dofs, V)

    # Source term is constant in this case, unit speed
    f = fem.Constant(domain, default_scalar_type(1))

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
    problem = fem.petsc.LinearProblem(a, L, bcs = [bc_w], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u = problem.solve()

    eps = 0.1
    F = ufl.sqrt(ufl.inner(ufl.grad(u), ufl.grad(u)))*v*ufl.dx - f*v*ufl.dx + eps * ufl.inner(ufl.grad(abs(u)), ufl.grad(v)) * ufl.dx


    # Jacobian of F
    J = ufl.derivative(F, u, ufl.TrialFunction(V))

    # Create nonlinear problem and solver
    problem = fem.petsc.NonlinearProblem(F, u, bcs = [bc_w], J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    # Set solver options
    solver.rtol = 1e-6
    solver.max_it = 50 # Increase the max iterations

    # Solve the problem
    solver.solve(u)
    return u

def centerlines(domain, ft, inlet_tag, c):


    V = functionspace(domain, ("Lagrange", 1))

    # Define boundary condition
    # finding the  facet to set a point to 0 
    bottom = ft.find(inlet_tag)
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    bottom_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, bottom)
   
    # now we need to find the dof at the center of the facet to select as the source point (the point with the greater distance from the walls)

    c_values = c.vector.array  # Get the solution values as a numpy array
    # Find the index of the maximum value in the selected values

    selected_values = c_values[bottom_dofs]
    max_selected_index = np.argmax(selected_values)

    # Find the corresponding dof (global index)
    max_dof = bottom_dofs[max_selected_index]

    # make it into an array
    bottom_dof = np.array([max_dof])

    # define the dirichlet boundary condition at one point
    bc_center = fem.dirichletbc(default_scalar_type(0), bottom_dof, V)
    bcs = [bc_center]

    # Source term 
    f = 1/(1+c)
    #f = fem.Constant(domain, default_scalar_type(1))
    
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
    problem = fem.petsc.LinearProblem(a, L, bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u = problem.solve()

    eps = 0.1
    F = ufl.sqrt(ufl.inner(ufl.grad(u), ufl.grad(u)))*v*ufl.dx - f*v*ufl.dx + eps * ufl.inner(ufl.grad(abs(u)), ufl.grad(v)) * ufl.dx


    # Jacobian of F
    J = ufl.derivative(F, u, ufl.TrialFunction(V))

    # Create nonlinear problem and solver
    problem = fem.petsc.NonlinearProblem(F, u, bcs, J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    # Set solver options
    solver.rtol = 1e-6
    solver.max_it = 50 # Increase the max iterations

    solver.solve(u)

    return u
 
'''
'''

#dis2 = compute_distance(domain, facet, 1)
#soln = centerlines(domain, facet, 2, dis)
#exporting distance field
#export_soln("results/Eikonal_3D/cylinder/trial/final_distanceboundary.xdmf",domain,dis)
#exporting solution
#export_soln("results/Eikonal_3D/cylinder/trial/final_soln.xdmf",domain,soln)


'''


'''
# THIS IS WHERE YOU CHANGE THE PATHS TO WHATEVER YOU NEED (as long as the wall of the mesh is the first tag on the facet)
domain, facet = import_mesh("Meshes/demomesh/demomesh.xdmf","Meshes/demomesh/demomesh_facet_markers.xdmf")
dis = compute_distance(domain, facet, 1)
soln = centerlines(domain, facet, 2, 69, dis)
#exporting distance field
export_soln("results/Eikonal_3D/cylinder/final_distanceboundary.xdmf",domain,dis)
#exporting solution
export_soln("results/Eikonal_3D/final_soln.xdmf",domain,soln)
'''
