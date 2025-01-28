# import vtk
# import os

# output_directory = 'stlmodel'
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)
    
# # Read the VTP file
# reader = vtk.vtkXMLPolyDataReader()
# reader.SetFileName('/Users/galasanchezvanmoer/Desktop/PhD_Project/VMR_models/SVProject/Models/demo.vtp')
# reader.Update()
# poly_data = reader.GetOutput()

# # Write to STL
# writer = vtk.vtkSTLWriter()
# writer.SetFileName('stlmodel/demo.stl')
# writer.SetInputData(poly_data)
# writer.Write()

# import meshio
# import os

# # Input and output paths
# input_mesh_file = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal/stlmodel/demomesh.msh"  # Replace with your .msh file
# output_mesh_file = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal/stlmodel/demomesh.xdmf"  # Path to save the unified .xdmf file

# # Read the .msh file
# msh = meshio.read(input_mesh_file)

# # Combine all cells into a single mesh object
# unified_mesh = meshio.Mesh(points=msh.points, cells=msh.cells)

# # Preserve physical groups if they exist
# if "cell_data" in msh and "gmsh:physical" in msh.cell_data_dict:
#     unified_mesh.cell_data["gmsh:physical"] = msh.cell_data["gmsh:physical"]

# # Write the combined mesh to a single XDMF file
# meshio.write(output_mesh_file, unified_mesh)

# print(f"Unified mesh saved at: {output_mesh_file}")
# import meshio

# # Read the 3D .msh file
# msh = meshio.read("/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal/stlmodel/demomesh.msh")

# # Separate the 3D volume elements and surface (boundary) elements
# volume_mesh = meshio.Mesh(
#     points=msh.points,
#     cells={"tetra": msh.get_cells_type("tetra")}
# )
# boundary_mesh = meshio.Mesh(
#     points=msh.points,
#     cells={"triangle": msh.get_cells_type("triangle")}
# )

# # Preserve physical groups if they exist
# if "gmsh:physical" in msh.cell_data_dict:
#     if "tetra" in msh.cell_data_dict["gmsh:physical"]:
#         volume_mesh.cell_data["gmsh:physical"] = [
#             msh.cell_data_dict["gmsh:physical"]["tetra"]
#         ]
#     if "triangle" in msh.cell_data_dict["gmsh:physical"]:
#         boundary_mesh.cell_data["gmsh:physical"] = [
#             msh.cell_data_dict["gmsh:physical"]["triangle"]
#         ]

# # Write the 3D volume mesh and the surface mesh to XDMF
# meshio.write("volume_mesh.xdmf", volume_mesh)
# meshio.write("boundary_mesh.xdmf", boundary_mesh)

# # Export the XDMF mesh
# def export_xdmf(mesh, file_path):
#     meshio.write(file_path, mesh)

# # Export the volume mesh and boundary mesh to XDMF
# export_xdmf(volume_mesh, "exported_volume_mesh.xdmf")
# export_xdmf(boundary_mesh, "exported_boundary_mesh.xdmf")

# print("Meshes have been successfully exported to XDMF format.")

import meshio

# Read the VTU file
vtu_mesh = meshio.read("/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal_mine/results/12082024/from_model_to_xdmf_run5/integration_parameters_change/numi_unsmoothed_models/pulm_monstrosity/results2_clean.vtu")

# Write it as an XDMF file
meshio.write("/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal_mine/results/12082024/from_model_to_xdmf_run5/integration_parameters_change/numi_unsmoothed_models/pulm_monstrosity/results2_clean.xdmf", vtu_mesh)