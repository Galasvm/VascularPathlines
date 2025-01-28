# import vtk

# # Step 1: Load the genus-zero triangulated surface mesh (VTP file)
# reader = vtk.vtkXMLPolyDataReader()
# reader.SetFileName("/Users/galasanchezvanmoer/Desktop/PhD_Project/VMR_models/SVProject/Models/demo.vtp")  # Replace with your file path
# reader.Update()
# surface_mesh = reader.GetOutput()

# # Step 2: Extract points from the surface mesh
# # vtkDelaunay3D works on points, so we use the points from the mesh
# points = vtk.vtkPoints()
# points.DeepCopy(surface_mesh.GetPoints())

# # Create a vtkPolyData to hold the points
# point_cloud = vtk.vtkPolyData()
# point_cloud.SetPoints(points)

# # Step 3: Apply Delaunay tetrahedralization
# delaunay = vtk.vtkDelaunay3D()
# delaunay.SetInputData(point_cloud)
# delaunay.Update()

# # Step 4: Save the tetrahedralization as a VTU file for visualization in ParaView
# writer = vtk.vtkXMLUnstructuredGridWriter()
# writer.SetFileName("delaunay_tetrahedralization.vtu")  # Output file name
# writer.SetInputData(delaunay.GetOutput())
# writer.Write()

# print("Delaunay tetrahedralization saved as 'delaunay_tetrahedralization.vtu'")

h=(( -0.00872387 - 1.9412) ** 2 + (0.0557245 + 0.435705) ** 2 + ( 15 -15) ** 2) ** 0.5
print(h)