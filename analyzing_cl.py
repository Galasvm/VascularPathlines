import vtk
import os
import csv
import yaml
from centerline_extraction import save_centerline_vtk

parent = os.path.dirname(__file__)
yaml_file = "0168_H_PULMFON_SVD_15.yaml"
gt = True

with open(parent + "/yaml_files/" + yaml_file, "rb") as f:
    params = yaml.safe_load(f)

yaml_file_name = params["file_name"]
# mesh_path = parent + params["saving_dir"] + yaml_file_name + "/eikonal" + yaml_file_name + "_dis_map_p0_000000.vtu"
#this one is for the mesh refinement 
# mesh_path = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal/results/11062024/mesh_refinement_study/0168_H_PULMFON_SVD_05/eikonal/0168_H_PULMFON_SVD_05_dis_map_p0_000000.vtu"
#GOOD DISTANCE FIELDS
mesh_path = parent + "/Experiments_11072024/fine_distance_fields" + yaml_file_name + "_FINE/eikonal" + yaml_file_name + "_FINE_dis_map_p0_000000.vtu"


# Function to load a VTK file
# def load_vtk_file(file_path, reader_type):
#     reader = reader_type()
#     reader.SetFileName(file_path)
#     reader.Update()
#     return reader.GetOutput()
def load_vtk_file(fname):
    """
    Read geometry from file, chose corresponding vtk reader
    Args:
        fname: vtp surface or vtu volume mesh

    Returns:
        vtk reader, point data, cell data
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader.GetOutput()

# Function to write slice data to CSV
def write_csv(file_path, data):
    with open(file_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["slice_number", "max_distance_value", "centerline_distance_value", "difference", "average_difference"])
        writer.writeheader()
        writer.writerows(data)

if gt == True:

    centerline_path = parent + "/GT_centerlines" + yaml_file_name + "_gt.vtp"


    # # Check if either .vtp or .vtu file exists
    # if os.path.exists(base_path + ".vtp"):
    #     centerline_path = base_path + ".vtp"
    # elif os.path.exists(base_path + ".vtu"):
    #     centerline_path = base_path + ".vtu"
    # else:
    #     raise FileNotFoundError("No valid .vtp or .vtu file found at the specified path.")
    # output_folder = parent + "/outputs_analyzing" + yaml_file_name + "/gt/"
    output_folder = parent + "/Experiments_11072024" + yaml_file_name + "/gt/"

    slice_interval = 100
    csv_file_path = output_folder + yaml_file_name + "_gt_slice_data.csv" # 
    combined_vtp_path = output_folder + yaml_file_name + "_gt_combined_data.vtp"
else:
    # centerline_path = parent + params["saving_dir"] + yaml_file_name + "/centerlines" + yaml_file_name + "smooth_centerline.vtp"
    centerline_path = parent + "/Experiments_11072024/mine_centerlines" + yaml_file_name + "/centerlines" + yaml_file_name + "smooth_centerline.vtp"
    # base_path = f"{parent}/Experiments_11072024/mine_centerlines{yaml_file_name}/centerlines{yaml_file_name}smooth_centerline"
    # if os.path.exists(base_path + ".vtp"):
    #     centerline_path = base_path + ".vtp"
    # elif os.path.exists(base_path + ".vtu"):
    #     centerline_path = base_path + ".vtu"
    # else:
    #     raise FileNotFoundError("No valid .vtp or .vtu file found.")
    output_folder = parent + "/outputs_analyzing" + yaml_file_name + "/mine/"
    # output_folder = parent + "/Experiments_11072024" + yaml_file_name + "/mine/"

    slice_interval = 20
    csv_file_path = output_folder +yaml_file_name + "_mine_slice_data.csv"# 
    combined_vtp_path = output_folder + yaml_file_name + "_mine_combined_data.vtp"
# Set paths
# centerline_path = parent + params["saving_dir"] + yaml_file_name + "/centerlines" + yaml_file_name + "smooth_centerline.vtp"
# output_folder = f"{parent}/outputs_analyzing{yaml_file_name}/slices/"
# slice_interval = 20
# csv_file_path = f"{output_folder}{yaml_file_name}_slice_data.csv"
# combined_vtp_path = f"{output_folder}{yaml_file_name}_combined_slices.vtp"
# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the centerline and vascular mesh
# centerline = load_vtk_file(centerline_path, vtk.vtkXMLPolyDataReader)
centerline = load_vtk_file(centerline_path)

save_centerline_vtk(centerline, output_folder + "centerline" + yaml_file_name + "_centerline.vtp")
# mesh = load_vtk_file(mesh_path, vtk.vtkXMLUnstructuredGridReader)
mesh = load_vtk_file(mesh_path)


# Initialize a slicer for the distance field and an append object for combining slices
cutter = vtk.vtkCutter()
cutter.SetInputData(mesh)
append_filter = vtk.vtkAppendPolyData()  # To accumulate slices into one VTP file

# Define clipping sphere radius
clip_radius = 0.3  # Adjust based on geometry scale

# Retrieve the distance field array from the mesh
point_locator = vtk.vtkPointLocator()
point_locator.SetDataSet(mesh)
point_locator.BuildLocator()
distance_field_array = mesh.GetPointData().GetArray("f")

# Use vtkProbeFilter to interpolate distance values at centerline points
probe_filter = vtk.vtkProbeFilter()
probe_filter.SetInputData(centerline)
probe_filter.SetSourceData(mesh)
probe_filter.Update()

# Retrieve interpolated distance values from the centerline
interpolated_distance_array = probe_filter.GetOutput().GetPointData().GetArray("f")

# Process and store results for each slice in a list
slice_data = []
num_points = centerline.GetNumberOfPoints()

# Start slicing from point 20
for i in range(20, num_points, slice_interval):
    current_point = centerline.GetPoint(i)
    next_point = centerline.GetPoint(i + 1) if i < num_points - 1 else centerline.GetPoint(i - 1)
    
    # Calculate tangent vector and normalize
    tangent_vector = [next_point[j] - current_point[j] for j in range(3)]
    vtk.vtkMath.Normalize(tangent_vector)
    
    # Define slicing plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(current_point)
    plane.SetNormal(tangent_vector)
    
    # Set cutter to slice the mesh with the plane
    cutter.SetCutFunction(plane)
    cutter.Update()
    sliced_output = cutter.GetOutput()

    # Clip the slice around the centerline point
    clip_sphere = vtk.vtkSphere()
    clip_sphere.SetCenter(current_point)
    clip_sphere.SetRadius(clip_radius)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(sliced_output)
    clipper.SetClipFunction(clip_sphere)
    clipper.InsideOutOn()  # Keep only points inside the sphere
    clipper.Update()

    clipped_slice = clipper.GetOutput()

    # Extract interpolated distance value at the centerline point
    centerline_distance_value = interpolated_distance_array.GetValue(i)

    # Extract distance values from the clipped slice
    slice_distance_values = [
        distance_field_array.GetValue(point_locator.FindClosestPoint(clipped_slice.GetPoint(j)))
        for j in range(clipped_slice.GetNumberOfPoints())
    ]

    # Compute max distance in slice
    max_distance_value = max(slice_distance_values) if slice_distance_values else None

    # Save each clipped slice as a VTP file
    slice_file_path = f"{output_folder}slice_{i}.vtp"
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(clipped_slice)
    writer.SetFileName(slice_file_path)
    writer.Write()
    print(f"Saved localized slice {i} to {slice_file_path}")

    # Append slice to the combined VTP
    append_filter.AddInputData(clipped_slice)

    # Collect data for CSV
    slice_data.append({
        "slice_number": i,
        "max_distance_value": max_distance_value,
        "centerline_distance_value": centerline_distance_value,
        "difference": abs(max_distance_value - centerline_distance_value) if max_distance_value is not None else None
    })

# Calculate the average of the differences
differences = [info["difference"] for info in slice_data if info["difference"] is not None]
average_difference = sum(differences) / len(differences) if differences else None

# Add the average difference to each row
for info in slice_data:
    info["average_difference"] = average_difference

# Write the slice data to a CSV file with the new column
write_csv(csv_file_path, slice_data)
print(f"CSV data written to {csv_file_path}")
print("All localized slices saved successfully.")

# Write the combined VTP file with all slices
append_filter.Update()
combined_writer = vtk.vtkXMLPolyDataWriter()
combined_writer.SetInputData(append_filter.GetOutput())
combined_writer.SetFileName(combined_vtp_path)
combined_writer.Write()
with open(csv_file_path + "_average_difference.txt", "w") as file:
    file.write(f"Average Difference: {average_difference}")
    file.close()
print(f"Combined VTP file with all slices saved to {combined_vtp_path}")
