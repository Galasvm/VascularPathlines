import vtk
import os
import csv
import yaml
from tracingcenterlines_old import save_centerline_vtk
import statistics


# h=(( -0.0360913 + 0.162105) ** 2 + (-0.643912 + 0.439815) ** 2 + ( -0.585914 + 0.657529) ** 2) ** 0.5
# h=(( -0.00872387 - 1.9412) ** 2 + (0.0557245 + 0.435705) ** 2 + ( 15 -15) ** 2) ** 0.5


# parent = os.path.dirname(__file__)
# yaml_file = "0100_A_AO_COA.yaml"
# gt = True
# with open(parent + "/yaml_files/" + yaml_file, "rb") as f:
#     params = yaml.safe_load(f)
# yaml_file_name = params["file_name"]


# mesh_path = parent + params["saving_dir"] + yaml_file_name + "/eikonal" + yaml_file_name + "_dis_map_p0_000000.vtu"
#this one is for the mesh refinement 
# mesh_path = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal/results/11062024/mesh_refinement_study/0168_H_PULMFON_SVD_05/eikonal/0168_H_PULMFON_SVD_05_dis_map_p0_000000.vtu"
#GOOD DISTANCE FIELDS
# mesh_path = parent + "/analyzing_results/11142024/fine_distance_fields" + yaml_file_name + "_FINE/eikonal" + yaml_file_name + "_FINE_dis_map_p0_000000.vtu"
# my_centerline_path = parent + "/analyzing_results/11142024/mine_centerlines" + yaml_file_name + "/centerlines" + yaml_file_name + "_smooth_centerline.vtp"# Function to load a VTK file

# # this is for numi 
# # my_centerline_path = parent + "/analyzing_results/11252024/numi_cl" + yaml_file_name + ".vtp" # Function to load a VTK file

# gt_centerline_path = parent + "/GT_centerlines" + yaml_file_name + "_gt.vtp"
# # output_folder = parent + "/analyzing_results/11142024" + yaml_file_name + "/mine/"

# # this is for numi 
# output_folder = parent + "/analyzing_results/11142024" + yaml_file_name + "/numi/"
# gt_output_folder = parent + "/analyzing_results/11142024" + yaml_file_name + "/gt/"

# slice_interval = 5
# csv_file_path = output_folder + yaml_file_name + "_slice_data.csv"
# gt_csv_file_path = gt_output_folder + yaml_file_name + "_gt_slice_data.csv"

# combined_vtp_path = output_folder + yaml_file_name + "_combined_data.vtp"
# gt_combined_vtp_path = gt_output_folder + yaml_file_name + "_gt_combined_data.vtp"


mesh_path = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal_mine/analyzing_results/11252024/fine_distance_fields/0010_H_AO_H_FINE/eikonal/0010_H_AO_H_FINE_dis_map_p0_000000.vtu"
my_centerline_path = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal_mine/analyzing_results/11122024/0010_H_AO_H/mine/centerline/0010_H_AO_H_mine_centerline.vtp"

# this is for numi 
# my_centerline_path = parent + "/analyzing_results/11252024/numi_cl" + yaml_file_name + ".vtp" # Function to load a VTK file

gt_centerline_path = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal_mine/analyzing_results/11252024/GT_centerlines/0010_H_AO_H_gt.vtp"
# output_folder = parent + "/analyzing_results/11142024" + yaml_file_name + "/mine/"

# this is for numi 
output_folder = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal_mine/results/01102025/mine/"
gt_output_folder = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal_mine/results/01102025/gt/"

slice_interval = 5
csv_file_path = output_folder + "0010_H_AO_H_gt_slice_data.csv"
gt_csv_file_path = gt_output_folder + "0010_H_AO_H_gt_slice_data.csv"

combined_vtp_path = output_folder + "0010_H_AO_H_combined_data.vtp"
gt_combined_vtp_path = gt_output_folder + "0010_H_AO_H_gt_combined_data.vtp"

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


def save_2dslice(file_path, slice):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(slice)
    writer.SetFileName(file_path)
    writer.Write()


def write_csv(file_path, data):
    fieldnames = ["slice_number", "max_distance_value", "centerline_distance_value", "difference", "euclidean_distance", "average_difference", "std_dev_difference", "average_euclidean_distance", "std_dev_euclidean_distance"]
    with open(file_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


os.makedirs(output_folder, exist_ok=True)
os.makedirs(gt_output_folder, exist_ok=True)

# Load the centerline and vascular mesh
# centerline = load_vtk_file(centerline_path, vtk.vtkXMLPolyDataReader)
mine_centerline = load_vtk_file(my_centerline_path)
gt_centerline = load_vtk_file(gt_centerline_path)

# save_centerline_vtk(mine_centerline, output_folder + "centerline" + yaml_file_name + "_mine_centerline.vtp")
# save_centerline_vtk(gt_centerline, gt_output_folder + "centerline" + yaml_file_name + "_gt_centerline.vtp")

save_centerline_vtk(mine_centerline, output_folder + "centerline/mine_centerline.vtp")
save_centerline_vtk(gt_centerline, gt_output_folder + "centerline/gt_centerline.vtp")

def cl_distance_field(cl, dis_field_map):
    # Use vtkProbeFilter to interpolate distance values at centerline points
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInputData(cl)
    probe_filter.SetSourceData(dis_field_map)
    probe_filter.Update()
    distance_array = probe_filter.GetOutput().GetPointData().GetArray("f")
    return distance_array


# mesh = load_vtk_file(mesh_path, vtk.vtkXMLUnstructuredGridReader)
mesh = load_vtk_file(mesh_path)
# Initialize a slicer for the distance field and an append object for combining slices

append_filter = vtk.vtkAppendPolyData()  # To accumulate slices into one VTP file
gt_append_filter = vtk.vtkAppendPolyData()  # To accumulate slices into one VTP file

# Define clipping sphere radius
clip_radius = 0.3  # Adjust based on geometry scale
# Retrieve the distance field array from the mesh
point_locator = vtk.vtkPointLocator()
point_locator.SetDataSet(mesh)
point_locator.BuildLocator()
distance_field_array = mesh.GetPointData().GetArray("f")

# Initialize point locator for GT centerline
gt_point_locator = vtk.vtkPointLocator()
gt_point_locator.SetDataSet(gt_centerline)
gt_point_locator.BuildLocator()


# interpolated_distance_array = probe_filter.GetOutput().GetPointData().GetArray("f")
mine_cl_distance_array = cl_distance_field(mine_centerline,mesh)
gt_cl_distance_array = cl_distance_field(gt_centerline,mesh)


# Process and store results for each slice in a list
slice_data = []
gt_slice_data = []
num_points = mine_centerline.GetNumberOfPoints()
gt_num_points = gt_centerline.GetNumberOfPoints()
# Start slicing from point 20
# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5

def cut_plane(point, tangent_vector, mesh):
    cutter = vtk.vtkCutter()
    cutter.SetInputData(mesh)
    # Define slicing plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(point)
    plane.SetNormal(tangent_vector)
    
    # Set cutter to slice the mesh with the plane
    cutter.SetCutFunction(plane)
    cutter.Update()
    sliced_output = cutter.GetOutput()
    # Clip the slice around the centerline point
    clip_sphere = vtk.vtkSphere()
    clip_sphere.SetCenter(point)
    clip_sphere.SetRadius(clip_radius)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(sliced_output)
    clipper.SetClipFunction(clip_sphere)
    clipper.InsideOutOn()  # Keep only points inside the sphere
    clipper.Update()

    clipped_slice = clipper.GetOutput()
    return clipped_slice

# Inside the loop where you process each slice
for i in range(20, num_points, slice_interval):

    # find point in centerline (mine) and locate the next point
    current_point = mine_centerline.GetPoint(i)
    next_point = mine_centerline.GetPoint(i + 1) if i < num_points - 1 else mine_centerline.GetPoint(i - 1)
    # Calculate tangent vector and normalize
    tangent_vector = [next_point[j] - current_point[j] for j in range(3)]
    vtk.vtkMath.Normalize(tangent_vector)

    # find closest point to the other centerline you are comparing to and locate the next point
    closest_gt_point_id = gt_point_locator.FindClosestPoint(current_point)
    closest_gt_point = gt_centerline.GetPoint(closest_gt_point_id)
    print(closest_gt_point)
    gt_next_point = gt_centerline.GetPoint(closest_gt_point_id + 1) if closest_gt_point_id < gt_num_points - 1 else gt_centerline.GetPoint(closest_gt_point_id - 1)
    # Calculate tangent vector and normalize
    gt_tangent_vector = [gt_next_point[j] - closest_gt_point[j] for j in range(3)]
    vtk.vtkMath.Normalize(gt_tangent_vector)

    # find the 2D slice of the plane at those points in the orthogonal direction
    clipped_slice = cut_plane(current_point,tangent_vector,mesh)
    gt_clipped_slice = cut_plane(closest_gt_point,gt_tangent_vector,mesh)

    # Extract distance values from the clipped slice
    slice_distance_values = [
        distance_field_array.GetValue(point_locator.FindClosestPoint(clipped_slice.GetPoint(j)))
        for j in range(clipped_slice.GetNumberOfPoints())
    ]
    gt_slice_distance_values = [
    distance_field_array.GetValue(point_locator.FindClosestPoint(gt_clipped_slice.GetPoint(j)))
    for j in range(gt_clipped_slice.GetNumberOfPoints())
    ]

    # CENTERLINE: Extract interpolated distance value at the centerline point
    centerline_distance_value = mine_cl_distance_array.GetValue(i)
    gt_centerline_distance_value = gt_cl_distance_array.GetValue(closest_gt_point_id)
    # 2D Slice: Compute max distance in slice
    max_distance_value = max(slice_distance_values) if slice_distance_values else None
    gt_max_distance_value = max(gt_slice_distance_values) if gt_slice_distance_values else None


    # Find the coordinates of the point corresponding to max_distance_value
    if max_distance_value is not None:
        max_distance_point_id = slice_distance_values.index(max_distance_value)
        max_distance_point_coords = clipped_slice.GetPoint(max_distance_point_id)
    else:
        max_distance_point_coords = None

    # Calculate Euclidean distance between max_distance_point and centerline point
    if max_distance_point_coords is not None:
        euclidean_dist = euclidean_distance(max_distance_point_coords, current_point)
    else:
        euclidean_dist = None


    # Find the coordinates of the point corresponding to max_distance_value
    if gt_max_distance_value is not None:
        gt_max_distance_point_id = gt_slice_distance_values.index(gt_max_distance_value)
        gt_max_distance_point_coords = gt_clipped_slice.GetPoint(gt_max_distance_point_id)

    else:
        gt_max_distance_point_coords = None

    # Calculate Euclidean distance between max_distance_point and centerline point
    if gt_max_distance_point_coords is not None:
        gt_euclidean_dist = euclidean_distance(gt_max_distance_point_coords, closest_gt_point)
    else:
        gt_euclidean_dist = None
    
    
    # print(f"mine center coordinates: {max_distance_point_coords}")
    # print(f"gt center coordinates: {gt_max_distance_point_coords}")
    # print(f"mine cl coordinates: {current_point}")
    # print(f"gt cl coordinates: {closest_gt_point}")

    # print(f"mine center distance value: {max_distance_value}")
    # print(f"gt center distance value: {gt_max_distance_value}")
    # print(f"mine cl distance value: {centerline_distance_value}")
    # print(f"gt cl distance value: {gt_centerline_distance_value}")


    # Saving or appending
    # Save the localized slice for mine
    slice_file_path = os.path.join(output_folder, f"slice_{i}.vtp")
    gt_slice_file_path = os.path.join(gt_output_folder, f"slice_{i}.vtp")
    save_2dslice(slice_file_path,clipped_slice)
    save_2dslice(gt_slice_file_path,gt_clipped_slice)
    print(f"Saved localized slice {i} to {slice_file_path}")

    # Append slice to the combined VTP
    append_filter.AddInputData(clipped_slice)
    gt_append_filter.AddInputData(gt_clipped_slice)

    # Collect data for CSV
    slice_data.append({
        "slice_number": i,
        "max_distance_value": max_distance_value,
        "centerline_distance_value": centerline_distance_value,
        "difference": 0 if max_distance_value is not None and max_distance_value < centerline_distance_value else max_distance_value - centerline_distance_value if max_distance_value is not None else None,
        "euclidean_distance": euclidean_dist
        
    })
    gt_slice_data.append({
        "slice_number": i,
        "max_distance_value": gt_max_distance_value,
        "centerline_distance_value": gt_centerline_distance_value,
        "difference": 0 if gt_max_distance_value is not None and gt_max_distance_value < gt_centerline_distance_value else gt_max_distance_value - gt_centerline_distance_value if gt_max_distance_value is not None else None,
        "euclidean_distance": gt_euclidean_dist
        
    })

# Calculate the average and standard deviation of the differences and Euclidean distances
differences = [info["difference"] for info in slice_data if info["difference"] is not None]
average_difference = sum(differences) / len(differences) if differences else None
std_dev_difference = statistics.stdev(differences) if len(differences) > 1 else None

euclidean_distances = [info["euclidean_distance"] for info in slice_data if info["euclidean_distance"] is not None]
average_euclidean_distance = sum(euclidean_distances) / len(euclidean_distances) if euclidean_distances else None
std_dev_euclidean_distance = statistics.stdev(euclidean_distances) if len(euclidean_distances) > 1 else None

gt_differences = [info["difference"] for info in gt_slice_data if info["difference"] is not None]
gt_average_difference = sum(gt_differences) / len(gt_differences) if gt_differences else None
gt_std_dev_difference = statistics.stdev(gt_differences) if len(gt_differences) > 1 else None

gt_euclidean_distances = [info["euclidean_distance"] for info in gt_slice_data if info["euclidean_distance"] is not None]
gt_average_euclidean_distance = sum(gt_euclidean_distances) / len(gt_euclidean_distances) if gt_euclidean_distances else None
gt_std_dev_euclidean_distance = statistics.stdev(gt_euclidean_distances) if len(gt_euclidean_distances) > 1 else None

# Add the average and standard deviation to each row
for info in slice_data:
    info["average_difference"] = average_difference
    info["std_dev_difference"] = std_dev_difference
    info["average_euclidean_distance"] = average_euclidean_distance
    info["std_dev_euclidean_distance"] = std_dev_euclidean_distance

for info in gt_slice_data:
    info["average_difference"] = gt_average_difference
    info["std_dev_difference"] = gt_std_dev_difference
    info["average_euclidean_distance"] = gt_average_euclidean_distance
    info["std_dev_euclidean_distance"] = gt_std_dev_euclidean_distance

# Write the slice data to CSV files with the new columns
write_csv(csv_file_path, slice_data)
print(f"CSV data written to {csv_file_path}")
write_csv(gt_csv_file_path, gt_slice_data)
print(f"CSV data written to {gt_csv_file_path}")
print("All localized slices saved successfully.")

# Write the combined VTP file with all slices
append_filter.Update()
combined_writer = vtk.vtkXMLPolyDataWriter()
combined_writer.SetInputData(append_filter.GetOutput())
combined_writer.SetFileName(combined_vtp_path)
combined_writer.Write()

# Write the combined VTP file with all slices for GT
gt_append_filter.Update()
gt_combined_writer = vtk.vtkXMLPolyDataWriter()
gt_combined_writer.SetInputData(gt_append_filter.GetOutput())
gt_combined_writer.SetFileName(gt_combined_vtp_path)
gt_combined_writer.Write()

# Write the average differences and Euclidean distances to text files
# with open(output_folder + yaml_file_name + "_average_difference.txt", "w") as file:
#     file.write(f"Average Difference: {average_difference}\n")
#     file.write(f"Standard Deviation of Differences: {std_dev_difference}\n")
#     file.write(f"Average Euclidean Distance: {average_euclidean_distance}\n")
#     file.write(f"Standard Deviation of Euclidean Distances: {std_dev_euclidean_distance}\n")
#     file.close()

# with open(gt_output_folder + yaml_file_name + "_gt_average_difference.txt", "w") as file:
#     file.write(f"Average Difference: {gt_average_difference}\n")
#     file.write(f"Standard Deviation of Differences: {gt_std_dev_difference}\n")
#     file.write(f"Average Euclidean Distance: {gt_average_euclidean_distance}\n")
#     file.write(f"Standard Deviation of Euclidean Distances: {gt_std_dev_euclidean_distance}\n")
#     file.close()

with open(output_folder + "average_difference.txt", "w") as file:
    file.write(f"Average Difference: {average_difference}\n")
    file.write(f"Standard Deviation of Differences: {std_dev_difference}\n")
    file.write(f"Average Euclidean Distance: {average_euclidean_distance}\n")
    file.write(f"Standard Deviation of Euclidean Distances: {std_dev_euclidean_distance}\n")
    file.close()

with open(gt_output_folder + "gt_average_difference.txt", "w") as file:
    file.write(f"Average Difference: {gt_average_difference}\n")
    file.write(f"Standard Deviation of Differences: {gt_std_dev_difference}\n")
    file.write(f"Average Euclidean Distance: {gt_average_euclidean_distance}\n")
    file.write(f"Standard Deviation of Euclidean Distances: {gt_std_dev_euclidean_distance}\n")
    file.close()

print(f"Combined VTP file with all slices saved to {combined_vtp_path}")
print(f"Combined VTP file with all slices saved to {gt_combined_vtp_path}")

