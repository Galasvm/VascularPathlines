import vtk
import os
import csv
from extracting_cl.tracingcenterlines import save_centerline_vtk
import statistics


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
    fieldnames = ["mine_point_id", "gt_point_id", "distance_points", "percent_difference","average_euclidean_distance", "std_dev_euclidean_distance","average_percent_difference"]
    with open(file_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def cl_distance_field(cl, dis_field_map):
    # Use vtkProbeFilter to interpolate distance values at centerline points
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInputData(cl)
    probe_filter.SetSourceData(dis_field_map)
    probe_filter.Update()
    distance_array = probe_filter.GetOutput().GetPointData().GetArray("f")
    return distance_array


# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5


def cut_plane(point, radius, tangent_vector, mesh):
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
    clip_sphere.SetRadius(radius)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(sliced_output)
    clipper.SetClipFunction(clip_sphere)
    clipper.InsideOutOn()  # Keep only points inside the sphere
    clipper.Update()

    clipped_slice = clipper.GetOutput()
    return clipped_slice


def slice_computedistance(mine_centerline, gt_centerline, slice_interval, output_folder, model_name):

    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(gt_centerline)
    point_locator.BuildLocator()

    # Process and store results for each slice in a list
    centerline_info = []

    num_points = mine_centerline.GetNumberOfPoints()

    radius_gt_array = gt_centerline.GetPointData().GetArray("MaximumInscribedSphereRadius")

    # Inside the loop where you process each slice
    for i in range(1, num_points, slice_interval):

        # find point in centerline (mine) and locate the next point
        current_point = mine_centerline.GetPoint(i)
        closest_point_id = point_locator.FindClosestPoint(current_point)
        closest_gt_point = gt_centerline.GetPoint(closest_point_id)

        point_distance = euclidean_distance(current_point, closest_gt_point)
        radius_value = radius_gt_array.GetValue(closest_point_id)

        percent_difference = point_distance / (2*radius_value) * 100

        # Collect data for CSV
        centerline_info.append({
            "mine_point_id": i,
            "gt_point_id": closest_point_id,
            "distance_points": point_distance,
            "percent_difference": percent_difference
        })

        print("done with slice", i)

    euclidean_distances = [info["distance_points"] for info in centerline_info if info["distance_points"] is not None]
    percent_differences = [info["percent_difference"] for info in centerline_info if info["percent_difference"] is not None]
    average_euclidean_distance = sum(euclidean_distances) / len(euclidean_distances) if euclidean_distances else None
    average_percent_difference = sum(percent_differences) / len(percent_differences) if percent_differences else None
    std_dev_euclidean_distance = statistics.stdev(euclidean_distances) if len(euclidean_distances) > 1 else None

    # Add the average and standard deviation to each row
    for info in centerline_info:
        info["average_euclidean_distance"] = average_euclidean_distance
        info["std_dev_euclidean_distance"] = std_dev_euclidean_distance
        info["average_percent_difference"] = average_percent_difference


    # Write the slice data to CSV files with the new columns
    write_csv(output_folder +"/"+model_name+"_point_data.csv", centerline_info)


    print("All localized slices saved successfully.")

    # Write the average differences and Euclidean distances to text files
    with open(output_folder +"/"+model_name+"_average_difference.txt", "w") as file:
        # file.write(f"Average Difference: {average_difference}\n")
        # file.write(f"Standard Deviation of Differences: {std_dev_difference}\n")
        file.write(f"Average Euclidean Distance: {average_euclidean_distance}\n")
        file.write(f"Standard Deviation of Euclidean Distances: {std_dev_euclidean_distance}\n")
        file.write(f"Average Percent Difference: {average_percent_difference}\n")
        file.close()



def main(model_name, mine_centerline_path, vmtk_centerline_path, output_folder):
    mine_centerline = load_vtk_file(mine_centerline_path)
    gt_centerline = load_vtk_file(vmtk_centerline_path)
    # Apply spline filter to resample the lines
    # spline_filter = vtk.vtkSplineFilter()
    # spline_filter.SetInputData(gt_centerline)

    # # Set the number of subdivisions: this controls how many points will be added
    # # For example, if you want ~3x more points, try something like:
    # spline_filter.SetSubdivideToSpecified()
    # spline_filter.SetNumberOfSubdivisions(gt_centerline.GetNumberOfPoints() * 3)

    # spline_filter.Update()

    # # Get the output
    # resampled_mine_centerline = spline_filter.GetOutput()


    save_centerline_vtk(mine_centerline, output_folder + "/centerline/" + model_name + "_mine_centerline.vtp")
    save_centerline_vtk(gt_centerline, output_folder + "/centerline/" + model_name + "_vmtk_centerline.vtp")

    slice_interval = 1
    slice_computedistance(mine_centerline, gt_centerline, slice_interval, output_folder, model_name)


import glob
def process_directory(directory, base_save_dir):
    # Find all model files in the specified directory
    mine_centerline_directory = directory + "/mine_cl"

    vtp_files = glob.glob(os.path.join(mine_centerline_directory, "*.vtp"))  # Adjust the pattern as needed

    for mine_cl in vtp_files:
        # Extract the model name without the extension
        model_name = os.path.splitext(os.path.basename(mine_cl))[0]
        vmtk_cl_path = directory + "/vmtk_cl/" + model_name + "_centerline.vtp"

        # Create a subdirectory for each model
        # model_save_dir = os.path.join(base_save_dir, model_name)
        model_save_dir = base_save_dir+"/"+model_name
        os.makedirs(model_save_dir, exist_ok=True)

        print(f"Processing model: {mine_cl}")
        main(model_name, mine_cl, vmtk_cl_path, model_save_dir)

if __name__ == "__main__":
    # Hardcode the folder and save directory
    folder_directory = "/Users/galasanchezvanmoer/PhD_project2/centerline_analysis"
    base_save_directory = "/Users/galasanchezvanmoer/PhD_project2/centerlines/results/03142025/analyzing_cl/newlines"

    process_directory(folder_directory, base_save_directory)