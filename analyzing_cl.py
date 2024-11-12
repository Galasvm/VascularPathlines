import vtk
import os
import csv
import yaml
from centerline_extraction import save_centerline_vtk

# Load parameters from YAML file
def load_params(yaml_file):
    parent = os.path.dirname(__file__)
    with open(os.path.join(parent, "yaml_files", yaml_file), "rb") as f:
        return yaml.safe_load(f)

# Load VTK file based on extension
def load_vtk_file(fname):
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError(f'File extension {ext} unknown.')
    reader.SetFileName(fname)
    reader.Update()
    return reader.GetOutput()

# Write slice data to CSV
def write_csv(file_path, data):
    with open(file_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["slice_number", "max_distance_value", "centerline_distance_value", "difference", "average_difference"])
        writer.writeheader()
        writer.writerows(data)

# Initialize paths and parameters
def initialize_paths_and_params(yaml_file, gt):
    parent = os.path.dirname(__file__)
    params = load_params(yaml_file)
    yaml_file_name = params["file_name"]

    if gt:
        centerline_path = os.path.join(parent, "GT_centerlines", f"{yaml_file_name}_gt.vtp")
        output_folder = os.path.join(parent, "analyzing_results", "11122024", yaml_file_name, "gt")
        slice_interval = 100
        csv_file_path = os.path.join(output_folder, f"{yaml_file_name}_gt_slice_data.csv")
        combined_vtp_path = os.path.join(output_folder, f"{yaml_file_name}_gt_combined_data.vtp")
    else:
        centerline_path = os.path.join(parent, "analyzing_results", "11122024", "mine_centerlines", yaml_file_name, "centerlines", f"{yaml_file_name}smooth_centerline.vtp")
        output_folder = os.path.join(parent, "analyzing_results", "11122024", yaml_file_name, "mine")
        slice_interval = 20
        csv_file_path = os.path.join(output_folder, f"{yaml_file_name}_mine_slice_data.csv")
        combined_vtp_path = os.path.join(output_folder, f"{yaml_file_name}_mine_combined_data.vtp")

    mesh_path = os.path.join(parent, "analyzing_results", "11122024", "fine_distance_fields", f"{yaml_file_name}_FINE", "eikonal", f"{yaml_file_name}_FINE_dis_map_p0_000000.vtu")
    os.makedirs(output_folder, exist_ok=True)

    return centerline_path, mesh_path, output_folder, slice_interval, csv_file_path, combined_vtp_path

# Process each slice and collect data
def process_slices(centerline, mesh, slice_interval, output_folder, distance_field_array, point_locator, interpolated_distance_array):
    cutter = vtk.vtkCutter()
    cutter.SetInputData(mesh)
    append_filter = vtk.vtkAppendPolyData()
    clip_radius = 0.3
    slice_data = []
    num_points = centerline.GetNumberOfPoints()

    for i in range(20, num_points, slice_interval):
        current_point = centerline.GetPoint(i)
        next_point = centerline.GetPoint(i + 1) if i < num_points - 1 else centerline.GetPoint(i - 1)
        tangent_vector = [next_point[j] - current_point[j] for j in range(3)]
        vtk.vtkMath.Normalize(tangent_vector)

        plane = vtk.vtkPlane()
        plane.SetOrigin(current_point)
        plane.SetNormal(tangent_vector)
        cutter.SetCutFunction(plane)
        cutter.Update()
        sliced_output = cutter.GetOutput()

        clip_sphere = vtk.vtkSphere()
        clip_sphere.SetCenter(current_point)
        clip_sphere.SetRadius(clip_radius)
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(sliced_output)
        clipper.SetClipFunction(clip_sphere)
        clipper.InsideOutOn()
        clipper.Update()
        clipped_slice = clipper.GetOutput()

        centerline_distance_value = interpolated_distance_array.GetValue(i)
        slice_distance_values = [
            distance_field_array.GetValue(point_locator.FindClosestPoint(clipped_slice.GetPoint(j)))
            for j in range(clipped_slice.GetNumberOfPoints())
        ]
        max_distance_value = max(slice_distance_values) if slice_distance_values else None

        slice_file_path = os.path.join(output_folder, f"slice_{i}.vtp")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(clipped_slice)
        writer.SetFileName(slice_file_path)
        writer.Write()
        print(f"Saved localized slice {i} to {slice_file_path}")

        append_filter.AddInputData(clipped_slice)
        slice_data.append({
            "slice_number": i,
            "max_distance_value": max_distance_value,
            "centerline_distance_value": centerline_distance_value,
            "difference": 0 if max_distance_value is not None and max_distance_value < centerline_distance_value else abs(max_distance_value - centerline_distance_value) if max_distance_value is not None else None
        })

    return slice_data, append_filter

# Calculate average difference and update slice data
def calculate_average_difference(slice_data):
    differences = [info["difference"] for info in slice_data if info["difference"] is not None]
    average_difference = sum(differences) / len(differences) if differences else None
    for info in slice_data:
        info["average_difference"] = average_difference
    return average_difference

# Main function to execute the analysis
def main():
    yaml_file = "0100_A_AO_COA.yaml"
    gt = False
    centerline_path, mesh_path, output_folder, slice_interval, csv_file_path, combined_vtp_path = initialize_paths_and_params(yaml_file, gt)

    centerline = load_vtk_file(centerline_path)
    save_centerline_vtk(centerline, os.path.join(output_folder, f"centerline_{yaml_file}_centerline.vtp"))
    mesh = load_vtk_file(mesh_path)

    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(mesh)
    point_locator.BuildLocator()
    distance_field_array = mesh.GetPointData().GetArray("f")

    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInputData(centerline)
    probe_filter.SetSourceData(mesh)
    probe_filter.Update()
    interpolated_distance_array = probe_filter.GetOutput().GetPointData().GetArray("f")

    slice_data, append_filter = process_slices(centerline, mesh, slice_interval, output_folder, distance_field_array, point_locator, interpolated_distance_array)
    average_difference = calculate_average_difference(slice_data)
    write_csv(csv_file_path, slice_data)
    print(f"CSV data written to {csv_file_path}")

    append_filter.Update()
    combined_writer = vtk.vtkXMLPolyDataWriter()
    combined_writer.SetInputData(append_filter.GetOutput())
    combined_writer.SetFileName(combined_vtp_path)
    combined_writer.Write()
    with open(csv_file_path + "_average_difference.txt", "w") as file:
        file.write(f"Average Difference: {average_difference}")
    print(f"Combined VTP file with all slices saved to {combined_vtp_path}")

if __name__ == "__main__":
    main()