import vtk


def remove_lines_from_vtp(input_vtp_path, output_vtp_path, cells_to_remove):
    """
    Remove specified lines (cells) from a VTP file and save the result to a new VTP file.

    Args:
        input_vtp_path (str): Path to the input VTP file.
        output_vtp_path (str): Path to save the output VTP file.
        cells_to_remove (list of int): List of cell IDs to remove from the VTP file.

    """

    # Read the input VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Use vtkIdFilter to add original cell IDs to the polydata
    id_filter = vtk.vtkIdFilter()
    id_filter.SetInputData(polydata)
    id_filter.SetCellIdsArrayName("vtkOriginalCellIds")
    id_filter.Update()

    # Convert cells_to_remove to a set for quick lookup
    cells_to_remove_set = set(cells_to_remove)

    # Create a new vtkPolyData to store the cells to keep
    cells_to_keep = vtk.vtkPolyData()
    cells_to_keep.DeepCopy(polydata)

    # Iterate over cells and keep those not in cells_to_remove
    cell_ids = vtk.vtkIdTypeArray()
    cell_ids.SetName("vtkOriginalCellIds")

    for i in range(cells_to_keep.GetNumberOfCells()):
        if i not in cells_to_remove_set:
            cell_ids.InsertNextValue(i)

    # Extract only the cells with IDs in cell_ids
    selection_node = vtk.vtkSelectionNode()
    selection_node.SetFieldType(vtk.vtkSelectionNode.CELL)
    selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
    selection_node.SetSelectionList(cell_ids)

    selection = vtk.vtkSelection()
    selection.AddNode(selection_node)

    extract_selection = vtk.vtkExtractSelection()
    extract_selection.SetInputData(0, cells_to_keep)
    extract_selection.SetInputData(1, selection)
    extract_selection.Update()

    # Convert the output to vtkPolyData
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(extract_selection.GetOutput())
    geometry_filter.Update()

    # Write the result to a new VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtp_path)
    writer.SetInputData(geometry_filter.GetOutput())
    writer.Write()


numbers = [83]


remove_lines_from_vtp("/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal_mine/results/02202025/summary_results_pointsource/0082_H_PULM_H/0082_H_PULM_H/centerlines/smooth_clean_radius_centerline.vtp","/Users/galasanchezvanmoer/Desktop/PhD_Project/centerline_analysis/mine_cl/0082_H_PULM_H.vtp",numbers)
