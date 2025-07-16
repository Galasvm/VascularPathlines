# Automatic Vascular Pathline Extraction

This framework provides an interactive workflow to extract centerlines (pathlines) from 3D vascular models. It can run fully automatically or pause for simple user steps (point‑source selection or postprocessing).

---

## Installation

1. Clone this repo  
   ```bash
   git clone https://github.com/Galasvm/VascularPathlines
   cd VascularPathlines
   ```
2. Create and activate your environment
    ```bash
    # conda
    conda create -n pathline
    conda activate pathline
    ```
3. **FEniCSx (>= 0.9)**  
   This framework relies on [FEniCSx](https://fenicsproject.org/) to solve the Eikonal equation on a volumetric mesh. Instructions for installation can be found [here](https://fenicsproject.org/download/)

    ```bash
    conda install -c conda-forge fenics-dolfinx mpich pyvista
    ```
4. **Python packages**  
    ```bash
    pip install meshio scipy h5py
    ```
3. **fTetWild (optional, but required for surface‑mesh extraction)**  

    This framework relies on [fTetWild](https://github.com/wildmeshing/fTetWild) to create a volumetric meshes from triangle surface meshes (`.vtp`, `.stl`).  
    Build/install instructions can be found [here](https://github.com/wildmeshing/fTetWild)

    After installing fTetWild, set it as a conda-env variable:

    ```bash
    conda activate myenv
    conda env config vars set FTETWILD_PATH="/full/path/to/FloatTetwild_bin"
    ```
    
    OR add its binary to your environment:

    ```bash
    export FTETWILD_PATH="/full/path/to/FloatTetwild_bin"
    ```

    The script will first look for `$FTETWILD_PATH`, then fall back to `which FloatTetwild_bin`.

## Usage

```bash
python3 extracting_cl/tracingcenterlines.py <models_dir> <save_dir>
```

* `<models_dir>`
  Directory of input files. Supported formats:

  * triangle surface meshes: `.vtp` / `.stl` (requires fTetWild)
  * tetrahedral volumetric meshes: `.xdmf`

* `<save_dir>`
  Directory where extracted centerlines will be saved.

### Optional arguments

* `-p,--pointsource <bool>`
  User selects inlet point source:

  * `False` (default): detect inlet point source automatically
  * `True`: launch an interactive selector to allow user to select the inlet point source

* `-r,--remove_extra_centerlines`
  Launch an interactive selector to remove unwanted centerlines after extraction.

### Examples

* **Extraction with automatic inlet detection**
```bash
python extracting_cl/tracingcenterlines.py test_cases/vol_mesh test_cases/results/AutomaticDetection
````

* **Allow user to select point source**
```bash
python extracting_cl/tracingcenterlines.py test_cases/vol_mesh test_cases/results/UserSelected -p True
````

* **Interactive cleanup of extra centerlines (if necessary)**
```bash
python extracting_cl/tracingcenterlines.py test_cases/vol_mesh test_cases/results/UserSelected -p True -r
````
