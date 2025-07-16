# Automatic Vascular Pathline Extraction

This framework provides an interactive workflow to extract centerlines (pathlines) from 3D vascular models. It can run fully automatically or pause for simple user steps (point‑source selection or postprocessing).

---

## Dependencies

1. **FEniCSx (>= 0.9)**  
   Used to solve the Eikonal equation on a volumetric mesh. Instructions for installation can be found [here](https://fenicsproject.org/download/)

2. **Python packages**  
    ```bash
    pip install meshio scipy h5py
    ```

3. **fTetWild (optional, but required for surface‑mesh extraction)**  

    Used to create a volumetric meshs from triangle surface meshes (`.vtp`, `.stl`).  
    Build/install instructions can be found [here](https://github.com/wildmeshing/fTetWild)

    After installing fTetWild, add its binary to your environment:

    ```bash
    export FTETWILD_PATH="/full/path/to/FloatTetwild_bin"
    ```

    The script will first look for `$FTETWILD_PATH`, then fall back to `which FloatTetwild_bin`.

---

## Usage

```bash
python3 extracting_cl/model_to_mesh.py <models_dir> <save_dir>
````

* `<models_dir>`
  Directory of input files. Supported formats:

  * triangle surface meshes: `.vtp` / `.stl`
  * tetrahedral volumetric meshes: `.xdmf`

* `<save_dir>`
  Directory where extracted centerlines will be saved.

### Optional arguments

* `-p,--pointsource <bool>`
  Select inlet point source:

  * `False` (default): detect inlet automatically
  * `True`: launch an interactive relector to allow user to select the inlet

* `-r,--remove_extra_centerlines`
  Launch an interactive selector to remove unwanted centerlines after extraction.
