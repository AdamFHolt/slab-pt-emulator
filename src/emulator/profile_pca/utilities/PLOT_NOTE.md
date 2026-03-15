# Path Plot Note

For talk/presentation consistency, the profile-PCA path preview utilities currently use fixed plot bounds:

- depth: `0` to `60 km`
- temperature: `-50` to `600 °C`

This applies to the left temperature-depth panel in:

- `compute_burial_path.py`
- `compute_many_burial_paths.py`
- `compute_burial_path_uncertain_parameter.py`

and the depth axis is fixed to `0–60 km` in the right-hand time-depth panels as well.
