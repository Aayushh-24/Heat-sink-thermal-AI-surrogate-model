import numpy as np
import pandas as pd
from heat_sink_model import heat_sink_design


def generate_parameter_grid(tdp_range, v_air_range, k_tim_range, n_points=20):
    """
    Build a full factorial grid of parameter combinations.

    Parameters
    ----------
    tdp_range, v_air_range, k_tim_range : tuple(min, max)
    n_points : int, number of levels per parameter

    Returns
    -------
    np.ndarray of shape (n_points**3, 3) -> columns: [TDP, V_air, k_tim]
    """
    tdp_values = np.linspace(tdp_range[0], tdp_range[1], n_points)
    v_air_values = np.linspace(v_air_range[0], v_air_range[1], n_points)
    k_tim_values = np.linspace(k_tim_range[0], k_tim_range[1], n_points)

    # meshgrid + reshape builds every combination of the three parameter lists
    grid = np.array(np.meshgrid(tdp_values, v_air_values, k_tim_values)).T.reshape(-1, 3)
    return grid


def run_parameter_sweep(grid):
    """
    Call heat_sink_design() once per row of the grid and collect the
    full physics output (inputs + outputs) into a DataFrame.
    """
    records = []
    for tdp, v_air, k_tim in grid:
        result = heat_sink_design(TDP=tdp, V_air=v_air, k_tim=k_tim)
        record = {"TDP": tdp, "V_air": v_air, "k_tim": k_tim, **result}
        records.append(record)
    return pd.DataFrame(records)


if __name__ == "__main__":
    grid = generate_parameter_grid(
        tdp_range=(30, 250),      # per assessment spec
        v_air_range=(0.5, 15),    # per assessment spec
        k_tim_range=(1, 12),      # per assessment spec
        n_points=20
    )
    print(f"Total parameter combinations: {len(grid)}")

    df = run_parameter_sweep(grid)

    # Save dataset
    output_path = "../data/heat_sink_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(df.head())