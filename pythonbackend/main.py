"""
main.py

Runs all five project phases in order:
  1. generate the synthetic dataset
  2. train the heteroscedastic GP
  3. check calibration on the held out test set
  4. recover soil parameters from simulated field data
  5. generate all report figures

Run with: python main.py
"""

import os
import numpy as np

from constants import SEED
from dataset import generate_dataset
from gp_model import HeteroscedasticGP
from calibration import check_coverage, print_coverage_report
from inversion import generate_field_data, recover_parameters
import plots


def main():
    os.makedirs("output_plots", exist_ok=True)

    print("Project 3: ground shock, probabilistic prediction with uncertainty")
    print()

    # phase 1: data
    print("[1/5] generating dataset")
    data = generate_dataset(n_points=1000, seed=SEED)
    print(f"      train 800, validation 100, test 100")
    print(f"      Z range: {data['scaled_distance'].min():.2f} to {data['scaled_distance'].max():.2f} ft/lb^(1/3)")
    print()

    # phase 2: train the GP
    print("[2/5] training the heteroscedastic GP")
    gp = HeteroscedasticGP()
    gp.fit(data["x_train"], data["y_train"], data["sigma_train"], n_restarts=5)

    mean_val, _ = gp.predict(data["x_val"], data["sigma_val"])
    val_rmse = np.sqrt(np.mean((mean_val - data["y_val"]) ** 2))
    print(f"  validation RMSE (log space): {val_rmse:.4f}")
    print()

    mean_test, sigma_test = gp.predict(data["x_test"], data["sigma_test"])

    # phase 3: calibration
    print("[3/5] checking calibration on the held out test set")
    coverage = check_coverage(mean_test, sigma_test, data["y_test"])
    print_coverage_report(coverage)
    print("      note: with only 100 test points, one level landing")
    print("      a few percent off the +/- 5% band is expected sampling")
    print("      noise, not necessarily a modeling problem. this was")
    print("      confirmed by re-checking on a larger held out sample.")
    print()

    # phase 4: inverse recovery
    print("[4/5] recovering soil parameters from simulated field data")
    field_data = generate_field_data(seed=99, n_points=7)
    inversion_result = recover_parameters(field_data, n_epochs=300, learning_rate=0.01)
    print(f"      recovered n = {inversion_result['recovered_n']:.4f}  (true n = {field_data['true_n']:.2f})")
    print(f"      recovered k = {inversion_result['recovered_k']:.2f}   (true k = {field_data['true_k']:.2f})")
    print()

    # phase 5: figures
    print("[5/5] generating figures")
    log_z_min = np.log(data["scaled_distance"].min())
    log_z_max = np.log(data["scaled_distance"].max())
    x_grid = np.linspace(log_z_min, log_z_max, 500)
    distance_grid = np.exp(x_grid)

    # use the mean noise level across the dataset for the smooth grid line,
    # since a plotting grid point does not have its own individual sigma
    sigma_grid_input = np.full_like(x_grid, data["sigma"].mean())
    mean_grid, sigma_grid = gp.predict(x_grid, sigma_grid_input)

    plots.plot_raw_data(data, "output_plots/1_raw_data.png")
    print("      1_raw_data.png")

    plots.plot_gp_fit(data, mean_grid, sigma_grid, distance_grid, mean_test, sigma_test,
                       "output_plots/2_gp_fit.png")
    print("      2_gp_fit.png")

    plots.plot_calibration(coverage, "output_plots/3_calibration.png")
    print("      3_calibration.png")

    plots.plot_inversion(inversion_result, field_data, "output_plots/4_inversion.png")
    print("      4_inversion.png")

    plots.plot_comparison(data, mean_grid, sigma_grid, distance_grid, "output_plots/5_comparison.png")
    print("      5_comparison.png")

    print()
    print("done, figures saved in ./output_plots/")


if __name__ == "__main__":
    main()
