# Copyright (C) 2026 KiharaLab, Purdue University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
import os
import shutil

import pandas as pd
import typer

app = typer.Typer(help="Select best results from CryoZeta and CryoZeta-Interpolate")


@app.command()
def main(
    dump_dir: str = typer.Option(
        ...,
        help="Base output directory (contains per-entry subdirectories)",
    ),
    input_json_path: str = typer.Option(..., help="Path to input JSON file"),
    num_select: int = typer.Option(
        ..., help="Number of top results to select per entry"
    ),
    seeds: str = typer.Option(..., help="Seed value used during inference"),
):
    """Select the best results from CryoZeta and CryoZeta-Interpolate models.

    Reads scores from ``{dump_dir}/{name}/CryoZeta/saved_data/scores.csv`` and
    ``{dump_dir}/{name}/CryoZeta-Interpolate/saved_data/scores.csv``, then
    copies the top results to ``{dump_dir}/{name}/CryoZeta-Final/``.
    """
    with open(input_json_path) as f:
        data = json.load(f)
    pdb_list = [item["name"] for item in data]

    # for each pdb_id, select the best num_select results from the two models
    # combined, based on recall_ccmask_ca, and save the results to the output_dir
    for pdb_id in pdb_list:
        cryozeta_dir = f"{dump_dir}/{pdb_id}/CryoZeta"
        interpolate_dir = f"{dump_dir}/{pdb_id}/CryoZeta-Interpolate"
        combined_dir = f"{dump_dir}/{pdb_id}/CryoZeta-Final"

        scores_path = f"{cryozeta_dir}/saved_data/scores.csv"
        scores_interp_path = f"{interpolate_dir}/saved_data/scores.csv"

        if not os.path.exists(scores_path) or not os.path.exists(scores_interp_path):
            typer.echo(
                f"Skipping {pdb_id}: scores.csv not found in one or both model directories"
            )
            continue

        df_scores = pd.read_csv(scores_path)
        df_scores_interpolation = pd.read_csv(scores_interp_path)

        results = []
        for _index, row in df_scores.iterrows():
            if row["pdb_id"] == pdb_id:
                results.append(
                    (
                        row["sample_idx"],
                        row["method"],
                        row["recall_ccmask_ca"],
                        "model1",
                    )
                )
        for _index, row in df_scores_interpolation.iterrows():
            if row["pdb_id"] == pdb_id:
                results.append(
                    (
                        row["sample_idx"],
                        row["method"],
                        row["recall_ccmask_ca"],
                        "model_interpolation",
                    )
                )
        if len(results) == 0:
            continue
        results = sorted(results, key=lambda x: x[2], reverse=True)
        results = results[:num_select]
        os.makedirs(combined_dir, exist_ok=True)
        for i in range(len(results)):
            result = results[i]
            if result[3] == "model1":
                file_path = f"{cryozeta_dir}/seed_{seeds}/predictions_{result[1]}/{pdb_id}_sample_{result[0]}.cif"
            else:
                file_path = f"{interpolate_dir}/seed_{seeds}/predictions_{result[1]}/{pdb_id}_sample_{result[0]}.cif"
            shutil.copy(file_path, f"{combined_dir}/{pdb_id}_sample_{i}.cif")

    typer.echo(f"Ensemble selection completed. Results saved to {dump_dir}")


if __name__ == "__main__":
    app()
