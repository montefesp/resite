"""
Formulation Description:

 This formulation aims at selecting sites such that their deployment cost is minimized under i) an energy
 balance constraint and ii) capacity bounds

 - Parameters:
    - time_resolution: The average production over each time step of this resolution must be
    greater than the average load over this same time step.
    Can currently be 'hour', 'day', 'week', 'month' and 'full' (average over the full time range)
    - perc_per_region (region): Percentage of the load that must be satisfied at each time step.

 - Variables:
    - y (tech, lon, lat): Portion of capacity potential selected at each site.
    - ens (region, t): Energy-not-served slack in the balance equation
    - p (tech, lon, lat, t): Production at each time step

 - Objective: Minimize deployed capacity

 - Constraints:
    - load requirement: generation_t,r + ens_t,r >= load_t,r * perc_per_region_r for all r,t
    - existing capacity: y * potential capacity >= existing capacity; at each site
    - limit generation: generation must be smaller than available power
"""

from typing import Dict
from os.path import join
from iepy.technologies.costs import get_costs
from iepy import data_path

import numpy as np
import pandas as pd


def build_model(resite, modelling: str, params: Dict):
    """
    Model build-up.

    Parameters:
    ------------
    modelling: str
        Name of the modelling language to use.
    params: List[float]
        List of parameters needed by the formulation
    """
    accepted_modelling = ["docplex", "gurobipy", "pyomo"]
    assert modelling in accepted_modelling, f"Error: This formulation was not coded with modelling language {modelling}"

    assert 'perc_per_region' in params and len(params['perc_per_region']) == len(resite.regions), \
        "Error: This formulation requires a vector of required RES penetration per region."
    accepted_resolutions = ["hour", "day", "week", "month", "full"]
    assert "time_resolution" in params and params["time_resolution"] in accepted_resolutions, \
        f"Error: This formulation requires a time resolution chosen among {accepted_resolutions}," \
        f" got {params['time_resolution']}"

    build_model_ = globals()[f"build_model_{modelling}"]
    build_model_(resite, params)


def define_time_slices(time_resolution: str, timestamps):
    timestamps_idxs = np.arange(len(timestamps))
    if time_resolution == 'day':
        time_slices = [list(timestamps_idxs[timestamps.dayofyear == day]) for day in timestamps.dayofyear.unique()]
    elif time_resolution == 'week':
        time_slices = [list(timestamps_idxs[timestamps.weekofyear == week]) for week in
                       timestamps.weekofyear.unique()]
    elif time_resolution == 'month':
        time_slices = [list(timestamps_idxs[timestamps.month == mon]) for mon in timestamps.month.unique()]
    elif time_resolution == 'hour':
        time_slices = [[t] for t in timestamps_idxs]
    else:  # time_resolution == 'full':
        time_slices = [timestamps_idxs]

    return time_slices


def get_cost_df(techs: list, timestamps):
    costs_df = pd.DataFrame(index=techs + ["ens"], columns=["capital", "marginal"])

    for tech in techs:
        capital_cost, marginal_cost = get_costs(tech, len(timestamps))
        costs_df.loc[tech, "capital"] = capital_cost
        costs_df.loc[tech, "marginal"] = marginal_cost

    tech_dir = f"{data_path}technologies/"
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)

    costs_df.loc["ens", "marginal"] = fuel_info.loc["load", "cost"]

    return costs_df


def build_model_pyomo(resite, params: Dict):
    """Model build-up using pyomo"""

    from pyomo.environ import ConcreteModel, NonNegativeReals, Var
    from resite.models.pyomo_utils import capacity_bigger_than_existing, minimize_total_cost

    data = resite.data_dict
    load = data["load"].values
    regions = resite.regions
    technologies = resite.technologies
    tech_points_tuples = list(resite.tech_points_tuples)
    time_slices = define_time_slices(params["time_resolution"], resite.timestamps)
    generation_potential_df = data["cap_factor_df"] * data["cap_potential_ds"]

    costs_df = get_cost_df(technologies, resite.timestamps)

    model = ConcreteModel()

    # - Parameters - #
    covered_load_perc_per_region = dict(zip(regions, params["perc_per_region"]))

    # - Variables - #
    # Energy not served
    model.ens = Var(list(regions), np.arange(len(resite.timestamps)), within=NonNegativeReals)

    # Portion of capacity at each location for each technology
    model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))

    # Generation at each time step
    model.p = Var(tech_points_tuples, np.arange(len(resite.timestamps)), within=NonNegativeReals)

    # - Constraints - #
    # Generation limited by generation potential
    from pyomo.environ import Constraint

    def generation_limit(model, tech, lon, lat, t):
        return model.p[tech, lon, lat, t] <= model.y[tech, lon, lat] * generation_potential_df.iloc[t][(tech, lon, lat)]
    model.generation_limit = Constraint(tech_points_tuples, np.arange(len(resite.timestamps)),
                                        rule=generation_limit)

    # Create generation dictionary for building speed up
    # Compute a sum of generation per time-step per region
    region_p_dict = dict.fromkeys(regions)
    for region in regions:
        # Get generation potential for points in region for each techno
        region_tech_points = resite.tech_points_regions_ds[resite.tech_points_regions_ds == region].index
        region_p_sum = pd.Series([sum(model.p[tech, lon, lat, t] for tech, lon, lat in region_tech_points)
                                 for t in np.arange(len(resite.timestamps))],
                                 index=np.arange(len(resite.timestamps)))
        region_p_dict[region] = region_p_sum

    # Impose a certain percentage of the load to be covered over each time slice
    def generation_check_rule(model, region, u):
        return sum(region_p_dict[region][t] for t in time_slices[u]) + \
            sum(model.ens[region, t] for t in time_slices[u]) >= \
            sum(load[t, regions.index(region)] for t in time_slices[u]) * covered_load_perc_per_region[region]
    model.generation_check = Constraint(regions, np.arange(len(time_slices)), rule=generation_check_rule)

    # Percentage of capacity installed must be bigger than existing percentage
    existing_cap_percentage_ds = data["existing_cap_ds"].divide(data["cap_potential_ds"])
    model.potential_constraint = capacity_bigger_than_existing(model, existing_cap_percentage_ds, tech_points_tuples)

    # - Objective - #
    # Minimize the capacity that is deployed
    model.objective = minimize_total_cost(model, data["cap_potential_ds"], regions,
                                          np.arange(len(resite.timestamps)), costs_df)

    resite.instance = model
