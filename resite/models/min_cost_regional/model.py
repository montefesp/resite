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

    accepted_resolutions = ["hour", "day", "week", "month", "full"]
    assert "time_resolution" in params and params["time_resolution"] in accepted_resolutions, \
        f"This formulation requires a time resolution chosen among {accepted_resolutions}," \
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


def build_model_gurobipy(resite, params: Dict):
    """Model build-up using gurobipy"""

    from gurobipy import Model
    from resite.models.gurobipy_utils import minimize_total_cost, \
        capacity_bigger_than_existing, infeed_lower_than_potential, supply_bigger_than_demand_regional
    import itertools as it

    data = resite.data_dict
    load = data["load"].values
    regions = resite.regions
    technologies = resite.technologies
    tech_points_tuples = list(resite.tech_points_tuples)
    time_slices = define_time_slices(params["time_resolution"], resite.timestamps)
    tech_points_regions_ds = resite.tech_points_regions_ds
    int_timestamps = np.arange(len(resite.timestamps))

    costs_df = get_cost_df(technologies, resite.timestamps)
    multipliers_dict = params['multiplier_per_region']
    if isinstance(multipliers_dict, float):
        multipliers_per_region = pd.Series(multipliers_dict, index=resite.regions)
    else:
        multipliers_per_region = pd.Series(multipliers_dict)

    model = Model()

    # - Parameters - #
    remote_regions = set(resite.regions).difference(set(multipliers_dict.keys()))

    if remote_regions:
        # For "remote" regions, the \xi value is set to 0.
        missing_regions_df = pd.Series(0., index=remote_regions)
        perc_per_region_df = pd.concat([multipliers_per_region, missing_regions_df]).loc[resite.regions]
    else:
        perc_per_region_df = multipliers_per_region.loc[resite.regions]

    assert len(perc_per_region_df.index) == len(resite.regions), \
        f"number of percentages ({len(perc_per_region_df.index)}) " \
        f"must be equal to number of regions ({len(resite.regions)})."

    # - Variables - #
    # Energy not served
    ens_tuples = it.product(regions, int_timestamps)
    ens = model.addVars(ens_tuples, lb=0., name=lambda k: 'ens_%s_%s' % (k[0], k[1]))

    # Portion of capacity at each location for each technology
    y = model.addVars(tech_points_tuples, lb=0., ub=1., name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))

    # Generation at each time step
    p = model.addVars(tech_points_tuples,  int_timestamps,
                      lb=0., name=lambda k: 'p_%s_%s_%s_%s' % (k[0], k[1], k[2], k[3]))

    # - Constraints - #
    # Generation limited by generation potential
    infeed_lower_than_potential(model, p, y, data["cap_factor_df"], data["cap_potential_ds"],
                                tech_points_tuples, int_timestamps)

    # Impose a certain percentage of the load to be covered over each time slice
    # On a regional basis
    supply_bigger_than_demand_regional(model, p, ens, regions, tech_points_regions_ds, load,
                                       int_timestamps, time_slices, perc_per_region_df)

    # Percentage of capacity installed must be bigger than existing percentage
    existing_cap_percentage_ds = data["existing_cap_ds"].divide(data["cap_potential_ds"])
    capacity_bigger_than_existing(model, y, existing_cap_percentage_ds, tech_points_tuples)

    # - Objective - #
    # Minimize the capacity that is deployed
    obj = minimize_total_cost(model, y, p, ens, data["cap_potential_ds"], costs_df,
                              tech_points_tuples, regions, int_timestamps)

    resite.instance = model
    resite.y = y
    resite.obj = obj


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
