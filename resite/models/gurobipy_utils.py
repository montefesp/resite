from gurobipy import GRB, quicksum, LinExpr

import pandas as pd
import numpy as np


def create_generation_y_dict(y, regions, tech_points_regions_ds, generation_potential_df):

    region_generation_y_dict = dict.fromkeys(regions)
    for region in regions:
        # Get generation potential for points in region for each techno
        region_tech_points = tech_points_regions_ds[tech_points_regions_ds == region].index
        tech_points_generation_potential = generation_potential_df[region_tech_points]
        region_ys = pd.Series([y[tech, lon, lat] for tech, lon, lat in region_tech_points],
                              index=region_tech_points)
        region_generation = tech_points_generation_potential.values*region_ys.values
        region_generation_y_dict[region] = np.sum(region_generation, axis=1)

    return region_generation_y_dict


def generation_bigger_than_load_proportion_with_slack(model, region_generation_y_dict, ens, load, regions, time_slices,
                                           load_perc_per_region):
    model.addConstrs((sum(region_generation_y_dict[region][t] for t in time_slices[u]) +
                      sum(ens[region, t] for t in time_slices[u]) >=
                      sum(load[t, regions.index(region)] for t in time_slices[u]) * load_perc_per_region[region]
                      for region in regions for u in np.arange(len(time_slices))),
                     name='generation_bigger_than_load_proportion')


def generation_bigger_than_load_proportion(model, region_generation_y_dict, load, regions, time_slices,
                                           load_perc_per_region):
    model.addConstrs((sum(region_generation_y_dict[region][t] for t in time_slices[u]) >=
                      sum(load[t, regions.index(region)] for t in time_slices[u]) * load_perc_per_region[region]
                      for region in regions for u in np.arange(len(time_slices))),
                     name='generation_bigger_than_load_proportion')


def generation_bigger_than_load_x(model, x, region_generation_y_dict, load, regions, timestamps_idxs):
    model.addConstrs(((region_generation_y_dict[region][t] >= load[t, regions.index(region)] * x[region, t])
                      for region in regions for t in timestamps_idxs),
                     name='generation_bigger_than_load_x')


def capacity_bigger_than_existing(model, y, existing_cap_percentage_ds, tech_points_tuples):
    model.addConstrs(((y[tech, lon, lat] >= existing_cap_percentage_ds[tech, lon, lat])
                      for (tech, lon, lat) in tech_points_tuples),
                     name='capacity_bigger_than_existing')


def tech_cap_bigger_than_limit(model, y, cap_potential_ds, tech_points_dict, technologies, required_cap_per_tech):
    model.addConstrs(((sum(y[tech, lon, lat] * cap_potential_ds[tech, lon, lat] for lon, lat in tech_points_dict[tech])
                       >= required_cap_per_tech[tech])
                      for tech in technologies),
                     name='tech_cap_bigger_than_limit')


def minimize_deployed_capacity(model, y, cap_potential_ds):
    obj = sum(y[tech, lon, lat] * cap_potential_ds[tech, lon, lat] for tech, lon, lat in cap_potential_ds.keys())
    model.setObjective(obj, GRB.MINIMIZE)
    return obj


def maximize_load_proportion(model, x, regions, timestamps_idxs):
    obj = sum(x[region, t] for region in regions for t in timestamps_idxs)
    model.setObjective(obj, GRB.MAXIMIZE)
    return obj


def minimize_cost(model, y, ens, cap_potential_ds, regions, timestamps_idx, cost_dict):
    obj = sum(y[tech, lon, lat] * cap_potential_ds[tech, lon, lat] * cost_dict[tech]
              for tech, lon, lat in cap_potential_ds.keys()) + \
          sum(ens[region, t] * cost_dict['ens'] for region in regions for t in timestamps_idx)
    model.setObjective(obj, GRB.MINIMIZE)
    return obj


def minimize_total_cost_old(model, y, p, ens, cap_potential_ds, regions, timestamps_idx, costs_df):

    capex = sum(y[tech, lon, lat] * cap_potential_ds[tech, lon, lat] * costs_df.loc[tech, "capital"]
                for tech, lon, lat in cap_potential_ds.keys())
    opex = sum(p[tech, lon, lat, t] * costs_df.loc[tech, "marginal"]
               for tech, lon, lat in cap_potential_ds.keys() for t in timestamps_idx)
    ens = sum(ens[region, t] * costs_df.loc['ens', 'marginal'] for region in regions for t in timestamps_idx)
    obj = capex + opex + ens

    model.setObjective(obj, GRB.MINIMIZE)

    return obj


def minimize_total_cost(model, y, p, ens, capacity_potential_ds, cost_df,
                        tech_points_tuples, regions, int_timestamps):

    capacity_potential_dict = capacity_potential_ds.to_dict()
    cost_dict = cost_df.to_dict(orient='index')

    cost_capacity_dict = dict.fromkeys(capacity_potential_dict.keys())
    for (tech, lon, lat) in cost_capacity_dict:
        cost_capacity_dict[(tech, lon, lat)] = capacity_potential_dict[(tech, lon, lat)] * cost_dict[tech]['capital']

    ens_cost = cost_dict['ens']['marginal']

    obj = quicksum(cost_capacity_dict[(tech, lon, lat)] * y[tech, lon, lat]
                   for (tech, lon, lat) in tech_points_tuples) + \
        quicksum(cost_dict[tech]['marginal'] * p[tech, lon, lat, t]
                 for (tech, lon, lat) in tech_points_tuples for t in int_timestamps) + \
        quicksum(ens_cost * ens[region, t] for region in regions for t in int_timestamps)

    model.setObjective(obj, GRB.MINIMIZE)

    return obj


def infeed_lower_than_potential(model, p, y, cap_factor_df, cap_potential_ds, tech_points_tuples, int_timestamps):

    generation_potential_df = cap_factor_df * cap_potential_ds
    generation_potential_dict = generation_potential_df.reset_index().to_dict()
    for (tech, lon, lat) in tech_points_tuples:
        for t in int_timestamps:
            rhs = LinExpr(generation_potential_dict[(tech, lon, lat)][t], y[tech, lon, lat])
            model.addLConstr(p[tech, lon, lat, t] <= rhs, f"infeed_lower_than_potential_{tech}_{lon}_{lat}_{t}")


def supply_bigger_than_demand_regional(model, p, ens, regions, tech_points_regions_ds, load,
                              int_timestamps, time_slices, covered_load_perc_per_region):
    # Compute a sum of generation per time-step per region
    region_p_dict = dict.fromkeys(regions)
    for region in regions:
        # Get generation potential for points in region for each techno
        region_tech_points = tech_points_regions_ds[tech_points_regions_ds == region].index
        region_p_sum = pd.Series([sum(p[tech, lon, lat, t] for tech, lon, lat in region_tech_points)
                                  for t in int_timestamps], index=int_timestamps)
        region_p_dict[region] = region_p_sum

    # Impose a certain percentage of the load to be covered over each time slice
    model.addConstrs((sum(region_p_dict[region][t] for t in time_slices[u]) +
                      sum(ens[region, t] for t in time_slices[u]) >=
                      sum(load[t, regions.index(region)] for t in time_slices[u]) * covered_load_perc_per_region[region]
                      for region in regions for u in np.arange(len(time_slices))),
                     name='generation_bigger_than_load_proportion')


def supply_bigger_than_demand_global(model, p, ens, regions, tech_points_regions_ds, load,
                                     int_timestamps, time_slices, covered_load_perc_global):
    # Compute a sum of generation per time-step per region
    region_p_dict = dict.fromkeys(regions)
    for region in regions:
        # Get generation potential for points in region for each techno
        region_tech_points = tech_points_regions_ds[tech_points_regions_ds == region].index
        region_p_sum = pd.Series([sum(p[tech, lon, lat, t] for tech, lon, lat in region_tech_points)
                                  for t in int_timestamps], index=int_timestamps)
        region_p_dict[region] = region_p_sum

    # Over all regions
    model.addConstrs(((sum(region_p_dict[region][t] for t in time_slices[u] for region in regions) +
                       sum(ens[region, t] for t in time_slices[u] for region in regions) >=
                       sum(load[t, regions.index(region)] for t in time_slices[u] for region in regions)
                       * covered_load_perc_global)
                       for u in np.arange(len(time_slices))), name='generation_check_global')

