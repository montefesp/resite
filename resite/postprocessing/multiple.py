from os import listdir
from os.path import isdir, join
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopy.distance

from resite.postprocessing.results import ResiteResults
from resite.resite import Resite


def compare_load_profile_to_cap_factor(r: Resite):

    # Normalize load profile
    load_required = r.data_dict["load"] * r.formulation_params  # TODO: this is shit
    load_required = load_required.sum(axis=1)
    norm_load_required = load_required - load_required.min()
    norm_load_required = norm_load_required/norm_load_required.max()

    cap_factor_df = r.sel_data_dict["cap_factor_df"]
    comparison = pd.Series(index=cap_factor_df.columns)
    for tech, lon, lat in cap_factor_df.columns:
        comparison[tech, lon, lat] = (norm_load_required - cap_factor_df[tech, lon, lat]).abs().mean()

    return comparison


def get_removed_points(r1: Resite, r2: Resite):
    removed_points = []
    for tech in r1.sel_tech_points_dict:
        points1 = r1.sel_tech_points_dict[tech]
        if tech in r2.sel_tech_points_dict:
            points2 = r2.sel_tech_points_dict[tech]
            removed_points += [(tech, point) for point in set(points1) - set(points2)]
        else:
            removed_points += [(tech, point) for point in points1]
    return removed_points


def get_added_points(r1: Resite, r2: Resite):
    added_points = []
    for tech in r2.sel_tech_points_dict:
        points2 = r2.sel_tech_points_dict[tech]
        if tech in r1.sel_tech_points_dict:
            points1 = r1.sel_tech_points_dict[tech]
            added_points += [(tech, point) for point in set(points2) - set(points1)]
        else:
            added_points += [(tech, point) for point in points2]
    return added_points


def get_saved_times_step_per_node(r: Resite):

    load_required = r.load_df * r.formulation_params
    load_required = load_required.sum(axis=1)

    generation = r.cap_factor_df*r.optimal_capacity_ds
    perc_non_covered_times = pd.Series(index=r.selected_capacity_potential_ds.index)
    perc_non_covered_load = pd.Series(index=r.selected_capacity_potential_ds.index)
    for tech_point in perc_non_covered_times.index:
        generation_without_point = generation.drop(tech_point, axis=1).sum(axis=1)
        perc_non_covered_times[tech_point] = np.mean(load_required > generation_without_point)
        residual_load = load_required - generation_without_point
        perc_non_covered_load[tech_point] = residual_load[residual_load > 0].sum()/load_required.sum()

    return perc_non_covered_times, perc_non_covered_load


def study_removed_and_added_points_influence(r1: Resite, r2: Resite):

    generation1 = r1.cap_factor_df*r1.optimal_capacity_ds
    generation2 = r2.cap_factor_df*r2.optimal_capacity_ds

    tech_point_tuples_r1 = [(tech, point) for tech, points in r1.sel_tech_points_dict.items() for point in points]
    tech_point_tuples_r2 = [(tech, point) for tech, points in r2.sel_tech_points_dict.items() for point in points]

    removed_points = get_removed_points(r1, r2)
    generation1_of_removed_points = r1.cap_factor_df[removed_points]*r1.optimal_capacity_ds.loc[removed_points]
    generation1_minus_removed_points = generation1.drop(removed_points, axis=1)

    added_points = get_added_points(r1, r2)
    generation2_of_added_points = r2.cap_factor_df[added_points]*r2.optimal_capacity_ds.loc[added_points]
    generation2_minus_added_points = generation2.drop(added_points, axis=1)

    load_required1 = r1.load_df * r1.formulation_params
    load_required1 = load_required1.sum(axis=1)
    load_required2 = r2.load_df * r2.formulation_params
    load_required2 = load_required2.sum(axis=1)

    generation1 = generation1.sum(axis=1)
    generation2 = generation2.sum(axis=1)
    generation1_of_removed_points = generation1_of_removed_points.sum(axis=1)
    generation1_minus_removed_points = generation1_minus_removed_points.sum(axis=1)
    generation2_of_added_points = generation2_of_added_points.sum(axis=1)
    generation2_minus_added_points = generation2_minus_added_points.sum(axis=1)

    print(f"Percentage of removed points: {len(removed_points)/len(tech_point_tuples_r1):.4f}")
    print(f"Percentage of unserved time-steps when removing points:"
          f"{np.mean(load_required1 > generation1_minus_removed_points):.4f}")
    print(f"Percentage of time-steps where additional load is covered by generation of added points:"
          f"{np.mean((load_required2-load_required1) < generation2_of_added_points):.4f}")
    print(f"Percentage of unserved time-steps before adding point:"
          f"{np.mean(load_required2 > generation2_minus_added_points):.4f}")
    print("")


def get_locations_average_distance_difference(r1: Resite, r2: Resite):

    assert r1.technologies == r2.technologies, "The two runs must be based on the same technologies"
    distances_ds = pd.DataFrame(columns=["Average distance", "Number of discarded"],
                                index=sorted(r1.sel_tech_points_dict))
    for tech in r1.technologies:

        # Get discarded and new points
        if tech in r2.technologies:
            discarded_points = list(set(r1.sel_tech_points_dict[tech]) - set(r2.sel_tech_points_dict[tech]))
            new_points = list(set(r2.sel_tech_points_dict[tech]) - set(r1.sel_tech_points_dict[tech]))
        else:
            continue

        distances_ds.loc[tech, "Number of discarded"] = len(discarded_points)

        if len(discarded_points) == 0:
            distances_ds.loc[tech, "Average distance"] = 0
            continue

        # Find closest new point for each discarded points and compute distance
        distances = pd.DataFrame(index=range(len(discarded_points)), columns=range(len(new_points)))
        for i, point in enumerate(discarded_points):
            distances.loc[i] = [geopy.distance.geodesic((point[1], point[0]), (new_point[1], new_point[0])).km for new_point in new_points]

        total_distance = 0
        distances = distances.values
        for i in range(len(discarded_points)):
            min_value = np.min(distances)
            total_distance += min_value
            min_index = np.where(distances == min_value)
            distances = np.delete(distances, min_index[0][0], 0)
            distances = np.delete(distances, min_index[1][0], 1)

        distances_ds.loc[tech, "Average distance"] = total_distance/len(discarded_points)

    return distances_ds


def get_locations_intersection(r1: Resite, r2: Resite):

    assert r1.technologies == r2.technologies, "The two runs must be based on the same technologies"
    intersection = pd.Series(0, index=sorted(r1.technologies), dtype=int)
    for tech in r1.technologies:
        if tech in r1.sel_tech_points_dict and tech in r2.sel_tech_points_dict:
            intersection[tech] = len(set(r1.sel_tech_points_dict[tech]) & set(r2.sel_tech_points_dict[tech]))
    return intersection


def get_locations_removal_of_capacity(r1: Resite, r2: Resite):

    assert r1.technologies == r2.technologies, "The two runs must be based on the same technologies"
    diff = r2.optimal_capacity_ds - r1.optimal_capacity_ds
    abs_diff = diff.abs()
    neg_diff = diff[diff < 0].abs()

    # Keep only points that have a positive capacity in r1 or r2
    tech_point_tuples_r1 = set([(tech, point) for tech, points in r1.sel_tech_points_dict.items() for point in points])
    tech_point_tuples_r2 = set([(tech, point) for tech, points in r2.sel_tech_points_dict.items() for point in points])
    tech_point_tuples = list(tech_point_tuples_r1 | tech_point_tuples_r2)
    abs_diff = abs_diff[tech_point_tuples]
    initial_cap = r1.optimal_capacity_ds[list(tech_point_tuples_r1)]
    print(f"Total removal of capacity: {neg_diff.sum()}")
    print(f"Total removal over initial capacity: {neg_diff.sum()/initial_cap.sum()}")
    print(f"Mean removal of capacity per points where capacity was removed: {neg_diff.mean()}")
    print(f"Std removal of capacity: {neg_diff.std()}")
    print(f"Max removel of capacity: {neg_diff.max()}\n")

    return neg_diff.sum()/initial_cap.sum()


def plot_load_vs_generation(r: Resite):

    generation = r.data_dict["cap_factor_df"]*r.data_dict["optimal_capacity_ds"]
    generation = generation.sum(axis=1)
    load_required = r.data_dict["load"] * r.formulation_params
    load_required = load_required.sum(axis=1)

    plt.figure()
    load_required.plot()
    generation.plot()
    plt.legend(["Load (GWh)", "Generation (GWh)"])


if __name__ == '__main__':

    # assert (len(sys.argv) == 2), "You need to provide one: output_dir"

    output_dir = "/home/utilisateur/Global_Grid/code/pyggrid/output/resite_EU_meet_RES_target_2016/"
    test_runs = sorted([l for l in listdir(output_dir) if isdir(join(output_dir, l))])
    print(test_runs)
    resite = []
    resite_results = []
    for run in test_runs:
        r = pickle.load(open(f"{output_dir}{run}/resite_instance.p", 'rb'))
        resite += [r]
        resite_results += [ResiteResults(r)]

    techs = ["pv_utility", "wind_offshore", "wind_onshore", "pv_residential"]

    if 0:
        # Show densities of capacity factors or generation
        for i in range(len(resite)):
            r = resite[i]
            sel_tech_points_tuples = r.sel_tech_points_tuples
            generation = r.sel_data_dict["cap_factor_df"] * r.sel_data_dict["cap_potential_ds"] * \
                r.y_ds[r.sel_tech_points_tuples]
            generation = generation.groupby(level=0, axis=1).sum(axis=1)
            print(generation)
            generation.plot(kind="density", title=test_runs[i])
        plt.show()
    if 0:
        # Compare load profiles to individual capacity factor profiles
        for i in range(1, len(resite)-1):
            print(i)
            comparison = compare_load_profile_to_cap_factor(resite[i])
            removed_points = get_removed_points(resite[i], resite[i+1])
            plt.figure()
            comparison.drop(removed_points).plot(kind="hist", bins=100)
            comparison[removed_points].plot(kind="hist", bins=100)
            plt.legend(["Kept points", "Removed points"])
            plt.show()
    if 1:
        # Look at how much capacity potential is used
        percentage_of_maximum_cap_sites = pd.DataFrame(columns=test_runs, index=techs + ["Total"], dtype=float)
        average_cap_per = pd.DataFrame(columns=test_runs, index=techs, dtype=float)
        cap_per_point_avg = pd.DataFrame(columns=test_runs, index=techs, dtype=float)
        for i, r in enumerate(resite):
            # plt.figure()
            test_run = test_runs[i]
            cap_perc_per_node = r.y_ds[r.sel_tech_points_tuples]
            percentage_of_maximum_cap_sites.loc["Total", test_run] = \
                len(cap_perc_per_node[cap_perc_per_node == 1])/len(cap_perc_per_node)
            average_cap_per.loc["Total", test_run] = cap_perc_per_node.mean()
            for i, tech in enumerate(techs):
                if tech in cap_perc_per_node.index:
                    percentage_of_maximum_cap_sites.loc[tech, test_run] = \
                        len(cap_perc_per_node.loc[tech][cap_perc_per_node.loc[tech] == 1]) / len(cap_perc_per_node.loc[tech])
                    average_cap_per.loc[tech, test_run] = cap_perc_per_node.loc[tech].mean()

                if tech in cap_perc_per_node.index:
                    """
                    plt.subplot(1, len(techs), i+1)
                    cap_perc_per_node.loc[tech].plot(kind="hist", bins=50)
                    plt.xlim([0., 1.])
                    plt.title(tech)
                    """
            # plt.show()
        percentage_of_maximum_cap_sites.T.plot(title="Percentage of sites with maximum capacity deployed", ylim=(0, 1))
        plt.savefig(f"{output_dir}perc_points_with_max_cap.pdf")
        print(f"Average percentage of potential capacity installed:\n{average_cap_per.round(3)}")
        average_cap_per.T.plot(title="Average percentage of potential capacity installed", ylim=(0, 1))
        plt.savefig(f"{output_dir}avg_perc_of_pot_cap.pdf")
        #print(f"Percentage of points where there is capacity:\n{perc_points_used.round(3)}")
        #perc_points_used.T.plot(title="Percentage of points where there is capacity")
        #plt.savefig(f"{output_dir}perc_points_with_cap.pdf")
    if 0:
        # Analyse how removed points affect the generation signal
        for i in range(1, len(resite)-1):
            print(i)
            perc_lost_times, perc_lost_load = get_saved_times_step_per_node(resite[i])
            removed_points = get_removed_points(resite[i], resite[i+1])
            plt.figure()
            perc_lost_times.drop(removed_points).plot(kind="hist", bins=100)
            perc_lost_times[removed_points].plot(kind="hist", bins=100)
            plt.legend(["Kept points", "Removed points"])
            plt.figure()
            perc_lost_load.drop(removed_points).plot(kind="hist", bins=100)
            perc_lost_load[removed_points].plot(kind="hist", bins=100)
            plt.legend(["Kept points", "Removed points"])
            plt.show()
    if 0:
        # Identify the impact of removed and added points to the signal
        for i in range(1, len(resite)-1):
            study_removed_and_added_points_influence(resite[i], resite[i+1])
            plt.show()
    if 0:
        # Analyse the capacity that is removed between changes of load requirement
        portion_of_cap_removed = pd.Series(index=test_runs[1:-1])
        for i in range(1, len(resite)-1):
            print(f"Comparison between: {test_runs[i]} and {test_runs[i+1]}")
            portion_of_cap_removed.loc[test_runs[i]] = get_locations_removal_of_capacity(resite[i], resite[i+1])
        portion_of_cap_removed.plot(kind="bar", use_index=False)
        plt.title("Portion of capacity that is removed from on load requirement to the next")
        plt.show()
    if 0:
        # Plot load vs generation
        for r in resite[1:]:
            plot_load_vs_generation(r)
        plt.show()
    if 0:
        # Look at how many points are discarded between each run
        discarded_points_perc = pd.DataFrame(index=test_runs, columns=test_runs, dtype=float)
        discarded_points_perc_principal = pd.Series(index=test_runs, dtype=float)
        for i in range(1, len(resite)):
            print(test_runs[i])
            number_points = resite_results[i].get_selected_points_number()
            number_points_sum = number_points.sum()
            for j in range(i, len(resite)):
                intersection = get_locations_intersection(resite[i], resite[j])
                intersection_sum = intersection.sum()
                discarded_points_perc.loc[test_runs[i], test_runs[j]] = \
                    (number_points_sum - intersection_sum) / number_points_sum
                if j == i+1:
                    discarded_points_perc_principal[i] = (number_points_sum - intersection_sum) / number_points_sum
                    print(number_points - intersection)

        discarded_points_perc = discarded_points_perc.round(2)
        print(f"Percentage of discarded points\n: {discarded_points_perc}")
        plt.figure()
        discarded_points_perc_principal[1:].plot(kind="bar")
        # plt.plot(test_runs[1:], [discarded_points_perc_principal[1:].mean()]*len(test_runs[1:]))
        plt.show()
    if 0:
        # Compute a metric showing how much points move from one value of load requirement to another - a bit shitty
        for i in range(1, len(resite)-1):
            print(f"\nFrom {test_runs[i]} to {test_runs[i+1]}")
            distances = get_locations_average_distance_difference(resite[i], resite[i+1])
            print(distances)
    if 1:
        # Compute points number statistics
        points_numbers = pd.DataFrame(columns=test_runs, index=techs + ["Total"], dtype=int)
        perc_points_used = pd.DataFrame(columns=test_runs, index=techs + ["Total"], dtype=float)
        for i, rr in enumerate(resite_results):
            selected_points_number = rr.get_selected_points_number()
            selected_points_number.loc["Total"] = selected_points_number.sum()
            initial_points_number = rr.get_initial_points_number()
            initial_points_number.loc["Total"] = initial_points_number.sum()
            points_numbers[test_runs[i]] = selected_points_number
            perc_points_used[test_runs[i]] = selected_points_number/initial_points_number
        points_numbers = points_numbers.fillna(0)
        points_numbers.T.plot(title="Points Number")
        plt.savefig(f"{output_dir}points_number.pdf")
        points_percentage = points_numbers/points_numbers.loc["Total"]
        points_percentage.T.plot(title="Points Percentage per Tech")
        plt.savefig(f"{output_dir}points_perc.pdf")
        perc_points_used.T.plot(title="Percentage of points that were selected")
        plt.savefig(f"{output_dir}perc_points_selected.pdf")
    if 0:
        # Capacity potential statistics
        initial_cap_potential_per_tech = 0
        initial_cap_potential_sum = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        initial_cap_potential_mean = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        initial_cap_potential_std = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        selected_cap_potential_sum = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        selected_cap_potential_mean = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        selected_cap_potential_std = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        for i, rr in enumerate(resite_results):
            initial_cap_potential_sum[test_runs[i]] = rr.get_initial_capacity_potential_sum()
            initial_cap_potential_mean[test_runs[i]] = rr.get_initial_capacity_potential_mean()
            initial_cap_potential_std[test_runs[i]] = rr.get_initial_capacity_potential_std()
            selected_cap_potential_sum[test_runs[i]] = rr.get_selected_capacity_potential_sum()
            selected_cap_potential_mean[test_runs[i]] = rr.get_selected_capacity_potential_mean()
            selected_cap_potential_std[test_runs[i]] = rr.get_selected_capacity_potential_std()
            if i == 0:
                initial_cap_potential_per_tech = resite[i].data_dict["cap_potential_ds"]
                plt.figure()
                initial_cap_potential_per_tech.unstack(level=0)\
                    .plot(title="Density of capacity potentials of initial points", kind="density", alpha=0.8)
                plt.savefig(f"{output_dir}capacity_potentials_density_initial.pdf")
                plt.figure()
                initial_cap_potential_per_tech.unstack(level=0)\
                    .plot(title="Histogram of capacity potentials of initial points", kind="hist", bins=100, alpha=0.5)
                plt.savefig(f"{output_dir}capacity_potentials_hist_initial.pdf")

        # initial_cap_potential_sum.loc["Total"] = initial_cap_potential_sum.sum(axis=0)
        selected_cap_potential_sum.loc["Total"] = selected_cap_potential_sum.sum(axis=0)
        initial_cap_potential_sum.index = [f"init {tech}" for tech in initial_cap_potential_sum.index]

        plt.figure()
        ax = selected_cap_potential_sum.loc[techs].T.plot(title="Selected vs Initial Potential Capacities (GW)")
        initial_cap_potential_sum.T.plot(ax=ax, ls="--", color=['C0', 'C1', 'C2'])
        plt.savefig(f"{output_dir}capacity_potentials_sum_selected_vs_initial.pdf")

        plt.figure()
        selected_cap_potential_sum.T.plot(title="Selected Potential Capacities (GW)")
        plt.savefig(f"{output_dir}capacity_potentials_sum_selected.pdf")

        plt.figure()
        selected_cap_potential_mean.T.plot(title="Selected Potential Capacities (GW)")
        plt.savefig(f"{output_dir}capacity_potentials_mean_selected.pdf")

        plt.figure()
        selected_cap_potential_std.T.plot(title="Selected Potential Capacities (GW)")
        plt.savefig(f"{output_dir}capacity_potentials_std_selected.pdf")

    if 0:
        # Capacity factors statistics
        selected_cap_factors_mean = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        selected_cap_factors_std = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        initial_cap_factors_mean = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        initial_cap_factors_std = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        unselected_cap_factors_mean = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        unselected_cap_factors_std = pd.DataFrame(index=techs, columns=test_runs, dtype=float)
        total_cap_factors_mean = pd.DataFrame(index=["Initial", "Selected"], columns=test_runs, dtype=float)
        for i, rr in enumerate(resite_results):
            selected_cap_factors_mean[test_runs[i]] = rr.get_selected_cap_factor_mean()
            selected_cap_factors_std[test_runs[i]] = rr.get_selected_cap_factor_std()
            initial_cap_factors_mean[test_runs[i]] = rr.get_initial_cap_factor_mean()
            initial_cap_factors_std[test_runs[i]] = rr.get_initial_cap_factor_std()
            unselected_cap_factors_mean[test_runs[i]] = rr.get_unselected_cap_factor_mean()
            unselected_cap_factors_std[test_runs[i]] = rr.get_unselected_cap_factor_std()
            total_cap_factors_mean.loc["Initial", test_runs[i]] = resite[i].cap_factor_df.mean().mean()
            total_cap_factors_mean.loc["Selected", test_runs[i]] = resite[i].selected_cap_factor_df.mean().mean()

        initial_cap_factors_mean.index = [f"init {tech}" for tech in initial_cap_factors_mean.index]
        initial_cap_factors_std.index = [f"init {tech}" for tech in initial_cap_factors_std.index]

        plt.figure()
        initial_cap_factors_mean.T.plot(title="Initial points mean of capacity factors mean")
        plt.ylim([0, 0.5])
        plt.grid(axis='y')
        plt.savefig(f"{output_dir}capacity_factors_mean_initial.pdf")

        plt.figure()
        initial_cap_factors_std.T.plot(title="Initial points mean of capacity factors std")
        plt.ylim([0.2, 0.3])
        plt.grid(axis='y')
        plt.savefig(f"{output_dir}capacity_factors_std_initial.pdf")

        plt.figure()
        ax = selected_cap_factors_mean.T.plot(title="Selected vs Initial points mean of capacity factors mean")
        initial_cap_factors_mean.T.plot(ax=ax, ls='--', color=['C0', 'C1', 'C2'])
        plt.ylim([0, 0.5])
        plt.grid(axis='y')
        plt.savefig(f"{output_dir}capacity_factors_mean_selected.pdf")

        plt.figure()
        selected_cap_factors_std.T.plot(title="Selected points mean of capacity factors std")
        plt.ylim([0.2, 0.3])
        plt.grid(axis='y')
        plt.savefig(f"{output_dir}capacity_factors_std_selected.pdf")

        plt.figure()
        ax = unselected_cap_factors_mean.T.plot(title="Unselected vs Initial points mean of capacity factors mean")
        initial_cap_factors_mean.T.plot(ax=ax, ls='--', color=['C0', 'C1', 'C2'])
        plt.ylim([0, 0.5])
        plt.grid(axis='y')
        plt.savefig(f"{output_dir}capacity_factors_mean_unselected.pdf")

        plt.figure()
        unselected_cap_factors_std.T.plot(title="Unselected points mean of capacity factors std")
        plt.ylim([0.2, 0.3])
        plt.grid(axis='y')
        plt.savefig(f"{output_dir}capacity_factors_std_unselected.pdf")

        plt.figure()
        total_cap_factors_mean.T.plot(title="Total points mean of capacity factors mean")
        plt.savefig(f"{output_dir}capacity_factors_mean_total_selected.pdf")
    if 1:
        # Produce a series of plots of main characteristics of the signals of selected points:
        capacities = pd.DataFrame(columns=test_runs, dtype=int, index=techs)
        generation = pd.DataFrame(columns=test_runs, dtype=int, index=techs)
        load = pd.Series(index=test_runs, dtype=int, name='Load')
        max_capacities = pd.DataFrame(columns=test_runs, dtype=int, index=techs)
        for i, rr in enumerate(resite_results):

            capacities[test_runs[i]] = rr.get_optimal_capacity()
            max_capacities[test_runs[i]] = rr.get_initial_capacity_potential_sum()
            generation[test_runs[i]] = rr.get_generation()
            load[test_runs[i]] = resite[i].data_dict["load"].sum().sum()
            # cap_factors_mean[test_runs[i]] = rr.get_selected_cap_factor_mean()
            # cap_factors_std[test_runs[i]] = rr.get_selected_cap_factor_std()

        max_capacities.index = [f"{tech}-max" for tech in max_capacities.index]
        capacities = capacities.fillna(0)
        generation = generation.fillna(0)

        capacities.loc["Total"] = capacities.sum(axis=0)
        generation.loc["Total"] = generation.sum(axis=0)

        ax = capacities.T.plot(title="Optimal Capacities (GW)")
        # max_capacities.T.plot(ax=ax, ls="--", color=[f'C{i}' for i in range(len(techs))])
        plt.savefig(f"{output_dir}optimal_capacities.pdf")
        generation.T.plot(title="Generation (GWh)")
        load.plot()
        plt.savefig(f"{output_dir}generation.pdf")

