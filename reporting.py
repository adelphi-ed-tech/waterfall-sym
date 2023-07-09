import pandas as pd
from settings import *
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Read the CSV file into a DataFrame
project_df = pd.read_csv('project_data.csv')
resources_df = pd.read_csv('resources_report.csv')
phases_df = pd.read_csv('phases_report.csv')
failures_df = pd.read_csv('failures_report.csv')
optimization_df = pd.read_csv('optimization.csv')

project_data = []
resources_data = []
phases_data = []
failures_data = []
optimization_data = []


def main():
    table6_figure_2_report_on_convergence()
    table7_count_bottle_neck_pre_optimization()
    table8_optimized_resources_report()
    table9_report_on_project_completion_times()
    table10_report_on_phase_completion_times_by_phase()
    table11_part1_report_on_wait_time()
    table11_part2_report_on_wait_time()
    table11_part3_report_on_wait_time()
    table11_part4_report_on_wait_time()
    table12_part1_failure_report()
    table12_part2_failure_report()
    table12_part3_failure_report()
    table12_part4_failure_report()
    table12_figure4_failure_report()
    figure3()


def figure3():
    # Filter the resources_df based on the specified criteria
    filtered_resources = resources_df[(resources_df['Stage'] == 'pre-optimization') &
                                      (resources_df['Iteration'] == 1)]

    # Get the unique resource types
    resource_types = filtered_resources['Resource Type'].unique()

    # Create a separate graph for each resource type
    for resource_type in resource_types:
        # Filter the data for the current resource type
        resource_data = filtered_resources[filtered_resources['Resource Type'] == resource_type]

        # Create a step plot of the resources available
        plt.figure(figsize=(10, 6))
        plt.step(resource_data['Timestamp'], resource_data['Resources Available'], where='post')
        plt.xlabel('Time')
        plt.ylabel('Resources Available')
        plt.title(f'Resources Available for {resource_type.capitalize()} (Pre-Optimization, Iteration 1)')
        plt.xticks(rotation=45)

        # Set x-axis limits to 400
        plt.xlim(0, 400)

        plt.show()


def table12_figure4_failure_report():
    # Filter failures_df and phases_df to exclude 'optimizing' stage
    filtered_failures_df = failures_df[failures_df['Stage'] != 'optimizing']
    filtered_phases_df = phases_df[phases_df['Stage'] != 'optimizing']

    # Create a table for phase failure counts
    failure_counts = filtered_failures_df.groupby(['Stage', 'Project Scale']).size().reset_index(
        name='Phase Failure Count')

    # Create a table for total phase counts
    phase_counts = filtered_phases_df.groupby(['Stage', 'Project Scale']).size().reset_index(
        name='Total Phase Count')

    # Merge the failure and phase counts tables
    report_table = pd.merge(failure_counts, phase_counts, on=['Stage', 'Project Scale'], how='outer')

    # Fill missing values with zeros
    report_table['Phase Failure Count'].fillna(0, inplace=True)

    # Calculate the percentage of phase failures
    report_table['Percentage'] = (report_table['Phase Failure Count'] / report_table['Total Phase Count']) * 100

    # Add a column for the difference between total phases and phase failures
    report_table['Diff'] = report_table['Total Phase Count'] - report_table['Phase Failure Count']

    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Display the report table
    display(report_table)

    # Group the report table by 'Stage' and 'Project Scale'
    grouped_table = report_table.groupby(['Stage', 'Project Scale']).sum().reset_index()

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Create the grouped bar chart
    sns.set(style='whitegrid')

    # Get unique 'Project Scale' values
    unique_scales = grouped_table['Project Scale'].unique()

    # Set the width of each bar
    bar_width = 0.2  # Adjust the value for thinner bars

    # Set the x positions of the bars
    x = np.arange(len(grouped_table['Stage'].unique()))

    # Choose a colorblind-friendly palette
    colors = sns.color_palette('colorblind', n_colors=len(unique_scales))

    # Iterate over unique project scales and create the bars
    for i, scale in enumerate(unique_scales):
        scale_data = grouped_table[grouped_table['Project Scale'] == scale]
        phase_failure = scale_data['Phase Failure Count']
        diff = scale_data['Diff']
        plt.bar(x + i * bar_width, phase_failure, width=bar_width, label=f'{scale} - Phase Failure', color=colors[i])
        plt.bar(x + i * bar_width, diff, width=bar_width, label=f'{scale} - Diff', bottom=phase_failure,
                color=colors[i], alpha=0.7)

    # Set the x-axis tick labels
    plt.xticks(x + bar_width * len(unique_scales) / 2, grouped_table['Stage'].unique())

    # Set the title and labels
    plt.title('Failure Report')
    plt.xlabel('Stage')
    plt.ylabel('Count')

    # Display the legend
    plt.legend(title='Project Scale')

    # Show the plot
    plt.show()


def table12_part4_failure_report():
    # Create a table for phase failures and their corresponding counts
    failures_limited_df = failures_df[failures_df['Stage'] != 'optimizing']

    # Group phase failures by stage
    stage_failures_table = failures_limited_df.groupby('Stage')['Iteration'].count().reset_index()

    # Calculate the total phases for each stage
    stage_counts = phases_df['Stage'].value_counts().reset_index()
    stage_counts.columns = ['Stage', 'Total Phases']

    # Merge phase failures table with total phase counts
    stage_failures_table = stage_failures_table.merge(stage_counts, on='Stage', how='inner')

    # Calculate the percentage of phase failures for each stage
    stage_failures_table['Percentage'] = (stage_failures_table['Iteration'] / stage_failures_table[
        'Total Phases']) * 100

    # Print phase failures table by stage
    print("Phase Failures Table by Stage:")
    print(stage_failures_table)
    stage_failures_table.to_csv("table12_part4_failure_report.csv", index=False, mode='w')


def table12_part3_failure_report():
    # Create the table of phase failures
    phase_failures_table = pd.pivot_table(failures_df, index=['Stage', 'Current Phase'], values='Iteration',
                                          aggfunc='count')
    phase_failures_table.reset_index(inplace=True)
    phase_failures_table.columns = ['Stage', 'Phase', 'Number of Failures']

    # Calculate the total number of corresponding phases
    total_phases_table = pd.pivot_table(phases_df, index=['Stage', 'Phase'], values='Iteration', aggfunc='count')
    total_phases_table.reset_index(inplace=True)
    total_phases_table.columns = ['Stage', 'Phase', 'Total Phases']

    # Merge the two tables
    phase_stats_table = pd.merge(phase_failures_table, total_phases_table, on=['Stage', 'Phase'], how='left')

    # Calculate the percentage of phase failures
    phase_stats_table['Percentage'] = (phase_stats_table['Number of Failures'] / phase_stats_table[
        'Total Phases']) * 100

    # Filter out the rows with stage 'optimizing'
    phase_stats_table = phase_stats_table[phase_stats_table['Stage'] != 'optimizing']

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    display(phase_stats_table)
    phase_stats_table.to_csv("table12_part3_failure_report.csv", index=False, mode='w')


def table12_part2_failure_report():
    # Filter failures_df and phases_df to exclude 'optimizing' stage
    filtered_failures_df = failures_df[failures_df['Stage'] != 'optimizing']
    filtered_phases_df = phases_df[phases_df['Stage'] != 'optimizing']

    # Create a table for phase failure counts
    failure_counts = filtered_failures_df.groupby(['Stage', 'Project Scale']).size().reset_index(name='Phase Failure Count')

    # Create a table for total phase counts
    phase_counts = filtered_phases_df.groupby(['Stage', 'Project Scale']).size().reset_index(name='Total Phase Count')

    # Merge the failure and phase counts tables
    report_table = pd.merge(failure_counts, phase_counts, on=['Stage', 'Project Scale'], how='outer')

    # Fill missing values with zeros
    report_table['Phase Failure Count'].fillna(0, inplace=True)

    # Calculate the percentage of phase failures
    report_table['Percentage'] = (report_table['Phase Failure Count'] / report_table['Total Phase Count']) * 100

    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Display the report table
    display(report_table)
    report_table.to_csv("table12_part2_failure_report.csv", index=False, mode='w')


def table12_part1_failure_report():
    # Create a table of phase failures and total phases by stage, project scale, and phase
    failure_counts = failures_df.groupby(['Stage', 'Project Scale', 'Current Phase']).size().reset_index(
        name='Phase Failures')
    total_phases = phases_df.groupby(['Stage', 'Project Scale', 'Phase']).size().reset_index(name='Total Phases')

    # Merge the failure counts and total phases tables
    phase_failure_table = failure_counts.merge(total_phases, left_on=['Stage', 'Project Scale', 'Current Phase'],
                                               right_on=['Stage', 'Project Scale', 'Phase'], how='left').fillna(0)

    # Filter out rows with stage = "optimizing"
    phase_failure_table = phase_failure_table[phase_failure_table['Stage'] != 'optimizing']

    # Add Percentage column
    phase_failure_table['Percentage'] = (phase_failure_table['Phase Failures'] / phase_failure_table[
        'Total Phases']) * 100

    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Print the phase failure table
    print("Phase Failure Table:")
    display(phase_failure_table)
    phase_failure_table.to_csv("table12_part1_failure_report.csv", index=False, mode='w')


def table12_part3():
    # Filter failures by excluding the stage "optimizing"
    failures_filtered_df = failures_df[failures_df["Stage"] != "optimizing"]

    # Generate pre-optimization and post-optimization failure summaries
    pre_optimization_summary = failures_filtered_df[failures_df["Iteration"] == 1]
    post_optimization_summary = failures_filtered_df[failures_df["Iteration"] > 1]

    # Display pre-optimization failure summary
    print("Pre-Optimization Failure Summary:")
    pre_optimization_failure_summary = pre_optimization_summary.groupby(['Project Scale', 'Stage', 'Current Phase'])[
        'Project Id'].count().reset_index()
    display(pre_optimization_failure_summary)

    # Display post-optimization failure summary
    print("Post-Optimization Failure Summary:")
    post_optimization_failure_summary = post_optimization_summary.groupby(['Project Scale', 'Stage', 'Current Phase'])[
        'Project Id'].count().reset_index()
    display(post_optimization_failure_summary)

    # Compute the percentage of phases that fail
    total_phases = len(phases_df)
    failed_phases = len(failures_df)
    failure_percentage = (failed_phases / total_phases) * 100
    print(f"Failure Percentage: {failure_percentage:.2f}%")

    # Create pre-optimization failure summary plots by project scale
    for project_scale in pre_optimization_failure_summary['Project Scale'].unique():
        project_scale_data = pre_optimization_failure_summary[
            pre_optimization_failure_summary['Project Scale'] == project_scale]
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Current Phase', y='Project Id', hue='Stage', data=project_scale_data)
        plt.title(f"Pre-Optimization Failure Summary - Project Scale: {project_scale}")
        plt.xlabel("Current Phase")
        plt.ylabel("Number of Failures")
        plt.show()

    # Create post-optimization failure summary plots by project scale
    for project_scale in post_optimization_failure_summary['Project Scale'].unique():
        project_scale_data = post_optimization_failure_summary[
            post_optimization_failure_summary['Project Scale'] == project_scale]
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Current Phase', y='Project Id', hue='Stage', data=project_scale_data)
        plt.title(f"Post-Optimization Failure Summary - Project Scale: {project_scale}")
        plt.xlabel("Current Phase")
        plt.ylabel("Number of Failures")
        plt.show()


def table12_part2():
    # Filter failures by excluding the stage "optimizing"
    failures_filtered_df = failures_df[failures_df["Stage"] != "optimizing"]

    # Generate failure summary
    failure_summary = failures_filtered_df.groupby(['Project Scale', 'Stage', 'Current Phase'])['Project Id'].count().reset_index()

    # Display failure summary
    print("Failure Summary:")
    display(failure_summary)

    # Create failure summary plots by project scale
    for project_scale in failure_summary['Project Scale'].unique():
        project_scale_data = failure_summary[failure_summary['Project Scale'] == project_scale]
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Current Phase', y='Project Id', hue='Stage', data=project_scale_data)
        plt.title(f"Failure Summary - Project Scale: {project_scale}")
        plt.xlabel("Current Phase")
        plt.ylabel("Number of Failures")
        plt.show()


def table12_part1():
    # Generate failure summary
    # Generate failure summary
    # Filter failures by excluding the stage "optimizing"
    failures_df_filtered = failures_df[failures_df["Stage"] != "optimizing"]

    # Generate failure summary
    failure_summary = failures_df_filtered.groupby(['Stage', 'Current Phase'])['Project Id'].count().reset_index()

    # Display failure summary
    print("Failure Summary:")
    display(failure_summary)

    # Create failure summary plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Current Phase', y='Project Id', hue='Stage', data=failure_summary)
    plt.title("Failure Summary")
    plt.xlabel("Current Phase")
    plt.ylabel("Number of Failures")
    plt.show()


def table8_optimized_resources_report():
    # Group the optimization data by phase
    grouped_data = optimization_df.groupby("Phase")

    # Initialize lists to store the report data
    phase_names = []
    original_resources = []
    final_resources = []
    steps_to_optimum = []

    # Iterate over each phase group
    for phase, group in grouped_data:
        phase_names.append(phase)
        original_resources.append(
            group["Original Number of Resources"].iloc[0])  # Get the first value of original resources
        final_resources.append(group["Number of Resources"].iloc[-1])  # Get the last value of resources
        steps_to_optimum.append(group.shape[0])  # Get the number of rows (steps) in the group

    # Create a new DataFrame for the report
    report_df = pd.DataFrame({
        "Phase": phase_names,
        "Original Resources": original_resources,
        "Final Resources": final_resources,
        "Steps to Optimum": steps_to_optimum
    })

    # Print the report
    print(report_df)
    report_df.to_csv('table8_optimized_resources_report.csv', index=False)


def table7_count_bottle_neck_pre_optimization():
    # Filter the data for the "pre-optimization" stage
    pre_optimization_data = phases_df[phases_df['Stage'] == 'pre-optimization']

    # Filter the data for wait times greater than zero
    wait_times_greater_than_zero = pre_optimization_data[pre_optimization_data['Wait Time to Obtain Resources'] > 0]

    # Group the data by phase, iteration, and project scale and count the number of wait times
    wait_times_count = wait_times_greater_than_zero.groupby(['Phase', 'Iteration', 'Project Scale'])[
        'Wait Time to Obtain Resources'].count()

    # Convert the result to a DataFrame and reset the index
    wait_times_count_df = wait_times_count.reset_index()

    # Compute the mean across all iterations
    mean_wait_time = wait_times_count_df['Wait Time to Obtain Resources'].mean()

    # Group the data by phase and calculate the total wait time
    total_wait_time_by_phase = wait_times_count_df.groupby('Phase')['Wait Time to Obtain Resources'].sum()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Print the total wait time by phase
    print("Total Wait Time by Phase:")
    print(total_wait_time_by_phase)

    total_wait_time_by_phase.to_csv('table7a_count_bottle_neck_pre_optimization.csv', index=False)

    # Filter the DataFrame to include only the "pre-optimization" stage
    pre_optimization_df = phases_df[phases_df['Stage'] == 'pre-optimization']

    # Group the DataFrame by phase and project scale, and count the number of occurrences
    wait_time_counts = pre_optimization_df[pre_optimization_df['Wait Time to Obtain Resources'] > 0].groupby(
        ['Phase', 'Project Scale']).size()

    # Print the report
    print("Wait Time Report - Pre-Optimization (Grouped by Project Size)")
    print("------------------------------------------------------------")
    print(wait_time_counts)

    wait_time_counts.to_csv('table7b_count_bottle_neck_pre_optimization.csv', index=False)

    # Filter the DataFrame by stage and select the relevant columns
    pre_optimization_phases_df = phases_df[phases_df['Stage'] == 'pre-optimization']
    pre_optimization_phases_df = pre_optimization_phases_df[['Phase', 'Wait Time to Obtain Resources']]

    # Group the data by phase and calculate the sum, mean, and count
    wait_time_summary = pre_optimization_phases_df.groupby('Phase')['Wait Time to Obtain Resources'].agg(
        ['sum', 'mean', 'count'])

    # Display the report
    print(wait_time_summary)
    wait_time_summary.to_csv('table7c_count_bottle_neck_pre_optimization.csv', index=False)

    # Filter the DataFrame by stage and select the relevant columns
    pre_optimization_phases_by_size_df = phases_df[phases_df['Stage'] == 'pre-optimization']
    pre_optimization_phases_by_size_df = pre_optimization_phases_by_size_df[
        ['Phase', 'Project Scale', 'Wait Time to Obtain Resources']]

    # Group the data by phase and project size and calculate the sum, mean, and count
    wait_time_summary_by_size = pre_optimization_phases_by_size_df.groupby(['Phase', 'Project Scale'])[
        'Wait Time to Obtain Resources'].agg(['sum', 'mean', 'count'])

    # Display the report
    print(wait_time_summary_by_size)
    wait_time_summary_by_size.to_csv('table7d_count_bottle_neck_pre_optimization.csv', index=False)


def abc2():
    # Filter the data for the "pre-optimization" stage
    pre_optimization_data = phases_df[phases_df['Stage'] == 'pre-optimization']

    # Filter the data for wait times greater than zero
    wait_times_greater_than_zero = pre_optimization_data[pre_optimization_data['Wait Time to Obtain Resources'] > 0]

    # Group the data by phase, iteration, and project scale and count the number of wait times
    wait_times_count = wait_times_greater_than_zero.groupby(['Phase', 'Iteration', 'Project Scale'])[
        'Wait Time to Obtain Resources'].count()

    # Convert the result to a DataFrame and reset the index
    wait_times_count_df = wait_times_count.reset_index()

    # Compute the mean across all iterations
    mean_wait_time = wait_times_count_df['Wait Time to Obtain Resources'].mean()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # Print the report
    print(wait_times_count_df)

    # Print the mean wait time
    print("Mean Wait Time across all iterations:", mean_wait_time)

    wait_times_count_df.to_csv('wait_times_count.csv', index=False)


def visualize_resource_frequency():
    # Filter the resource data for the first iteration and the "designers" resource type
    first_iteration_df = resources_df[(resources_df["Iteration"] == 1) & (resources_df["Resource Type"] == "designers")]

    # Calculate the resource utilization for each timestamp
    timestamps = first_iteration_df["Timestamp"]
    actions = first_iteration_df["Action"]
    utilization = []
    unused_designers = []
    current_utilization = 0

    for i, action in enumerate(actions):
        if action == "obtain" or action == "after release":
            current_utilization += 1
        elif action == "release":
            current_utilization -= 1
        utilization.append(current_utilization)
        unused_designers.append(5 - current_utilization)

    # Create the bar chart for resource utilization
    plt.bar(timestamps, utilization)
    plt.xlabel("Timestamp")
    plt.ylabel("Resource Utilization")
    plt.title("Designers Resource Utilization (First Iteration)")
    plt.show()

    # Create the bar chart for unused designers
    plt.bar(timestamps, unused_designers)
    plt.xlabel("Timestamp")
    plt.ylabel("Number of Unused Designers")
    plt.title("Number of Unused Designers (First Iteration)")
    plt.show()


def report_average_resource_utilization_pre_opt():
    # Filter the resources_df DataFrame for the desired stage
    filtered_df = resources_df[resources_df['Stage'] == 'pre-optimization']

    # Calculate the total utilization for each resource type
    resource_utilization = filtered_df.groupby('Resource Type')['Resources Requested'].sum()

    # Calculate the average utilization for each resource type
    average_utilization = resource_utilization / len(filtered_df['Project Id'].unique())

    # Print the average resource utilization
    print("Average Resource Utilization (Stage: pre-optimization):")
    for resource_type, utilization in average_utilization.items():
        print(f"{resource_type}: {utilization}")

        # Filter the resources_df DataFrame for the desired stage
    filtered_df = resources_df[resources_df['Stage'] == 'pre-optimization']

    # Calculate the total utilization for each resource type
    resource_utilization = filtered_df.groupby('Resource Type')['Resources Requested'].sum()

    # Calculate the average utilization for each resource type as a percentage
    total_projects = len(filtered_df['Project Id'].unique())
    average_utilization = (resource_utilization / total_projects) * 100

    # Print the average resource utilization as a percentage
    print("Average Resource Utilization (Stage: pre-optimization):")
    for resource_type, utilization in average_utilization.items():
        print(f"{resource_type}: {utilization:.2f}%")


def wait_time_counts_report_pre_opt():
    pre_optimization_phases = phases_df[
        (phases_df['Stage'] == 'pre-optimization') & (phases_df['Wait Time to Obtain Resources'] > 0)]
    wait_time_counts = pre_optimization_phases['Phase'].value_counts()

    print(wait_time_counts)

    sns.barplot(x=wait_time_counts.index, y=wait_time_counts.values)
    plt.xlabel('Phase')
    plt.ylabel('Count')
    plt.title('Number of Times Wait Time > 0 by Phase (Pre-optimization)')
    plt.show()


def bottleneck_report():
    print("Hi")


def table6_figure_2_report_on_convergence():
    # Set the pandas options to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Filter the DataFrame for the "pre-optimization" stage
    pre_optimization_df = phases_df[phases_df['Stage'] == 'pre-optimization']

    # Group the data by iteration and phase, and calculate the mean duration
    mean_durations = pre_optimization_df.groupby(['Iteration', 'Phase'])['Phase Duration'].mean().unstack()

    # Print the mean durations for each phase across iterations
    print(mean_durations)

    # Visualize mean durations as a line graph
    sns.set(style='darkgrid')
    line_colors = {'Design': sns.color_palette('colorblind')[0], 'Implementation': sns.color_palette('colorblind')[9], 'Maintenance': sns.color_palette('colorblind')[7], 'Requirements': sns.color_palette('colorblind')[1], 'Testing': sns.color_palette('colorblind')[7]} # Replace with your custom labels
    mean_durations.plot(kind='line', color=line_colors.values())
    plt.xlabel('Iteration', fontweight='bold')
    plt.ylabel('Mean Duration (in units of time)', fontweight='bold')
    plt.title('Convergence of Phase Durations in pre-optimization stage', fontweight='bold')
    line_labels = ['Design', 'Implementation', 'Maintenance', 'Requirements', 'Testing']  # Replace with your custom labels
    legend_title = 'Phases'  # Replace with your desired title
    legend = plt.legend(line_labels, loc='best')
    legend.set_title(legend_title)
    legend.get_title().set_fontweight('bold')
    plt.show()


def report_on_convergence_by_difference():
    # Set the pandas options to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Filter the DataFrame for the "pre-optimization" stage
    pre_optimization_df = phases_df[phases_df['Stage'] == 'pre-optimization']

    # Group the data by iteration and phase, and calculate the mean duration
    mean_durations = pre_optimization_df.groupby(['Iteration', 'Phase'])['Phase Duration'].mean().unstack()

    # Calculate the differences of means between iterations
    diff_of_means = mean_durations.diff(axis=0)

    # Print the mean durations for each phase across iterations
    print("Mean Durations:")
    print(mean_durations)

    # Print the differences of means between iterations
    print("\nDifferences of Means between Iterations:")
    print(diff_of_means)


def conversion_data_report():
    RATE_OF_CHANGE_THRESHOLD = 0.01  # You might need to adjust this threshold depending on your specific simulation.
    MAX_ITERATIONS = 100
    NUM_OF_PHASES = 5
    CONVERGENCE_STREAK_THRESHOLD = 3

    def is_converged(means, prev_means):
        return all(abs((mean - prev_mean) / (prev_mean if prev_mean != 0 else 1)) < RATE_OF_CHANGE_THRESHOLD for
                   mean, prev_mean in zip(means, prev_means))

    output_data = [[] for _ in range(NUM_OF_PHASES)]
    prev_means = [0] * NUM_OF_PHASES
    consecutive_non_improvements = 0
    phases = ['requirements_analysis', 'design', 'implementation', 'testing', 'maintenance']

    for num_iterations in range(1, MAX_ITERATIONS + 1):
        print("num_iterations", num_iterations)
        reset_simulation_data()
        random.seed(random.random())
        env = simpy.Environment()
        env.process(run_software_house(env, stage, num_iterations, RESOURCES))
        env.run(until=num_iterations * 7000)  # I've used num_iterations as a basic scaling factor for simulation time

        for i, phase in enumerate(phases):
            output_data[i].append(get_mean_completion_time(phase, num_iterations))

        def get_mean_completion_time(phase_name):
            phases_data_mean_completion_time = []
            for activity in phases_log:
                phases_data_mean_completion_time.append([
                    activity.phase, activity.phase_start, activity.phase_end, activity.phase_duration,
                    activity.software_house_id,
                    activity.project_id, activity.project_scale, activity.timestamp, activity.resources_obtain_time
                ])
            phases_mean_completion_time_df = pd.DataFrame(phases_data_mean_completion_time, columns=[
                "Phase", "Phase Start", "Phase End", "Phase Duration", "Iteration", "Project Id", "Project Scale",
                "Timestamp", "Wait Time to Obtain Resources"
            ])
            phase_mean_completion_time_data = phases_mean_completion_time_df[
                phases_mean_completion_time_df['Phase'] == phase_name]
            completion_times = phase_mean_completion_time_data['Phase Duration']
            if not completion_times.empty:
                mean_completion_time = completion_times.mean()
            else:
                mean_wait_time = 0
            phases_data_mean_completion_time.clear()
            return mean_completion_time



        means = [np.mean(variable_data) for variable_data in output_data]
        # conversion_log.extend([
        #    {'phase': phase, 'num_iterations': num_iterations, 'stage': stage, 'mean': mean}
        #   for phase, mean in zip(phases, means)
        # ])
        print("means", means)

        if is_converged(means, prev_means):
            consecutive_non_improvements += 1
            if consecutive_non_improvements >= CONVERGENCE_STREAK_THRESHOLD:
                return
        else:
            consecutive_non_improvements = 0

        prev_means = means

def table11_part1_report_on_wait_time():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    def count_non_zero(x):
        return (x > 0).sum()

    wait_time_stats_by_stage_phase_size = phases_df.groupby(['Stage', 'Phase', 'Project Scale'])['Wait Time to Obtain Resources'].agg(
        ['count', 'mean', 'min', 'max', count_non_zero])

    wait_time_stats_by_stage_phase_size = wait_time_stats_by_stage_phase_size.rename(columns={"count_non_zero": "Count > 0"})

    wait_time_stats_by_stage_phase_size = wait_time_stats_by_stage_phase_size.sort_values(by=['Stage', 'mean'],
                                                                                        ascending=[True, False])

    report_df = pd.DataFrame({'Stage': wait_time_stats_by_stage_phase_size.index.get_level_values(0),
                              'Phase': wait_time_stats_by_stage_phase_size.index.get_level_values(1),
                              'Project Scale': wait_time_stats_by_stage_phase_size.index.get_level_values(2),
                              'Count': wait_time_stats_by_stage_phase_size['count'],
                              'Count > 0': wait_time_stats_by_stage_phase_size['Count > 0'],
                              'Mean Wait Time': wait_time_stats_by_stage_phase_size['mean'],
                              'Min Wait Time': wait_time_stats_by_stage_phase_size['min'],
                              'Max Wait Time': wait_time_stats_by_stage_phase_size['max']})

    print(report_df)
    report_df.to_csv('table11_part2_report_on_wait_time.csv', index=False)


def table11_part2_report_on_wait_time():
    # Set pandas options to display the entire DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Define custom aggregation function to count instances where wait time is greater than zero
    def count_non_zero(x):
        return (x > 0).sum()

    # Group the data by stage and phase and calculate the statistics for wait time
    wait_time_stats_by_stage_phase = phases_df.groupby(['Stage', 'Phase'])['Wait Time to Obtain Resources'].agg(
        ['count', 'mean', 'min', 'max', count_non_zero])

    # Rename count_non_zero column to 'Count > 0'
    wait_time_stats_by_stage_phase = wait_time_stats_by_stage_phase.rename(columns={"count_non_zero": "Count > 0"})

    # Sort the data by stage and mean wait time in descending order
    wait_time_stats_by_stage_phase = wait_time_stats_by_stage_phase.sort_values(by=['Stage', 'mean'],
                                                                                ascending=[True, False])

    # Create a report DataFrame with stage, phase, and wait time statistics columns
    report_df = pd.DataFrame({'Stage': wait_time_stats_by_stage_phase.index.get_level_values(0),
                              'Phase': wait_time_stats_by_stage_phase.index.get_level_values(1),
                              'Count': wait_time_stats_by_stage_phase['count'],
                              'Count > 0': wait_time_stats_by_stage_phase['Count > 0'],
                              'Mean Wait Time': wait_time_stats_by_stage_phase['mean'],
                              'Min Wait Time': wait_time_stats_by_stage_phase['min'],
                              'Max Wait Time': wait_time_stats_by_stage_phase['max']})

    # Print the report
    print(report_df)
    report_df.to_csv('table11_part2_report_on_wait_time.csv', index=False)


def table11_part4_report_on_wait_time():
    # Set pandas options to display the entire DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Define custom aggregation function to count instances where wait time is greater than zero
    def count_non_zero(x):
        return (x > 0).sum()

    # Group the data by stage and calculate the statistics for wait time
    wait_time_stats_by_stage = phases_df.groupby(['Stage'])['Wait Time to Obtain Resources'].agg(
        ['count', 'mean', 'min', 'max', count_non_zero])

    # Rename count_non_zero column to 'Count > 0'
    wait_time_stats_by_stage = wait_time_stats_by_stage.rename(columns={"count_non_zero": "Count > 0"})

    # Sort the data by stage and mean wait time in descending order
    wait_time_stats_by_stage = wait_time_stats_by_stage.sort_values(by=['Stage', 'mean'],
                                                                    ascending=[True, False])

    # Create a report DataFrame with stage, and wait time statistics columns
    report_df = pd.DataFrame({'Stage': wait_time_stats_by_stage.index,
                              'Count': wait_time_stats_by_stage['count'],
                              'Count > 0': wait_time_stats_by_stage['Count > 0'],
                              'Mean Wait Time': wait_time_stats_by_stage['mean'],
                              'Min Wait Time': wait_time_stats_by_stage['min'],
                              'Max Wait Time': wait_time_stats_by_stage['max']})

    print(report_df)

    # Save the DataFrame to a csv file
    report_df.to_csv('table11_part4_report_on_wait_time.csv', index=False)


def table11_part3_report_on_wait_time():
    # Set pandas options to display the entire DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Define custom aggregation function to count instances where wait time is greater than zero
    def count_non_zero(x):
        return (x > 0).sum()

    # Group the data by stage and project size and calculate the statistics for wait time
    wait_time_stats_by_stage_size = phases_df.groupby(['Stage', 'Project Scale'])['Wait Time to Obtain Resources'].agg(
        ['count', 'mean', 'min', 'max', count_non_zero])

    # Rename count_non_zero column to 'Count > 0'
    wait_time_stats_by_stage_size = wait_time_stats_by_stage_size.rename(columns={"count_non_zero": "Count > 0"})

    # Sort the data by stage, project size and mean wait time in descending order
    wait_time_stats_by_stage_size = wait_time_stats_by_stage_size.sort_values(by=['Stage', 'Project Scale', 'mean'],
                                                                              ascending=[True, True, False])

    # Create a report DataFrame with stage, project size, and wait time statistics columns
    report_df = pd.DataFrame({'Stage': wait_time_stats_by_stage_size.index.get_level_values(0),
                              'Project Scale': wait_time_stats_by_stage_size.index.get_level_values(1),
                              'Count': wait_time_stats_by_stage_size['count'],
                              'Count > 0': wait_time_stats_by_stage_size['Count > 0'],
                              'Mean Wait Time': wait_time_stats_by_stage_size['mean'],
                              'Min Wait Time': wait_time_stats_by_stage_size['min'],
                              'Max Wait Time': wait_time_stats_by_stage_size['max']})

    print(report_df)
    # Save the DataFrame to a csv file
    report_df.to_csv('table11_part3_report_on_wait_time.csv', index=False)


def table9_report_on_project_completion_times():
    print("Pre-Optimization Project Duration Statistics by Project Scale:")
    pre_optimization_project_df = project_df[
        project_df['Stage'] == 'pre-optimization']  # Filter for pre-optimization stage
    pre_optimization_duration_stats_by_scale = pre_optimization_project_df.groupby('Project Scale')['Duration'].agg(
        ['min', 'max', 'mean'])
    print(pre_optimization_duration_stats_by_scale)

    pre_optimization_duration_stats_by_scale.to_csv("table9a_report_on_project_completion_times.csv",
                                                    header=True)

    print("Pre-Optimization Project Duration Statistics Across All Project Scales:")
    pre_optimization_project_df = project_df[
        project_df['Stage'] == 'pre-optimization']  # Filter for pre-optimization stage
    pre_optimization_duration_stats = pre_optimization_project_df['Duration'].agg(
        ['min', 'max', 'mean'])
    print(pre_optimization_duration_stats)

    pre_optimization_duration_stats_by_scale.to_csv("table9b_report_on_project_completion_times.csv",
                                                    header=True)

    print("Post-Optimization Project Duration Statistics by Project Scale:")
    post_optimization_project_df = project_df[
        project_df['Stage'] == 'post-optimization']  # Filter for post-optimization stage
    post_optimization_duration_stats_by_scale = post_optimization_project_df.groupby('Project Scale')['Duration'].agg(
        ['min', 'max', 'mean'])
    print(post_optimization_duration_stats_by_scale)

    post_optimization_duration_stats_by_scale.to_csv("table9c_report_on_project_completion_times.csv",
                                                     header=True)

    print("Post-Optimization Project Duration Statistics Across All Project Scales:")
    post_optimization_project_df = project_df[
        project_df['Stage'] == 'post-optimization']  # Filter for post-optimization stage
    post_optimization_duration_stats = post_optimization_project_df['Duration'].agg(
        ['min', 'max', 'mean'])
    print(post_optimization_duration_stats)

    post_optimization_duration_stats_by_scale.to_csv("table9d_report_on_project_completion_times.csv",
                                                     header=True)


def report_on_phase_completion_times():
    print("Table 7a - Pre-Optimization - Project Duration")
    # Filter the phases_df DataFrame to include relevant columns and where Stage is "pre-optimization"
    phases_pre_optimization_data = phases_df[phases_df['Stage'] == 'pre-optimization'][
        ['Project Scale', 'Phase Duration']]

    # Group the data by 'Project Scale' and calculate min, max, and mean duration
    report_df = phases_pre_optimization_data.groupby('Project Scale')['Phase Duration'].agg(['min', 'max', 'mean'])

    # Sort the DataFrame based on the project scale order
    project_scales = ['small', 'medium', 'large']
    report_df = report_df.reindex(project_scales)

    print(report_df)
    report_df.to_csv("table7a-pre-optimization-by-size.csv")

    print("Table 7b - Post-Optimization")
    # Filter the phases_df DataFrame to include relevant columns and where Stage is "post-optimization"
    phases_post_optimization_data = phases_df[phases_df['Stage'] == 'post-optimization'][
        ['Project Scale', 'Phase Duration']]

    # Group the data by 'Project Scale' and calculate min, max, and mean duration
    report_df = phases_post_optimization_data.groupby('Project Scale')['Phase Duration'].agg(['min', 'max', 'mean'])

    # Sort the DataFrame based on the project scale order
    project_scales = ['small', 'medium', 'large']
    report_df = report_df.reindex(project_scales)

    # Display the report DataFrame
    print(report_df)
    report_df.to_csv("table7b-post-optimization-by-size.csv")

    # Filter the DataFrame for pre-optimization and post-optimization stages
    pre_optimization_phases = phases_df[phases_df['Stage'] == 'pre-optimization']
    post_optimization_phases = phases_df[phases_df['Stage'] == 'post-optimization']

    # Group the data by phase and calculate the min, max, and mean duration for pre-optimization
    pre_optimization_stats = pre_optimization_phases['Phase Duration'].agg(['min', 'max', 'mean'])

    # Group the data by phase and calculate the min, max, and mean duration for post-optimization
    post_optimization_stats = post_optimization_phases['Phase Duration'].agg(['min', 'max', 'mean'])

    print("Table 7c - Pre-Optimization")
    # Display the statistics for pre-optimization
    print("Pre-Optimization Phase Duration Statistics:")
    print(pre_optimization_stats)
    pre_optimization_stats.to_csv("table7c-pre-optimization-totals.csv")

    print("Table 7d - Post-Optimization")
    # Display the statistics for post-optimization
    print("\nPost-Optimization Phase Duration Statistics:")
    print(post_optimization_stats)
    post_optimization_stats.to_csv("table7d-post-optimization-totals.csv")


def table10_report_on_phase_completion_times_by_phase():
    print("Table 8a - Pre-Optimization")
    pre_optimization_phases = phases_df[phases_df["Stage"] == STAGE[0]]

    # Group phases by project scale and phase name, and calculate min, max, and mean phase duration
    phase_stats = pre_optimization_phases.groupby(["Project Scale", "Phase"])["Phase Duration"].agg(
        ["min", "max", "mean"])

    print("Phase Duration Statistics (Pre-Optimization):\n")
    print(phase_stats)

    phase_stats.to_csv("table10_report_on_phase_completion_times_by_phase.csv")

    print("Table 8b - Post-Optimization")
    post_optimization_phases = phases_df[phases_df["Stage"] == STAGE[2]]

    # Group phases by project scale and phase name, and calculate min, max, and mean phase duration
    phase_stats2 = post_optimization_phases.groupby(["Project Scale", "Phase"])["Phase Duration"].agg(
        ["min", "max", "mean"])

    print("Phase Duration Statistics (Post-Optimization):\n")
    print(phase_stats2)
    #
    phase_stats2.to_csv("table10_report_on_phase_completion_times_by_phase.csv")
    #
    print("Phase Duration Statistics All Sizes:\n")
    #
    # Filter the phases_df DataFrame to include relevant columns
    phases_data = phases_df[['Stage', 'Phase', 'Phase Duration']]
    #
    # Group the data by 'Stage', 'Phase', and calculate the min, max, and mean duration
    report_df = phases_data.groupby(['Stage', 'Phase'])['Phase Duration'].agg(['min', 'max', 'mean'])
    #
    # # Sort the DataFrame based on the order of stages and phases
    stages = ['pre-optimization', 'post-optimization']  # Modify this based on your stage values
    phases = ['requirements_analysis', 'design', 'implementation', 'testing',
              'maintenance']  # Modify this based on your phase values
    report_df = report_df.reindex(pd.MultiIndex.from_product([stages, phases], names=['Stage', 'Phase']))
    #
    # # Display the report DataFrame
    print(report_df)
    #
    report_df.to_csv("table10_report_on_phase_completion_times_by_phase.csv")


def xxx():
    print("Table 8a - Pre-Optimization-Wait-Time")
    # Filter the phases_df DataFrame to include relevant columns and where Stage is "pre-optimization"
    phases_pre_optimization_data = phases_df[phases_df['Stage'] == 'pre-optimization'][
        ['Project Scale', 'Wait Time to Obtain Resources']]

    # Group the data by 'Project Scale' and calculate min, max, and mean duration
    report_df = phases_pre_optimization_data.groupby('Project Scale')['Wait Time to Obtain Resources'].agg(
        ['min', 'max', 'mean'])

    # Sort the DataFrame based on the project scale order
    project_scales = ['small', 'medium', 'large']
    report_df = report_df.reindex(project_scales)

    # Display the report DataFrame
    print(report_df)
    report_df.to_csv("table7a-pre-optimization-by-size-wait-time.csv")

    print("Table 7b - Post-Optimization-Wait-Time")
    # # Filter the phases_df DataFrame to include relevant columns and where Stage is "post-optimization"
    phases_post_optimization_data = phases_df[phases_df['Stage'] == 'post-optimization'][
        ['Project Scale', 'Wait Time to Obtain Resources']]

    # Group the data by 'Project Scale' and calculate min, max, and mean duration
    report_df = phases_post_optimization_data.groupby('Project Scale')['Wait Time to Obtain Resources'].agg(
        ['min', 'max', 'mean'])

    # # Sort the DataFrame based on the project scale order
    project_scales = ['small', 'medium', 'large']
    report_df = report_df.reindex(project_scales)
    #
    # # Display the report DataFrame
    print(report_df)
    report_df.to_csv("table8b-post-optimization-by-size-wait-time.csv")
    #
    # # Filter the DataFrame for pre-optimization and post-optimization stages
    pre_optimization_phases = phases_df[phases_df['Stage'] == 'pre-optimization']
    post_optimization_phases = phases_df[phases_df['Stage'] == 'post-optimization']
    #
    # # Group the data by phase and calculate the min, max, and mean duration for pre-optimization
    pre_optimization_stats = pre_optimization_phases['Wait Time to Obtain Resources'].agg(['min', 'max', 'mean'])
    #
    # # Group the data by phase and calculate the min, max, and mean duration for post-optimization
    post_optimization_stats = post_optimization_phases['Wait Time to Obtain Resources'].agg(['min', 'max', 'mean'])
    #
    print("Table 8c - Pre-Optimization-Wait-Time")
    # Display the statistics for pre-optimization
    print("Pre-Optimization Phase Duration Statistics:")
    print(pre_optimization_stats)
    pre_optimization_stats.to_csv("table8c-pre-optimization-totals.csv")
    #
    print("Table 8d - Post-Optimization-Wait_time")
    # Display the statistics for post-optimization
    print("\nPost-Optimization Phase Duration Statistics:")
    print(post_optimization_stats)
    post_optimization_stats.to_csv("table7d-post-optimization-totals-wait-time.csv")
    #
    print("Table 9a - Pre-Optimization-wait-time")
    pre_optimization_phases = phases_df[phases_df["Stage"] == STAGE[0]]
    #
    # # Group phases by project scale and phase name, and calculate min, max, and mean phase duration
    phase_stats = pre_optimization_phases.groupby(["Project Scale", "Phase"])["Wait Time to Obtain Resources"].agg(
        ["min", "max", "mean"])
    #
    print("Wait Time to Obtain Resources Statistics (Pre-Optimization):\n")
    print(phase_stats)
    #
    phase_stats.to_csv("table8a-preoptimization-wait-time.csv")
    #
    print("Table 9b - Post-Optimization")
    post_optimization_phases = phases_df[phases_df["Stage"] == STAGE[2]]
    #
    # # Group phases by project scale and phase name, and calculate min, max, and mean phase duration
    phase_stats2 = post_optimization_phases.groupby(["Project Scale", "Phase"])["Wait Time to Obtain Resources"].agg(
        ["min", "max", "mean"])
    #
    print("Phase Duration Statistics (Post-Optimization)-wait-time:\n")
    print(phase_stats2)
    #
    phase_stats2.to_csv("table9c-postoptimization-waittime.csv")
    #
    print("Phase Duration Statistics All Sizes-wait-time:\n")
    #
    # # Filter the phases_df DataFrame to include relevant columns
    phases_data = phases_df[['Stage', 'Phase', 'Wait Time to Obtain Resources']]
    #
    # # Group the data by 'Stage' and 'Phase' and calculate the min, max, mean, count, and sum
    report_df = phases_data.groupby(['Stage', 'Phase']).agg({
        'Wait Time to Obtain Resources': ['min', 'max', 'mean', 'count', 'sum']
    })
    #
    # # Flatten the multi-level column index
    report_df.columns = ['min', 'max', 'mean', 'count', 'sum']
    #
    # # Sort the DataFrame based on the order of stages and phases
    stages = ['pre-optimization', 'post-optimization']  # Modify this based on your stage values
    phases = ['requirements_analysis', 'design', 'implementation', 'testing',
              'maintenance']  # Modify this based on your phase values
    report_df = report_df.reindex(pd.MultiIndex.from_product([stages, phases], names=['Stage', 'Phase']))
    #
    # # Display the report DataFrame
    print(report_df)
    #
    report_df.to_csv("table9c-all sizes.csv")


def yyy():
    # Check for convergence in each phase
    convergence_data = []
    for phase in phases_df['Phase'].unique():
        phase_data = phases_df[phases_df['Phase'] == phase]
        durations = phase_data['Phase Duration']
        num_iterations = len(durations)

        # Calculate convergence rate
        convergence_rate = sum(durations.diff().abs() <= 1e-6) / (num_iterations - 1) if num_iterations > 1 else 0

        # Create convergence report
        convergence_report = {
            'Phase': phase,
            'Iterations': num_iterations,
            'Convergence Rate': convergence_rate,
            'Converged': convergence_rate == 1
        }
        convergence_data.append(convergence_report)

    # Create DataFrame from convergence data
    convergence_df = pd.DataFrame(convergence_data)

    # Print convergence report
    print("Convergence Report:")
    display(convergence_df)


def report_on_iterations_to_optimize():
    print("Number of Iterations Needed to Optimize Each Phase:")
    iterations_by_phase = optimization_df.groupby('Phase')['Iteration'].max()
    print(iterations_by_phase)
    iterations_by_phase.to_csv("iterations_needed_to_optimize.csv", header=True)


if __name__ == '__main__':
    main()