import simpy
import random
import pandas as pd
import sys
import numpy as np
import scipy.stats as stats

NEW_PROJECT_ARRIVAL_INTERVAL = [30, 40, 35]

RESOURCES = {
    'business_analysts': 5,
    'designers': 5,
    'programmers': 10,
    'testers': 20,
    'maintenance_people': 5
}

PHASES = {
    'requirements_analysis': {'duration_range': [3, 5], 'resource': 'business_analysts'},
    'design': {'duration_range': [5, 10], 'resource': 'designers'},
    'implementation': {'duration_range': [15, 20], 'resource': 'programmers'},
    'testing': {'duration_range': [5, 10], 'resource': 'testers'},
    'maintenance': {'duration_range': [1, 3], 'resource': 'maintenance_people'}
}

PROJECT_TYPES = {
    'small': {'proportion': 0.7, 'requirements': {'business_analysts': 1, 'designers': 1, 'programmers': 2, 'testers': 2, 'maintenance_people': 1}, 'error_probability': 0.1},
    'medium': {'proportion': 0.25, 'requirements': {'business_analysts': 2, 'designers': 2, 'programmers': 4, 'testers': 6, 'maintenance_people': 2}, 'error_probability': 0.2},
    'large': {'proportion': 0.05, 'requirements': {'business_analysts': 5, 'designers': 5, 'programmers': 10, 'testers': 20, 'maintenance_people': 5}, 'error_probability': 0.3},
}

TOTAL_PROJECTS = 50
list_of_projects = []
failures_log = []
resources_wait_time_log = []
resources_log = []
phases_log = []
failure_log = []


class SoftwareHouse(object):
    def __init__(self, env, id_input, resources_input):
        self.env = env
        self.id = id_input
        self.resources = {resource: simpy.Container(env, init=amount) for resource, amount in resources_input.items()}


def run_software_house(env, id_input, resources_input):
    software_house = SoftwareHouse(env, id_input, resources_input)
    project_number = 0
    while True:
        new_project_wait_time = random.triangular(*NEW_PROJECT_ARRIVAL_INTERVAL)
        yield env.timeout(new_project_wait_time)
        if project_number < TOTAL_PROJECTS:
            project_number += 1
            project = Project(software_house, project_number)
            list_of_projects.append(project)
            env.process(project.start_project())


class ResourceReport(object):
    def __init__(self, type_of_resource_input, software_house_id_input, project_id_input, resources_available_input, action_input, timestamp_input, number_of_resources_request_input, project_scale_input):
        self.type_of_resource = type_of_resource_input
        self.software_house_id = software_house_id_input
        self.project_id = project_id_input
        self.resources_available = resources_available_input
        self.action = action_input
        self.timestamp = timestamp_input
        self.number_of_resources_request = number_of_resources_request_input
        self.project_scale = project_scale_input


class FailureReport(object):
    def __init__(self, current_phase_input, fail_to_phase_input, software_house_id_input, project_id_input, timestamp_input, project_scale_input):
        self.current_phase = current_phase_input
        self.fail_to_phase = fail_to_phase_input
        self.software_house_id = software_house_id_input
        self.project_id = project_id_input
        self.timestamp = timestamp_input
        self.project_scale = project_scale_input


class PhasesReport(object):
    def __init__(self, phase_input, phase_start_input, phase_end_input, phase_duration_input, software_house_id_input, project_id_input, project_scale_input, timestamp_input, resources_obtain_time_input):
        self.phase = phase_input
        self.phase_start = phase_start_input
        self.phase_end = phase_end_input
        self.phase_duration = phase_duration_input
        self.software_house_id = software_house_id_input
        self.project_id = project_id_input
        self.project_scale = project_scale_input
        self.timestamp = timestamp_input
        self.resources_obtain_time = resources_obtain_time_input


class Project(object):
    def __init__(self, software_house_input, id_input):
        self.software_house_id = software_house_input.id
        self.software_house = software_house_input
        self.id = id_input
        self.start_time = 0
        self.end_time = 0
        self.duration = 0
        self.project_scale = ""
        self.design_phase_failures = 0
        self.implementation_phase_failures = 0
        self.testing_phase_failures = 0
        self.maintenance_phase_failures = 0
        self.compute_project_scale()

    def compute_project_scale(self):
        option = random.uniform(0, 1)
        cumulative_proportion = 0
        for project_type, item in PROJECT_TYPES.items():
            cumulative_proportion += item['proportion']
            if option <= cumulative_proportion:
                self.project_scale = project_type
                break

    def get_failure_probability(self):
        failure_probability = PROJECT_TYPES[self.project_scale]['error_probability']
        return failure_probability

    def project_resource_request(self, resource_type, number_of_resources):
        resource = self.software_house.resources[resource_type]
        resources_log.append(ResourceReport(resource_type, self.software_house.id, self.id, resource.level, "request", self.software_house.env.now, number_of_resources, self.project_scale))
        yield resource.get(number_of_resources)
        resources_log.append(ResourceReport(resource_type, self.software_house.id, self.id, resource.level, "obtain", self.software_house.env.now, number_of_resources, self.project_scale))

    def project_resource_release(self, resource_type, number_of_resources):
        resource = self.software_house.resources[resource_type]
        resources_log.append(ResourceReport(resource_type, self.software_house.id, self.id, resource.level, "release", self.software_house.env.now, number_of_resources, self.project_scale))
        yield resource.put(number_of_resources)
        resources_log.append(ResourceReport(resource_type, self.software_house.id, self.id, resource.level, "after release", self.software_house.env.now, number_of_resources, self.project_scale))

    def project_failure_check(self, current_phase):
        phase_names = list(PHASES.keys())
        if current_phase in phase_names:
            current_phase_index = phase_names.index(current_phase)
            random_number = random.uniform(0, 1)
            if (random_number <= self.get_failure_probability() / 100.0) and (current_phase_index > 0):
                previous_phase_index = current_phase_index - 1
                previous_phase_name = phase_names[previous_phase_index]
                previous_phase_resource = PHASES.get(previous_phase_name, {}).get('resource')
                failures_log.append(FailureReport(current_phase, previous_phase_name, self.software_house.id, self.id,self.software_house.env.now, self.project_scale))
                yield from self.project_phase(previous_phase_name, previous_phase_resource)
            else:
                next_phase_index = (current_phase_index + 1) % len(phase_names)
                next_phase_name = phase_names[next_phase_index]
                next_phase_resource = PHASES[next_phase_name]['resource']
                yield from self.project_phase(next_phase_name, next_phase_resource)

    def project_phase(self, phase_name, resource_type):
        phase_start = self.software_house.env.now
        yield from self.project_resource_request(resource_type, PROJECT_TYPES[self.project_scale]['requirements'][resource_type])
        phase_obtain_time = self.software_house.env.now
        duration = round(random.uniform(*PHASES[phase_name]['duration_range']))
        yield self.software_house.env.timeout(duration)
        yield from self.project_resource_release(resource_type, PROJECT_TYPES[self.project_scale]['requirements'][resource_type])
        phase_end = self.software_house.env.now
        time_to_obtain_resources = phase_obtain_time - phase_start
        time_for_phase = phase_end - phase_start
        phases_log.append(PhasesReport(phase_name, phase_start, phase_end, time_for_phase, self.software_house.id, self.id, self.project_scale, self.software_house.env.now, time_to_obtain_resources))
        yield from self.project_failure_check(phase_name)

    def start_project(self):
        self.start_time = self.software_house.env.now

        first_phase_name, first_phase_details = next(iter(PHASES.items()))
        first_resource = first_phase_details['resource']
        yield from self.project_phase(first_phase_name, first_resource)

        self.end_time = self.software_house.env.now
        self.duration = self.end_time - self.start_time


def get_mean_wait_time(phase_name):
    phases_data = []
    for activity in phases_log:
        phases_data.append([
            activity.phase, activity.phase_start, activity.phase_end, activity.phase_duration, activity.software_house_id,
            activity.project_id, activity.project_scale, activity.timestamp, activity.resources_obtain_time
        ])
    phases_df = pd.DataFrame(phases_data, columns=[
        "Phase", "Phase Start", "Phase End", "Phase Duration", "Iteration", "Project Id", "Project Scale",
        "Timestamp", "Wait Time to Obtain Resources"
    ])
    phase_data = phases_df[phases_df['Phase'] == phase_name]
    wait_times = phase_data['Wait Time to Obtain Resources']
    if not wait_times.empty:
        mean_wait_time = wait_times.mean()
    else:
        mean_wait_time = 0
    return mean_wait_time


def get_mean_completion_time(phase_name):
    phases_data = []
    for activity in phases_log:
        phases_data.append([
            activity.phase, activity.phase_start, activity.phase_end, activity.phase_duration,
            activity.software_house_id,
            activity.project_id, activity.project_scale, activity.timestamp, activity.resources_obtain_time
        ])
    phases_df = pd.DataFrame(phases_data, columns=[
        "Phase", "Phase Start", "Phase End", "Phase Duration", "Iteration", "Project Id", "Project Scale",
        "Timestamp", "Wait Time to Obtain Resources"
    ])
    phase_data = phases_df[phases_df['Phase'] == phase_name]
    completion_times = phase_data['Phase Duration']
    if not completion_times.empty:
        mean_completion_time = completion_times.mean()
    else:
        mean_wait_time = 0
    return mean_completion_time


def get_mean_completion_time(phase_name, iteration):
    phases_data = []
    for activity in phases_log:
        phases_data.append([
            activity.phase, activity.phase_start, activity.phase_end, activity.phase_duration,
            activity.software_house_id,
            activity.project_id, activity.project_scale, activity.timestamp, activity.resources_obtain_time
        ])
    phases_df = pd.DataFrame(phases_data, columns=[
        "Phase", "Phase Start", "Phase End", "Phase Duration", "Iteration", "Project Id", "Project Scale",
        "Timestamp", "Wait Time to Obtain Resources"
    ])
    filtered_data = phases_df[(phases_df['Phase'] == phase_name) &
                              (phases_df['Iteration'] == iteration)]
    completion_times = filtered_data['Phase Duration']
    if not completion_times.empty:
        mean_completion_time = completion_times.mean()
    else:
        mean_completion_time = 0
    return mean_completion_time


def reset_simulation_data():
    phases_log.clear()


def optimize_resource(phase, resource_name, min_value, max_value, initial_step_size):
    step_size = initial_step_size
    best_mean = get_mean_wait_time(phase)

    # Set the smallest step size we'll consider
    min_step_size = 1

    # Number of iterations without improvements
    no_improvements = 0

    # Maximum number of iterations without improvements
    max_no_improvements = 10

    print ("starting optimization of ", phase)
    while step_size >= min_step_size:
        for direction in [1, -1]:
            old_resource_value = RESOURCES[resource_name]
            RESOURCES[resource_name] += direction * step_size
            RESOURCES[resource_name] = min(max(RESOURCES[resource_name], min_value), max_value)

            reset_simulation_data()
            simulate()
            new_mean = get_mean_wait_time(phase)

            # If we haven't made an improvement, restore the old resource value
            if new_mean >= best_mean:
                RESOURCES[resource_name] = old_resource_value
                no_improvements += 1
            else:
                best_mean = new_mean
                no_improvements = 0

                # Adjust the step size based on the rate of improvement
                step_size = max(min_step_size, int(abs(new_mean - best_mean) * step_size))

            if no_improvements >= max_no_improvements:
                # Random restart
                RESOURCES[resource_name] = random.randint(min_value, max_value)
                no_improvements = 0

        # If we couldn't improve in either direction, reduce the step size
        if RESOURCES[resource_name] == old_resource_value:
            step_size //= 2

    print(f"Optimal Number of {resource_name}:", RESOURCES[resource_name])
    print ("ending optimization of ", phase)

    return best_mean


def optimize_resources():
    # minimum should be the lowest value to run a large project
    resource_info = {
        'business_analysts': {
            'phase': 'requirements_analysis',
            'min_value': PROJECT_TYPES['large']['requirements']['business_analysts'],
            'max_value': 200,
            'step_size': 1
        },
        'designers': {
            'phase': 'design',
            'min_value': PROJECT_TYPES['large']['requirements']['designers'],
            'max_value': 200,
            'step_size': 1
        },
        'programmers': {
            'phase': 'implementation',
            'min_value': PROJECT_TYPES['large']['requirements']['programmers'],
            'max_value': 200,
            'step_size': 1
        },
        'testers': {
            'phase': 'testing',
            'min_value': PROJECT_TYPES['large']['requirements']['testers'],
            'max_value': 200,
            'step_size': 1
        },
        'maintenance_people': {
            'phase': 'maintenance',
            'min_value': PROJECT_TYPES['large']['requirements']['maintenance_people'],
            'max_value': 200,
            'step_size': 1
        }
    }

    for resource_name, resource_data in resource_info.items():
        phase = resource_data['phase']
        min_value = resource_data['min_value']
        max_value = resource_data['max_value']
        initial_step_size = resource_data['step_size']
        optimize_resource(phase, resource_name, min_value, max_value, initial_step_size)


def main():
    new_recursion_limit = 3000  # Set the new recursion limit

    # Check if the new recursion limit is within the allowable range
    if new_recursion_limit > sys.getrecursionlimit():
        print("Warning: New recursion limit is higher than the default limit of ", sys.getrecursionlimit(), ".")

    sys.setrecursionlimit(new_recursion_limit)  # Set the new recursion limit

    print("New recursion limit:", sys.getrecursionlimit())  # Print the updated recursion limit

    print("resources initial simulation")
    simulate()

    print("optimizing resources")
    optimize_resources()

    print("running optimized simulation")
    simulate()

    report()


def simulate():
    RATE_OF_CHANGE_THRESHOLD = 0.01  # You might need to adjust this threshold depending on your specific simulation.
    MAX_ITERATIONS = 100
    NUM_OF_PHASES = 5
    CONVERGENCE_STREAK_THRESHOLD = 3

    def is_converged(means, prev_means):
        return all(abs((mean - prev_mean) / (prev_mean if prev_mean != 0 else 1)) < RATE_OF_CHANGE_THRESHOLD for mean, prev_mean in zip(means, prev_means))

    output_data = [[] for _ in range(NUM_OF_PHASES)]
    prev_means = [0] * NUM_OF_PHASES
    consecutive_non_improvements = 0
    phases = ['requirements_analysis', 'design', 'implementation', 'testing', 'maintenance']

    for num_iterations in range(1, MAX_ITERATIONS + 1):
        reset_simulation_data()
        random.seed(random.random())
        env = simpy.Environment()
        env.process(run_software_house(env, num_iterations, RESOURCES))
        env.run(until=num_iterations * 1000)  # I've used num_iterations as a basic scaling factor for simulation time

        for i, phase in enumerate(phases):
            output_data[i].append(get_mean_completion_time(phase, num_iterations))

        means = [np.mean(variable_data) for variable_data in output_data]

        print(f"Iteration {num_iterations} done!")
        if is_converged(means, prev_means):
            consecutive_non_improvements += 1
            if consecutive_non_improvements >= CONVERGENCE_STREAK_THRESHOLD:
                print("Convergence achieved")
                return
        else:
            consecutive_non_improvements = 0

        prev_means = means


    # Rest of the code...


    #comTime =  get_mean_completion_time('design')
    #print ("design mean completion time is ", comTime)

    # Initialize variables for convergence analysis
    output_data = []

    # Run the simulation and collect data for the output metric


def report():
    project_data = []
    resources_data = []
    phases_data = []
    failures_data = []

    for individual_project in list_of_projects:
        project_completion = "NO" if individual_project.end_time == 0 else "YES"
        project_data.append([
            individual_project.software_house_id, individual_project.id, individual_project.start_time,
            individual_project.end_time, individual_project.duration, individual_project.project_scale,
            individual_project.design_phase_failures, individual_project.implementation_phase_failures,
            individual_project.testing_phase_failures, individual_project.maintenance_phase_failures, project_completion
        ])

    for activity in resources_log:
        resources_data.append([
            activity.type_of_resource, activity.software_house_id, activity.project_id, activity.resources_available,
            activity.action, activity.timestamp, activity.number_of_resources_request, activity.project_scale
        ])

    for activity in phases_log:
        phases_data.append([
            activity.phase, activity.phase_start, activity.phase_end, activity.phase_duration, activity.software_house_id,
            activity.project_id, activity.project_scale, activity.timestamp, activity.resources_obtain_time
        ])

    for activity in failures_log:
        failures_data.append([
            activity.current_phase, activity.fail_to_phase, activity.software_house_id, activity.project_id,
            activity.timestamp, activity.project_scale
        ])

    project_df = pd.DataFrame(project_data, columns=[
        "Iteration", "Project Id", "Project Start Time", "Project End Time", "Duration", "Project Scale",
        "Number of Design Phase Failures", "Number of Implementation Phase Failures",
        "Number of Testing Phase Failures", "Number of Maintenance Phase Failures", "Project Completion"
    ])

    resources_df = pd.DataFrame(resources_data, columns=[
        "Resource Type", "Iteration", "Project Id", "Resources Available", "Action", "Timestamp",
        "Resources Requested", "Project Scale"
    ])

    phases_df = pd.DataFrame(phases_data, columns=[
        "Phase", "Phase Start", "Phase End", "Phase Duration", "Iteration", "Project Id", "Project Scale",
        "Timestamp", "Wait Time to Obtain Resources"
    ])

    failures_df = pd.DataFrame(failures_data, columns=[
        "Current Phase", "Fail to Phase", "Iteration", "Project Id", "Timestamp", "Project Scale"
    ])

    project_df.to_csv("project_data.csv", index=False)
    resources_df.to_csv("resources_report.csv", index=False)
    phases_df.to_csv("phases_report.csv", index=False)
    failures_df.to_csv("failures_report.csv", index=False)

    # Output simulation statistics

    # Count the occurrences of each project scale
    scale_counts = project_df["Project Scale"].value_counts()

    # Calculate the average number of projects by scale
    average_projects_by_scale = scale_counts.mean()

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame({"Project Scale": scale_counts.index, "Count": scale_counts.values})

    # Add a row for the average number of projects
    average_row = pd.DataFrame({"Project Scale": ["Average"], "Count": [average_projects_by_scale]})
    result_df = pd.concat([result_df, average_row], ignore_index=True)

    # Get the number of unique iterations
    num_unique_software_houses = len(project_df["Iteration"].unique())

    # Add a new column "Average" by dividing the "Count" column by number of iterations
    result_df["Average"] = result_df["Count"] / num_unique_software_houses

    # Calculate the total duration by project scale
    total_duration_by_scale = project_df.groupby("Project Scale")["Duration"].sum().reset_index()

    # Merge the total duration by scale with the result_df DataFrame
    result_df = result_df.merge(total_duration_by_scale, on="Project Scale")

    # Rename the column for clarity
    result_df = result_df.rename(columns={"Duration": "Total Duration"})

    # Calculate the average duration by dividing "Total Duration" by "Count"
    result_df["Average Duration"] = result_df["Total Duration"] / result_df["Count"]

    # Print the result DataFrame
    print("Number of Projects Average by Project Scale Across All Iterations:")
    print(result_df)

    ##########################################################

    # Group the data by the "Phase" column and calculate the average, minimum, and maximum duration
    duration_stats_by_phase = phases_df.groupby("Phase")["Phase Duration"].agg(["mean", "min", "max"]).reset_index()

    # Rename the columns for clarity
    duration_stats_by_phase = duration_stats_by_phase.rename(
        columns={"mean": "Average Duration", "min": "Minimum Duration", "max": "Maximum Duration"})

    # Display the new table in a tabular format
    print("Duration Statistics by Phase:")
    print(duration_stats_by_phase.to_string(index=False))

    ##########################################################

    # Calculate the duration statistics by phase and project scale using pivot_table
    duration_stats_by_phase_scale = pd.pivot_table(phases_df, values="Phase Duration", index="Phase",
                                                   columns="Project Scale",
                                                   aggfunc=["mean", "min", "max"]).reset_index()

    # Rename the columns for clarity
    duration_stats_by_phase_scale.columns = ["Phase", "Large Scale (Mean)", "Medium Scale (Mean)", "Small Scale (Mean)",
                                             "Large Scale (Min)", "Medium Scale (Min)", "Small Scale (Min)",
                                             "Large Scale (Max)", "Medium Scale (Max)", "Small Scale (Max)"]

    # Display the new table in a tabular format
    print("Duration Statistics by Phase and Project Scale:")
    print(duration_stats_by_phase_scale.to_string(index=False))

    ##########################################################

    # Group the data by "Phase" and calculate the average wait time to obtain resources
    wait_time_stats_by_phase = phases_df.groupby("Phase")["Wait Time to Obtain Resources"].mean().reset_index()

    # Display the wait time statistics by phase in a tabular format
    print("Wait Time to Obtain Resources by Phase:")
    print(wait_time_stats_by_phase)

    ##########################################################

    # Compute power for each phase
    phases_df['Power'] = 1 / phases_df['Phase Duration']

    # Group data by 'Phase' and calculate the mean power
    power_by_phase = phases_df.groupby('Phase')['Power'].mean().reset_index()

    # Print power for each phase
    print("Power for each Phase:")
    print(power_by_phase)

    ##########################################################


def analyze_simulation_results():
    # Load the simulation results from CSV files
    project_df = pd.read_csv("project_data.csv")
    resources_df = pd.read_csv("resources_report.csv")
    phases_df = pd.read_csv("phases_report.csv")
    failures_df = pd.read_csv("failures_report.csv")

    # Resource Utilization
    resource_utilization = resources_df.groupby("Resource Type")["Resources Available"].mean()
    print("Resource Utilization:")
    print(resource_utilization)

    # Phase Duration Variation
    phase_duration_variation = phases_df.groupby(["Phase", "Project Scale"])["Phase Duration"].std()
    print("Phase Duration Variation:")
    print(phase_duration_variation)

    # Failure Analysis
    failure_counts = failures_df["Current Phase"].value_counts()
    print("Failure Counts by Phase:")
    print(failure_counts)

    # Phase Overlap
    phase_overlap = phases_df.groupby("Project Id")[["Phase Start", "Phase End"]].agg(["min", "max"])
    phase_overlap["Overlap"] = phase_overlap[("Phase Start", "max")] - phase_overlap[("Phase End", "min")]
    print("Phase Overlap:")
    print(phase_overlap["Overlap"])


if __name__ == '__main__':
    main()
