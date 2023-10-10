# Waterfall Sym
# Copyright (C) 2023. Antonios Saravanos and Matthew X. Curinga
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU GENERAL PUBLIC LICENSE (the "License") as
# published by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the License for more details.
#
# You should have received a copy of the License along with this program.
# If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import simpy
import random
import pandas as pd
import sys
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from settings import *

# declare logs
projects_log = []
failures_log = []
resources_wait_time_log = []
resources_log = []
phases_log = []
failure_log = []
#conversion_log = []

optimization_log = []

history_phases_log = []
history_failure_log = []
history_resources_log = []
history_resources_wait_time_log = []
history_failures_log = []
history_projects_log = []
#history_conversion_log = []

count = 0


class SoftwareHouse(object):
    def __init__(self, env, stage_input, id_input, resources_input):
        self.env = env
        self.stage = stage_input
        self.id = id_input
        self.resources = {resource: simpy.Container(env, init=amount) for resource, amount in resources_input.items()}


def run_software_house(env, stage, id_input, resources_input):
    software_house = SoftwareHouse(env, stage, id_input, resources_input)
    project_number = 0
    while True:
        new_project_wait_time = random.triangular(*NEW_PROJECT_ARRIVAL_INTERVAL)
        yield env.timeout(new_project_wait_time)
        if project_number < TOTAL_PROJECTS:
            project_number += 1
            project = Project(stage, software_house, project_number)
            projects_log.append(project)
            env.process(project.start_project(stage))


class ResourceReport(object):
    def __init__(self, stage_input, type_of_resource_input, software_house_id_input, project_id_input, resources_available_input, action_input, timestamp_input, number_of_resources_request_input, project_scale_input):
        self.stage = stage_input
        self.type_of_resource = type_of_resource_input
        self.software_house_id = software_house_id_input
        self.project_id = project_id_input
        self.resources_available = resources_available_input
        self.action = action_input
        self.timestamp = timestamp_input
        self.number_of_resources_request = number_of_resources_request_input
        self.project_scale = project_scale_input


class FailureReport(object):
    def __init__(self, stage_input, current_phase_input, fail_to_phase_input, software_house_id_input, project_id_input, timestamp_input, project_scale_input):
        self.stage = stage_input
        self.current_phase = current_phase_input
        self.fail_to_phase = fail_to_phase_input
        self.software_house_id = software_house_id_input
        self.project_id = project_id_input
        self.timestamp = timestamp_input
        self.project_scale = project_scale_input


class PhasesReport(object):
    def __init__(self, stage_input, phase_input, phase_start_input, phase_end_input, phase_duration_input, software_house_id_input, project_id_input, project_scale_input, timestamp_input, resources_obtain_time_input):
        self.stage = stage_input
        self.phase = phase_input
        self.phase_start = phase_start_input
        self.phase_end = phase_end_input
        self.phase_duration = phase_duration_input
        self.software_house_id = software_house_id_input
        self.project_id = project_id_input
        self.project_scale = project_scale_input
        self.timestamp = timestamp_input
        self.resources_obtain_time = resources_obtain_time_input


class OptimizationReport(object):
    def __init__(self, stage_input, phase_input, mean_wait_time_input, iteration_input, original_resource_value_input, resource_value_input):
        self.stage = stage_input
        self.phase = phase_input
        self.mean_wait_time = mean_wait_time_input
        self.iteration = iteration_input
        self.resource_value = resource_value_input
        self.original_resource_value = original_resource_value_input


class Phase:
    def __init__(self, name, resource_type):
        self.name = name
        self.resource_type = resource_type
        self.start_time = 0
        self.end_time = 0
        self.duration = 0
        self.wait_time = 0  # Attribute to capture wait time for resources

    def perform_phase_actions(self, env, project, software_house):
        resource_type = self.resource_type
        resource_request_time = env.now
        yield from project.project_resource_request(software_house.stage, resource_type,
                                                    PROJECT_TYPES[project.project_scale]['requirements'][resource_type])
        self.start_time = env.now
        duration = round(random.uniform(*PHASES[self.name]['duration_range']))
        yield env.timeout(duration)
        self.end_time = env.now
        yield from project.project_resource_release(software_house.stage, resource_type,
                                                    PROJECT_TYPES[project.project_scale]['requirements'][resource_type])
        self.duration = self.end_time - self.start_time
        self.wait_time = self.start_time - resource_request_time  # Calculate wait time for resources


class Project(object):
    def __init__(self, stage_input, software_house_input, id_input):
        self.software_house_id = software_house_input.id
        self.software_house = software_house_input
        self.id = id_input
        self.stage = stage_input
        self.start_time = 0
        self.end_time = 0
        self.duration = 0
        self.project_scale = ""
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

    def log_resource(self, stage: str, resource_type: str, status: str, number_of_resources: int):
        resource = self.software_house.resources[resource_type]
        resources_log.append(ResourceReport(stage, resource_type, self.software_house.id, self.id, resource.level, status, self.software_house.env.now, number_of_resources, self.project_scale))
        return resource

    def project_resource_request(self, stage: str, resource_type: str, number_of_resources: int):
        resource = self.log_resource(stage, resource_type, "request", number_of_resources)
        yield resource.get(number_of_resources)
        self.log_resource(stage, resource_type, "obtain", number_of_resources)

    def project_resource_release(self, stage: str, resource_type: str, number_of_resources: int):
        resource = self.log_resource(stage, resource_type, "release", number_of_resources)
        yield resource.put(number_of_resources)
        self.log_resource(stage, resource_type, "after release", number_of_resources)

    def project_phase(self, stage, phase_name, resource_type):
        current_phase = phase_name
        phase_start = self.software_house.env.now
        phase = Phase(phase_name, resource_type)
        yield from phase.perform_phase_actions(self.software_house.env, self, self.software_house)
        phase_end = self.software_house.env.now
        time_for_phase = phase_end - phase_start
        self.log_phase_info(stage, phase_name, phase_start, phase_end, time_for_phase,
                            phase.wait_time)

        if self.check_for_failure(phase_name):
            previous_phase_name, previous_phase_resource = self.get_previous_phase_info(phase_name)
            failures_log.append(FailureReport(stage, current_phase, previous_phase_name, self.software_house.id,
                                              self.id, self.software_house.env.now, self.project_scale))
            yield from self.project_phase(stage, previous_phase_name, previous_phase_resource)
        elif self.is_not_last_phase(phase_name):
            next_phase_name, next_phase_resource = self.get_next_phase_info(phase_name)
            yield from self.project_phase(stage, next_phase_name, next_phase_resource)

    def check_for_failure(self, phase_name: str):
        phase_names = list(PHASES.keys())
        current_phase_index = phase_names.index(phase_name)
        random_number = random.uniform(0, 1)
        if (random_number <= self.get_failure_probability()) and (current_phase_index > 0):
            return True
        return False

    def get_next_phase_info(self, phase_name: str):
        phase_names = list(PHASES.keys())
        next_phase_index = (phase_names.index(phase_name) + 1) % len(phase_names)
        next_phase_name = phase_names[next_phase_index]
        next_phase_resource = PHASES[next_phase_name]['resource']
        return next_phase_name, next_phase_resource

    def get_previous_phase_info(self, phase_name: str):
        phase_names = list(PHASES.keys())
        current_phase_index = phase_names.index(phase_name)
        previous_phase_index = current_phase_index - 1
        previous_phase_name = phase_names[previous_phase_index]
        previous_phase_resource = PHASES.get(previous_phase_name, {}).get('resource')
        return previous_phase_name, previous_phase_resource

    def log_phase_info(self, stage: str, phase_name: str, phase_start: float, phase_end: float, time_for_phase: float,
                       time_to_obtain_resources: float):
        phases_log.append(
            PhasesReport(stage, phase_name, phase_start, phase_end, time_for_phase, self.software_house_id, self.id,
                         self.project_scale, self.software_house.env.now, time_to_obtain_resources))

    def is_not_last_phase(self, phase_name: str):
        phase_names = list(PHASES.keys())
        current_phase_index = phase_names.index(phase_name)
        return current_phase_index != len(phase_names) - 1

    def start_project(self, stage):
        self.start_time = self.software_house.env.now
        first_phase_name, first_phase_details = next(iter(PHASES.items()))
        first_resource = first_phase_details['resource']
        yield from self.project_phase(stage, first_phase_name, first_resource)

        self.end_time = self.software_house.env.now
        self.duration = self.end_time - self.start_time


def get_mean_wait_time(phase_name):
    phases_data_mean_wait_time = []
    for activity in phases_log:
        phases_data_mean_wait_time.append([
            activity.phase, activity.phase_start, activity.phase_end, activity.phase_duration, activity.software_house_id,
            activity.project_id, activity.project_scale, activity.timestamp, activity.resources_obtain_time
        ])
    phases_data_mean_wait_time_df = pd.DataFrame(phases_data_mean_wait_time, columns=[
        "Phase", "Phase Start", "Phase End", "Phase Duration", "Iteration", "Project Id", "Project Scale",
        "Timestamp", "Wait Time to Obtain Resources"
    ])
    phase_data_mean_wait_time = phases_data_mean_wait_time_df[phases_data_mean_wait_time_df['Phase'] == phase_name]
    wait_times = phase_data_mean_wait_time['Wait Time to Obtain Resources']
    if not wait_times.empty:
        mean_wait_time = wait_times.mean()
    else:
        mean_wait_time = 0
    phases_data_mean_wait_time.clear()
    return mean_wait_time


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
    phase_mean_completion_time_data = phases_mean_completion_time_df[phases_mean_completion_time_df['Phase'] == phase_name]
    completion_times = phase_mean_completion_time_data['Phase Duration']
    if not completion_times.empty:
        mean_completion_time = completion_times.mean()
    else:
        mean_wait_time = 0
    phases_data_mean_completion_time.clear()
    return mean_completion_time


def get_mean_completion_time(phase_name, iteration):
    phases_data_mean_completion_time_with_iteration = []
    for activity in phases_log:
        phases_data_mean_completion_time_with_iteration.append([
            activity.phase, activity.phase_start, activity.phase_end, activity.phase_duration,
            activity.software_house_id,
            activity.project_id, activity.project_scale, activity.timestamp, activity.resources_obtain_time
        ])
    phases_data_mean_completion_time_with_iteration_df = pd.DataFrame(phases_data_mean_completion_time_with_iteration, columns=[
        "Phase", "Phase Start", "Phase End", "Phase Duration", "Iteration", "Project Id", "Project Scale",
        "Timestamp", "Wait Time to Obtain Resources"
    ])
    filtered_data = phases_data_mean_completion_time_with_iteration_df[(phases_data_mean_completion_time_with_iteration_df['Phase'] == phase_name) &
                              (phases_data_mean_completion_time_with_iteration_df['Iteration'] == iteration)]
    completion_times = filtered_data['Phase Duration']
    if not completion_times.empty:
        mean_completion_time = completion_times.mean()
    else:
        mean_completion_time = 0
    phases_data_mean_completion_time_with_iteration.clear()
    return mean_completion_time


def reset_simulation_data():
    history_phases_log.extend(phases_log)
    history_failure_log.extend(failure_log)
    history_resources_log.extend(resources_log)
    history_resources_wait_time_log.extend(resources_wait_time_log)
    history_failures_log.extend(failures_log)
    history_projects_log.extend(projects_log)
    phases_log.clear()
    projects_log.clear()
    failures_log.clear()
    resources_wait_time_log.clear()
    resources_log.clear()
    failure_log.clear()


def optimize_resource(stage, phase, resource_name, min_value, max_value, initial_step_size):
    iterations = 0
    step_size = initial_step_size
    best_mean = get_mean_wait_time(phase)
    starting_resource_value = RESOURCES[resource_name]

    # Set the smallest step size we'll consider
    min_step_size = 1

    # Number of iterations without improvements
    no_improvements = 0

    # Maximum number of iterations without improvements
    max_no_improvements = 10

    while step_size >= min_step_size:
        for direction in [1, -1]:
            iterations += 1
            old_resource_value = RESOURCES[resource_name]
            RESOURCES[resource_name] += direction * step_size
            RESOURCES[resource_name] = min(max(RESOURCES[resource_name], min_value), max_value)

            reset_simulation_data()
            simulate(stage)
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

            optimization_log.append(
                OptimizationReport(stage, phase, new_mean, iterations, starting_resource_value, RESOURCES[resource_name]))

        # If we couldn't improve in either direction, reduce the step size
        if RESOURCES[resource_name] == old_resource_value:
            step_size //= 2

    print(f"Optimal Number of {resource_name}:", RESOURCES[resource_name])

    return best_mean


def optimize_resources(stage):
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
        optimize_resource(stage, phase, resource_name, min_value, max_value, initial_step_size)


def main():
    new_recursion_limit = 3000  # Set the new recursion limit

    # Check if the new recursion limit is within the allowable range
    if new_recursion_limit > sys.getrecursionlimit():
        print("Warning: New recursion limit is higher than the default limit of ", sys.getrecursionlimit(), ".")

    sys.setrecursionlimit(new_recursion_limit)  # Set the new recursion limit

    print("New recursion limit:", sys.getrecursionlimit())  # Print the updated recursion limit

    print("Running: ", STAGE[0])
    simulate(STAGE[0])
    reset_simulation_data()

    print("Running: ", STAGE[1])
    optimize_resources(STAGE[1])
    reset_simulation_data()

    print("Running: ", STAGE[2])
    simulate(STAGE[2])
    reset_simulation_data()

    report()


def simulate(stage):
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
        env.process(run_software_house(env, stage, num_iterations, RESOURCES))
        env.run(until=num_iterations * 7000)  # I've used num_iterations as a basic scaling factor for simulation time

        for i, phase in enumerate(phases):
            output_data[i].append(get_mean_completion_time(phase, num_iterations))

        means = [np.mean(variable_data) for variable_data in output_data]

        if is_converged(means, prev_means):
            consecutive_non_improvements += 1
            if consecutive_non_improvements >= CONVERGENCE_STREAK_THRESHOLD:
                return
        else:
            consecutive_non_improvements = 0

        prev_means = means


def report():
    project_data = []
    resources_data = []
    phases_data = []
    failures_data = []
    optimization_data = []

    for activity in optimization_log:
        optimization_data.append([
            activity.stage, activity.phase, activity.mean_wait_time, activity.iteration, activity.original_resource_value, activity.resource_value
        ])

    for activity in history_projects_log:
        project_completion = "NO" if activity.end_time == 0 else "YES"
        project_data.append([
            activity.stage, activity.software_house_id, activity.id, activity.start_time,
            activity.end_time, activity.duration, activity.project_scale, project_completion
        ])

    for activity in history_resources_log:
        resources_data.append([
            activity.stage, activity.type_of_resource, activity.software_house_id, activity.project_id, activity.resources_available,
            activity.action, activity.timestamp, activity.number_of_resources_request, activity.project_scale
        ])

    for activity in history_phases_log:
        phases_data.append([
            activity.stage, activity.phase, activity.phase_start, activity.phase_end, activity.phase_duration, activity.software_house_id,
            activity.project_id, activity.project_scale, activity.timestamp, activity.resources_obtain_time
        ])

    for activity in history_failures_log:
        failures_data.append([
            activity.stage, activity.current_phase, activity.fail_to_phase, activity.software_house_id, activity.project_id,
            activity.timestamp, activity.project_scale
        ])

    optimization_df = pd.DataFrame(optimization_data, columns=[
        "Stage", "Phase", "Previous Mean Wait Time", "Iteration", "Original Number of Resources", "Number of Resources"
    ])

    project_df = pd.DataFrame(project_data, columns=[
        "Stage", "Iteration", "Project Id", "Project Start Time", "Project End Time", "Duration", "Project Scale", "Project Completion"
    ])

    resources_df = pd.DataFrame(resources_data, columns=[
        "Stage", "Resource Type", "Iteration", "Project Id", "Resources Available", "Action", "Timestamp",
        "Resources Requested", "Project Scale"
    ])

    phases_df = pd.DataFrame(phases_data, columns=[
        "Stage", "Phase", "Phase Start", "Phase End", "Phase Duration", "Iteration", "Project Id", "Project Scale",
        "Timestamp", "Wait Time to Obtain Resources"
    ])

    failures_df = pd.DataFrame(failures_data, columns=[
        "Stage", "Current Phase", "Fail to Phase", "Iteration", "Project Id", "Timestamp", "Project Scale"
    ])

    project_df.to_csv("project_data.csv", index=False, mode='w')
    resources_df.to_csv("resources_report.csv", index=False, mode='w')
    phases_df.to_csv("phases_report.csv", index=False, mode='w')
    failures_df.to_csv("failures_report.csv", index=False, mode='w')
    optimization_df.to_csv("optimization.csv", index=False, mode='w')



if __name__ == '__main__':
    main()
