import simpy
import random
import statistics
import sympy
from sympy.stats import Normal
import csv

SIM_TIME = 1500
NEW_PROJECT_LOW_TIME = 30
MEW_PROJECT_HIGH_TIME = 40
NEW_PROJECT_MODE_TIME = 35

# resources
BUSINESS_ANALYSTS = 5
DESIGNERS = 5
PROGRAMMERS = 10
TESTERS = 20
MAINTENANCE_PEOPLE = 5

BUSINESS_ANALYSTS_SMALL_PROJECT = 1
BUSINESS_ANALYSTS_MEDIUM_PROJECT = 2
BUSINESS_ANALYSTS_LARGE_PROJECT = 5
DESIGNERS_SMALL_PROJECT = 1
DESIGNERS_MEDIUM_PROJECT = 2
DESIGNERS_LARGE_PROJECT = 5
PROGRAMMERS_SMALL_PROJECT = 2
PROGRAMMERS_MEDIUM_PROJECT = 4
PROGRAMMERS_LARGE_PROJECT = 10
TESTERS_SMALL_PROJECT = 2
TESTERS_MEDIUM_PROJECT = 6
TESTERS_LARGE_PROJECT = 20
MAINTENANCE_PEOPLE_SMALL_PROJECT = 1
MAINTENANCE_PEOPLE_MEDIUM_PROJECT = 2
MAINTENANCE_PEOPLE_LARGE_PROJECT = 5

# definitions
smallProject = 70
mediumProject = 25
largeProject = 5
smallProjectErrorProbability = 10
mediumProjectErrorProbability = 20
largeProjectProbability = 30

TOTAL_PROJECTS = 50
list_of_projects = []
failures_log = []
resources_wait_time_log = []
resources_log = []
phases_log = []
failure_log = []

class SoftwareHouse(object):
    def __init__(self, env, id_input, number_business_analyst, number_designers, number_programmers, number_testers, number_maintenance_people):
        self.env = env
        self.id = id_input
        self.business_analysts = simpy.Container(env, init=number_business_analyst)
        self.designers = simpy.Container(env, init=number_designers)
        self.programmers = simpy.Container(env, init=number_programmers)
        self.testers = simpy.Container(env, init=number_testers)
        self.maintenance_people = simpy.Container(env, init=number_maintenance_people)


def run_software_house(env, id_input, number_business_analyst, number_designers, number_programmers, number_testers, number_maintenance_people):
    software_house = SoftwareHouse(env, id_input, number_business_analyst, number_designers, number_programmers, number_testers, number_maintenance_people)
    project_number = 0
    while True:
        new_project_wait_time = random.triangular(NEW_PROJECT_LOW_TIME, MEW_PROJECT_HIGH_TIME, NEW_PROJECT_MODE_TIME)
        yield env.timeout(new_project_wait_time)  # wait a bit
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
        self.project_scale = 0 # 0 - not set; 1 - small; 2 - medium; 3 - large
        self.design_phase_failures = 0
        self.implementation_phase_failures = 0
        self.testing_phase_failures = 0
        self.maintenance_phase_failures = 0
        self.compute_project_scale()

    def compute_project_scale(self):
        option = round(random.uniform(1, 100))
        if 1 <= option <= smallProject:
            self.project_scale = 1
        elif smallProject < option <= smallProject + mediumProject:
            self.project_scale = 2
        elif smallProject + mediumProject < option <= smallProject + mediumProject + largeProject:
            self.project_scale = 3

    def get_failure_probability(self):
        failure_probability = 0
        if self.project_scale == 1:
            failure_probability = smallProjectErrorProbability
        elif self.project_scale == 2:
            failure_probability = mediumProjectErrorProbability
        elif self.project_scale == 3:
            failure_probability = largeProjectProbability
        return failure_probability

    def requirements_phase(self):
        requirements_phase_start = self.software_house.env.now
        number_of_resources = 0
        if self.project_scale == 1:
            number_of_resources = BUSINESS_ANALYSTS_SMALL_PROJECT
        elif self.project_scale == 2:
            number_of_resources = BUSINESS_ANALYSTS_MEDIUM_PROJECT
        elif self.project_scale == 3:
            number_of_resources = BUSINESS_ANALYSTS_LARGE_PROJECT

        analysts_request_time = self.software_house.env.now
        resources_log.append(ResourceReport("business analysts", self.software_house.id, self.id, self.software_house.business_analysts.level, "request", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.business_analysts.get(number_of_resources)
        analysts_obtain_time = self.software_house.env.now
        resources_log.append(ResourceReport("business analysts", self.software_house.id, self.id, self.software_house.business_analysts.level, "obtain", self.software_house.env.now, number_of_resources, self.project_scale))
        value = round(random.uniform(3, 5))
        yield self.software_house.env.timeout(value)
        resources_log.append(ResourceReport("business analysts", self.software_house.id, self.id, self.software_house.business_analysts.level, "release", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.business_analysts.put(number_of_resources)
        resources_log.append(ResourceReport("business analysts", self.software_house.id, self.id, self.software_house.business_analysts.level, "after release", self.software_house.env.now, number_of_resources, self.project_scale))
        requirements_phase_end = self.software_house.env.now
        time_to_obtain_analysts_resources = analysts_obtain_time - analysts_request_time
        time_for_requirements_phase = requirements_phase_end - requirements_phase_start
        phases_log.append(PhasesReport("requirements", requirements_phase_start, requirements_phase_end, time_for_requirements_phase, self.software_house.id, self.id, self.project_scale, self.software_house.env.now, time_to_obtain_analysts_resources))

    def design_phase(self):
        design_phase_start = self.software_house.env.now
        number_of_resources = 0
        if self.project_scale == 1:
            number_of_resources = DESIGNERS_SMALL_PROJECT
        elif self.project_scale == 2:
            number_of_resources = DESIGNERS_MEDIUM_PROJECT
        elif self.project_scale == 3:
            number_of_resources = DESIGNERS_LARGE_PROJECT

        designers_request_time = self.software_house.env.now
        resources_log.append(ResourceReport("designers", self.software_house.id, self.id, self.software_house.designers.level, "request", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.designers.get(number_of_resources)
        designers_obtain_time = self.software_house.env.now
        resources_log.append(ResourceReport("designers", self.software_house.id, self.id, self.software_house.designers.level, "obtain", self.software_house.env.now, number_of_resources, self.project_scale))
        value = round(random.uniform(5, 10))
        yield self.software_house.env.timeout(value)
        resources_log.append(ResourceReport("designers", self.software_house.id, self.id, self.software_house.designers.level, "release", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.designers.put(number_of_resources)
        resources_log.append(ResourceReport("designers", self.software_house.id, self.id, self.software_house.designers.level, "after release", self.software_house.env.now, number_of_resources, self.project_scale))
        option = round(random.uniform(1, 100))
        design_phase_end = self.software_house.env.now
        time_to_obtain_designers_resources = designers_obtain_time - designers_request_time
        time_for_design_phase = design_phase_end - design_phase_start
        phases_log.append(PhasesReport("design", design_phase_start, design_phase_end, time_for_design_phase, self.software_house.id, self.id, self.project_scale, self.software_house.env.now, time_to_obtain_designers_resources))
        if 1 <= option <= self.get_failure_probability():
            self.design_phase_failures += 1
            failures_log.append(FailureReport("design phase", "requirements phase", self.software_house.id, self.id, self.software_house.env.now, self.project_scale))
            yield self.software_house.env.process(self.requirements_phase())

    def implementation_phase(self):
        implementation_phase_start = self.software_house.env.now
        number_of_resources = 0
        if self.project_scale == 1:
            number_of_resources = PROGRAMMERS_SMALL_PROJECT
        elif self.project_scale == 2:
            number_of_resources = PROGRAMMERS_MEDIUM_PROJECT
        elif self.project_scale == 3:
            number_of_resources = PROGRAMMERS_LARGE_PROJECT

        programmers_request_time = self.software_house.env.now
        resources_log.append(ResourceReport("programmers", self.software_house.id, self.id, self.software_house.programmers.level, "request", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.programmers.get(number_of_resources)
        programmers_obtain_time = self.software_house.env.now
        resources_log.append(ResourceReport("programmers", self.software_house.id, self.id, self.software_house.programmers.level, "obtain", self.software_house.env.now, number_of_resources, self.project_scale))
        value = round(random.uniform(15, 20))
        yield self.software_house.env.timeout(value)
        resources_log.append(ResourceReport("programmers", self.software_house.id, self.id, self.software_house.programmers.level, "release", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.programmers.put(number_of_resources)
        resources_log.append(ResourceReport("programmers", self.software_house.id, self.id, self.software_house.programmers.level, "after release", self.software_house.env.now, number_of_resources, self.project_scale))
        option = round(random.uniform(1, 100))
        implementation_phase_end = self.software_house.env.now
        time_to_obtain_programmers_resources = programmers_obtain_time - programmers_request_time
        time_for_implementation_phase = implementation_phase_end - implementation_phase_start
        phases_log.append(PhasesReport("implementation", implementation_phase_start, implementation_phase_end, time_for_implementation_phase, self.software_house.id, self.id, self.project_scale, self.software_house.env.now, time_to_obtain_programmers_resources))
        if 1 <= option <= self.get_failure_probability():
            self.implementation_phase_failures += 1
            failures_log.append(FailureReport("design phase", "requirements phase", self.software_house.id, self.id, self.software_house.env.now, self.project_scale))
            yield self.software_house.env.process(self.design_phase())

    def testing_phase(self):
        testing_phase_start = self.software_house.env.now
        number_of_resources = 0
        if self.project_scale == 1:
            number_of_resources = TESTERS_SMALL_PROJECT
        elif self.project_scale == 2:
            number_of_resources = TESTERS_MEDIUM_PROJECT
        elif self.project_scale == 3:
            number_of_resources = TESTERS_LARGE_PROJECT

        testers_request_time = self.software_house.env.now
        resources_log.append(ResourceReport("testers", self.software_house.id, self.id, self.software_house.testers.level, "request", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.testers.get(number_of_resources)
        testers_obtain_time = self.software_house.env.now
        resources_log.append(ResourceReport("testers", self.software_house.id, self.id, self.software_house.testers.level, "obtain", self.software_house.env.now, number_of_resources, self.project_scale))
        value = round(random.uniform(5, 10))
        yield self.software_house.env.timeout(value)
        resources_log.append(ResourceReport("testers", self.software_house.id, self.id, self.software_house.testers.level, "release", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.testers.put(number_of_resources)
        resources_log.append(ResourceReport("testers", self.software_house.id, self.id, self.software_house.testers.level, "after release", self.software_house.env.now, number_of_resources, self.project_scale))
        option = round(random.uniform(1, 100))
        testing_phase_end = self.software_house.env.now
        time_to_obtain_testers_resources = testers_obtain_time - testers_request_time
        time_for_testing_phase = testing_phase_end - testing_phase_start
        phases_log.append(PhasesReport("testing", testing_phase_start, testing_phase_end, time_for_testing_phase, self.software_house.id, self.id, self.project_scale, self.software_house.env.now, time_to_obtain_testers_resources))
        if 1 <= option <= self.get_failure_probability():
            self.testing_phase_failures += 1
            failures_log.append(FailureReport("design phase", "requirements phase", self.software_house.id, self.id, self.software_house.env.now, self.project_scale))
            yield self.software_house.env.process(self.implementation_phase())

    def maintenance_phase(self):
        maintenance_phase_start = self.software_house.env.now
        number_of_resources = 0
        if self.project_scale == 1:
            number_of_resources = MAINTENANCE_PEOPLE_SMALL_PROJECT
        elif self.project_scale == 2:
            number_of_resources = MAINTENANCE_PEOPLE_MEDIUM_PROJECT
        elif self.project_scale == 3:
            number_of_resources = MAINTENANCE_PEOPLE_LARGE_PROJECT

        maintenance_people_request_time = self.software_house.env.now
        resources_log.append(ResourceReport("maintenance people", self.software_house.id, self.id, self.software_house.maintenance_people.level, "request", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.maintenance_people.get(number_of_resources)
        maintenance_people_obtain_time = self.software_house.env.now
        resources_log.append(ResourceReport("maintenance people", self.software_house.id, self.id, self.software_house.maintenance_people.level, "obtain", self.software_house.env.now, number_of_resources, self.project_scale))
        value = round(random.uniform(1, 3))
        yield self.software_house.env.timeout(value)
        resources_log.append(ResourceReport("maintenance people", self.software_house.id, self.id, self.software_house.maintenance_people.level, "release", self.software_house.env.now, number_of_resources, self.project_scale))
        yield self.software_house.maintenance_people.put(number_of_resources)
        resources_log.append(ResourceReport("maintenance people", self.software_house.id, self.id, self.software_house.maintenance_people.level, "after release", self.software_house.env.now, number_of_resources, self.project_scale))
        option = round(random.uniform(1, 100))
        maintenance_phase_end = self.software_house.env.now
        time_to_obtain_maintenance_people_resources = maintenance_people_obtain_time - maintenance_people_request_time
        time_for_maintenance_phase = maintenance_phase_end - maintenance_phase_start
        phases_log.append(PhasesReport("maintenance", maintenance_phase_start, maintenance_phase_end, time_for_maintenance_phase, self.software_house.id, self.id, self.project_scale, self.software_house.env.now, time_to_obtain_maintenance_people_resources))
        if 1 <= option <= self.get_failure_probability():
            self.maintenance_phase_failures += 1
            failures_log.append(FailureReport("maintenance phase", "testing phase", self.software_house.id, self.id, self.software_house.env.now, self.project_scale))
            yield self.software_house.env.process(self.testing_phase())

    def start_project(self):
        # Project begins
        self.start_time = self.software_house.env.now

        yield self.software_house.env.process(self.requirements_phase())
        yield self.software_house.env.process(self.design_phase())
        yield self.software_house.env.process(self.implementation_phase())
        yield self.software_house.env.process(self.testing_phase())
        yield self.software_house.env.process(self.maintenance_phase())

        self.end_time = self.software_house.env.now
        self.duration = self.end_time - self.start_time
        # project ends


def main():
    for i in range(1, 6):
        random.seed(random.random())
        env = simpy.Environment()
        env.process(run_software_house(env, i, BUSINESS_ANALYSTS, DESIGNERS, PROGRAMMERS, TESTERS, MAINTENANCE_PEOPLE))
        env.run(until=SIM_TIME)
        print(f"attempt {i} done!")
        report()


def print_stats(res):
    print(f'{res.count} of {res.capacity} slots are allocated')
    print(f'   Users: {res.users}')
    print(f'   Number of Users: {len(res.users)}')
    print(f'   Queued events: {res.queue}')
    print(f'   Number of Queued events: {len(res.queue)}')


def user(res):
    print_stats(res)
    with res.request() as req:
        yield req
        print_stats(res)
    print_stats(res)


def report():
    total_time = 0
    count = 0
    no_size_project_count = []
    small_project_count = []
    medium_project_count = []
    large_project_count = []

    # get highest number of iterations and initialize lists
    highest_value = 0
    for individual_project in list_of_projects:
        if individual_project.id > highest_value:
            highest_value = individual_project.id

    small_project_count = [0] * highest_value
    medium_project_count = [0] * highest_value
    large_project_count = [0] * highest_value

    for individual_project in list_of_projects:
        total_time += (individual_project.end_time - individual_project.start_time)
        if individual_project.project_scale == 0:
            no_size_project_count[individual_project.id - 1] += 1
        elif individual_project.project_scale == 1:
            small_project_count[individual_project.id - 1] += 1
        elif individual_project.project_scale == 2:
            medium_project_count[individual_project.id - 1] += 1
        elif individual_project.project_scale == 3:
            large_project_count[individual_project.id - 1] += 1

    file = open("project_data.csv", "w")
    writer = csv.writer(file)
    writer.writerow(["Software House Id", "Project Id", "Project Start Time", "Project End Time", "Duration", "Project Scale",
                     "Number of Design Phase Failures", "Number of Implementation Phase Failures",
                     "Number of Testing Phase Failures", "Number of Maintenance Phase Failures", "Project Completion"])
    for individual_project in list_of_projects:
        project_completion = ""
        if individual_project.end_time == 0:
            project_completion = "NO"
        else:
            project_completion = "YES"
        writer.writerow([individual_project.software_house_id, individual_project.id, individual_project.start_time,
                         individual_project.end_time, individual_project.duration, individual_project.project_scale,
                         individual_project.design_phase_failures, individual_project.implementation_phase_failures,
                         individual_project.testing_phase_failures, individual_project.maintenance_phase_failures, project_completion])
    file.close()

    file = open("resources_report.csv", "w")
    writer = csv.writer(file)
    writer.writerow(["Resource Type", "Software House Id", "Project Id", "Resources Available", "Action", "Timestamp", "Resources Requested", "Project Scale"])
    for activity in resources_log:
        writer.writerow([activity.type_of_resource, activity.software_house_id, activity.project_id, activity.resources_available,
                         activity.action, activity.timestamp, activity.number_of_resources_request, activity.project_scale])
    file.close()

    file = open("phases_report.csv", "w")
    writer = csv.writer(file)
    writer.writerow(["Phase", "Phase Start", "Phase End", "Phase Duration", "Software House Id", "Project Id", "Project Scale", "Timestamp", "Wait Time to Obtain Resources"])
    for activity in phases_log:
        writer.writerow(
            [activity.phase, activity.phase_start, activity.phase_end, activity.phase_duration, activity.software_house_id, activity.project_id, activity.project_scale, activity.timestamp, activity.resources_obtain_time])
    file.close()

    file = open("failures_report.csv", "w")
    writer = csv.writer(file)
    writer.writerow(["Current Phase", "Fail to Phase", "Software House Id", "Project Id", "Timestamp", "Project Scale"])
    for activity in failures_log:
        writer.writerow(
            [activity.current_phase, activity.fail_to_phase, activity.software_house_id, activity.project_id, activity.timestamp, activity.project_scale])
    file.close()

    #average_wait = total_time / count
    #minutes, frac_minutes = divmod(average_wait, 1)
    #seconds = frac_minutes * 60
    #print("Running sim", f"is {minutes} and {seconds} ok")
    print("---------------------------------------------")
    print("Projects by size")
    #print("No Size", f" {no_size_project_count}")
    #print("Small", f" {small_project_count}")
    #print("Medium", f" {medium_project_count}")
    #print("Large", f" {large_project_count}")

    #self.design_phase_failure = 0
    #self.implementation_phase_failure = 0
    #self.testing_phase_failure = 0
    #self.maintenance_phase_failure = 0


if __name__ == '__main__':
    main()


