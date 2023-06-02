# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import simpy
import random
import statistics

wait_times = []


class Theater(object):
    def __init__(self, env, num_cashiers):
        self.env = env
        self.cashier = simpy.Resource(env, num_cashiers)

    def purchase_ticket(self, moviegoer):
        yield self.env.timeout(random.randint(1, 3))

    def go_to_movies(env, moviegoer, theater):
        # Moviegoer arrives at the theater
        arrival_time = env.now

        with theater.cashier.request() as request:
            yield request
            yield env.process(theater.purchase_ticket(moviegoer))

        with theater.usher.request() as request:
            yield request
            yield env.process(theater.check_ticket(moviegoer))

        if random.choice([True, False]):
            with theater.server.request() as request:
                yield request
                yield env.process(theater.sell_food(moviegoer))

        # Moviegoer heads into the theater
        wait_times.append(env.now - arrival_time)

    def run_theater(env, num_cashiers, num_servers, num_ushers):
        theater = Theater(env, num_cashiers, num_servers, num_ushers)

        for moviegoer in range(3):
            env.process(theater.go_to_movies(env, moviegoer, theater))

        while True:
            yield env.timeout(0.20)  # Wait a bit before generating a new person

            moviegoer += 1
            env.process(theater.go_to_movies(env, moviegoer, theater))


def get_user_input():
    num_cashiers = input("Input # of cashiers working: ")
    num_servers = input("Input # of servers working: ")
    num_ushers = input("Input # of ushers working: ")
    params = [num_cashiers, num_servers, num_ushers]
    if all(str(i).isdigit() for i in params):  # Check input is valid
        params = [int(x) for x in params]
    else:
        print(
            "Could not parse input. The simulation will use default values:",
            "\n1 cashier, 1 server, 1 usher.",
        )
        params = [1, 1, 1]
    return params


def get_average_wait_time(wait_times_b):
    average_wait = statistics.mean(wait_times_b)
    return average_wait


def calculate_wait_time(wait_times_a):
    average_wait = get_average_wait_time(wait_times_a)
    # Pretty print the results
    minutes, fraction_minutes = divmod(average_wait, 1)
    seconds = fraction_minutes * 60
    return round(minutes), round(seconds)


def main():
    # Setup
    random.seed(42)
    num_cashiers, num_servers, num_ushers = get_user_input()

    # Run the simulation
    env = simpy.Environment()

    # env.process(run_theater(env, num_cashiers, num_servers, num_ushers))
    env.run(until=90)

    # View the results
    #minutes, secs = calculate_wait_time(wait_times)
    #print("Running simulation...", "\nThe average wait time is ", minutes, "minutes and ", secs, " seconds.")
    print (get_average_wait_time(wait_times))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
