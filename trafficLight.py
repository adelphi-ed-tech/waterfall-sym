import simpy
import random
import statistics


def traffic_light(env):
    while True:
        print ("Light turned GREEN on at t=", env.now)
        yield env.timeout(30)
        print ("Light turned YELLOW on at t=", env.now)
        yield env.timeout(5)
        print ("Light turned RED on at t=", env.now)
        yield env.timeout(20)


def main():
    env = simpy.Environment()
    env.process(traffic_light(env))
    env.run(until=120)
    print ("Done!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
