# Arttu Hyvönen 11/2017

import numpy as np
import random
import pygame
from scipy.special import expit

pygame.init()


# World properties

# Size of map
map_size = 50
# Max time one generation can spend
time_limit = 2000
# regrowth speed
reg_speed = 0.01


# Generation cycle properties

# Number of creatures in generation
creatures = 100
# How many creatures are saved from each generation for base of next
ancestors = 20
# How many random new creatures per genration
new_random_creatures = 40
# How many child for each ancestor
remainder = (creatures - new_random_creatures) % ancestors
new_random_creatures += remainder
childs = (creatures - new_random_creatures) / ancestors

max_generations = 100

# How many pixels is one side of tile in map
window_scale = 10
# pygame window
screen = pygame.display.set_mode((map_size*window_scale,
                                  map_size*window_scale))


# Energy properties

# movement energy penalty
movement_energy = 0.03
# eating energy penalty
eating_energy = 0.05
# eating speed (%)
eating_speed = 0.25
# energy passed to child (%)
child_energy = 0.25
# energy lost in reproducing
reproduce_energy = 0.2
# energy lost by default
standing_energy = 0.03


# Creature properties

# neurons in hidden layer
hid_neurons = 40
# lenght of visual
sight = 2
# Mutation rate
mr = 1
# Weight magnitude
wm = 10
# actions: move: 0 up, 1 down, 2 left, 3 right,
# 4 eat, 5 reproduce
output_neurons = 6




class World:

    def __init__(self, size):
        self.size = size
        self.values = np.random.random((self.size, self.size))

    def update(self, regrowth):
        # making food grow
        g = np.ones((self.size, self.size))
        n = g * regrowth
        self.values = np.clip(np.add(self.values, n), 0, 1)


class Network:

    def __init__(self, neurons_1, neurons_2, neurons_3,
                 w1=None, w2=None, b1=None, b2=None):
        # checking if there's parent and copying parent's network
        if w1 is not None:
            self.w1 = np.clip(w1 + np.random.uniform(-mr, mr, (neurons_2, neurons_1)), -wm, wm)
            self.w2 = np.clip(w2 + np.random.uniform(-mr, mr, (neurons_3, neurons_2)), -wm, wm)
            self.b1 = np.clip(b1 + np.random.uniform(-mr, mr, neurons_2), -wm, wm)
            self.b2 = np.clip(b2 + np.random.uniform(-mr, mr, neurons_3), -wm, wm)
            return

        # making new network, if no parents
        self.w1 = np.random.randint(-wm, wm, size=(neurons_2, neurons_1))
        self.w2 = np.random.randint(-wm, wm, size=(neurons_3, neurons_2))
        self.b1 = np.random.randint(-wm, wm, size=neurons_2)
        self.b2 = np.random.randint(-wm, wm, size=neurons_3)

    def output(self, a0):
        # gettin second layer from input layer
        # a1 = sigmoid(weights_1 * a0 + bias_1)
        a1 = expit(np.dot(self.w1, a0) + self.b1)
        # output layer from second layer
        # output = sigmoid(weights_2 * a1 + bias_2)
        return expit(np.dot(self.w2, a1) + self.b2)


class Organism:

    born = 0
    died = 0
    lifespan = 0
    brain = None

    def __init__(self, place, brain_neurons, sight,
                 energy=1, w1=None, w2=None, b1=None, b2=None):
        self.energy = energy
        self.place = place
        self.brain_neurons = brain_neurons
        self.sight = sight
        in_sight = (2*sight+1)*(2*sight+1)
        self.input_neurons = in_sight + 1
        # checking if there's parent, and copying parent's brains
        if w1 is not None:
            self.brain = Network(self.input_neurons, brain_neurons,
                                 output_neurons, w1, w2, b1, b2)
        else:
            # or making new brains
            self.brain = Network(self.input_neurons,
                                 brain_neurons, output_neurons)

    def action(self, world, t):
        # getting values for actions
        action_val = self.brain.output(np.insert(self.visual_input(world), 0, self.energy))
        # if multiple highest values, picking one at random
        action = random.choice(np.where(action_val == action_val.max()))

        action_result = 0

        if self.energy <= 0:
            return "dead"

        # some how random.choice doesn't work all the time,
        # so killing anyone who's causing bugs
        if action.size > 1:
            return "dead"
        # doing actions
        elif int(action) == 0:
            # move up
            if self.place[1] > 0:
                self.place[1] -= 1
                self.energy -= movement_energy
        elif int(action) == 1:
            # move down
            if self.place[1] < world.size - 1:
                self.place[1] += 1
                self.energy -= movement_energy
        elif int(action) == 2:
            # move left
            if self.place[0] > 0:
                self.place[0] -= 1
                self.energy -= movement_energy
        elif int(action) == 3:
            # move right
            if self.place[0] < world.size - 1:
                self.place[0] += 1
                self.energy -= movement_energy
        elif int(action) == 4:
            # eat
            self.energy += world.values[self.place[0], self.place[1]]*eating_speed
            action_result = -world.values[self.place[0], self.place[1]]*eating_speed
            self.energy -= eating_energy
        elif int(action) == 5:
            # reproduce
            self.energy -= self.energy*child_energy
            action_result = Organism(self.place, self.brain_neurons, self.sight,
                                     energy=self.energy*child_energy-reproduce_energy,
                                     w1=self.brain.w1, w2=self.brain.w2,
                                     b1=self.brain.b1, b2=self.brain.b2)
            action_result.born = t
            self.energy -= reproduce_energy
        else:
            self.energy -= movement_energy

        self.energy -= standing_energy
        # returnin action and its result if there's one
        return [action_result, int(action)]

    def visual_input(self, world):
        # defining area of sight
        x0, y0 = self.place[0]-self.sight, self.place[1]-self.sight
        size = 2*self.sight+1
        visual = np.empty((size, size))
        # getting values of food in area
        for i in range(size):
            for j in range(size):
                if x0+i < 0 or x0+i > world.size - 1 or y0+j < 0 or y0+j > world.size - 1:
                    visual[i, j] = 0
                else:
                    visual[i, j] = world.values[x0+i, y0+j]

        # making data one dimensional array
        return visual.flatten()


class Game:

    def __init__(self, world_size, num_of_organisms, ancestor_orgs=None):
        # making world and organisms
        self.world = World(world_size)
        self.organisms = []
        if ancestor_orgs is not None:
            self.new_generation(ancestor_orgs)
        else:
            for i in range(num_of_organisms):
                self.organisms.append(Organism(np.random.randint(0, world_size-1, 2), hid_neurons, sight))

    def simulate(self, max_time):

        # elapsed iterations
        t = 0
        # making lists for best creatures in this generation
        best = []
        lifespans = [0]

        # simulation loop
        while t < max_time:
            # update food
            self.world.update(reg_speed)
            new_born = []

            # doing organism actions
            for o in self.organisms:
                result = o.action(self.world, t)
                if result == "dead":
                    o.died = t
                    o.lifespan = o.died-o.born
                    if o.lifespan > min(lifespans):
                        best.insert(0, o)
                        lifespans.insert(0, o.lifespan)
                        if len(best) > ancestors:
                            del lifespans[-1]
                            del best[-1]
                        self.organisms.remove(o)
                elif result[1] == 4:
                    self.world.values[o.place[0], o.place[1]] += result[0]
                elif result[1] == 5:
                    new_born.append(result[0])

            # limiting food values between 0 and 1 after eating
            self.world.values = np.clip(self.world.values, 0, 1)
            # birthing new organisms
            self.organisms += new_born

            t += 1
            print("time spend: "+str(t))

            # making visual map and updating it to screen
            self.visual_map()
            pygame.display.update()

            # checking if anyone is alive
            if self.organisms == []:
                print("Everyone died.  :(")
                break
        print(lifespans)
        return best

    def new_generation(self, ancestor_orgs):

        for o in ancestor_orgs:
            self.organisms.append(Organism(np.random.randint(0, world_size-1, 2), hid_neurons, sight,
                                           w1=o.brain.w1, w2=o.brain.w2, b1=o.brain.b1, b2=o.brain.b2))

    def visual_map(self):

        # Map
        for i in range(map_size):
            for j in range(map_size):
                value = int(self.world.values[i, j]*255)
                pygame.draw.rect(screen, (255-value, 236, 31, 255),
                                 (i*window_scale, j*window_scale,
                                  window_scale, window_scale), 0)

        # Organisms
        for o in self.organisms:
            pygame.draw.rect(screen, (0, 0, 255, 255),
                             (o.place[0]*window_scale, o.place[1]*window_scale,
                              window_scale, window_scale), 2)

def generation_cycle():

    g = 0

    ancestor_orgs = None
    
    while g < max_generations:

        game = Game(map_size, creatures, ancestor_orgs=ancestor_orgs)
        ancestor_orgs = game.simulate(time_limit)

        g += 1




if __name__ == "__main__":
    generation_cycle()
