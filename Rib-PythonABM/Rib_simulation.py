import random as r
import numpy as np
import math

from pythonabm import Simulation, record_time, normal_vector


class RibSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """

    def __init__(self):
        # initialize the Simulation object
        Simulation.__init__(self)

        # read parameters from YAML file and add them to instance variables
        self.yaml_parameters("templates\\general.yaml")

        self.prolifratemult = 0
        self.pRed1 = 0
        self.shhC = 0
        self.pBlue1 = 0
        self.celldeathmult = 0
        self.localfate = 0

        self.patches_shhC = np.zeros((17, 68))
        self.patches_intens = np.zeros((17, 68))
        self.patches_pressure = np.zeros((17, 68))
        self.patches_vx = np.zeros((17, 68))
        self.patches_vy = np.zeros((17, 68))
        self.patches_redness = np.zeros((17, 68))
        self.patches_blueness = np.zeros((17, 68))

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation
        self.add_agents(self.num_to_start)

        # create the following agent arrays with initial conditions.
        self.indicate_arrays("locations", "radii", "colors", "states")
        self.locations = np.random.rand(self.number_agents, 3) * self.size
        self.radii = self.agent_array(initial=lambda: 0.5)
        self.colors = np.full((self.number_agents, 3), np.array([255, 255, 255]), dtype=int)
        self.states = self.agent_array(initial=lambda: 0)    # 0: yellow, 1: red, 2: blue

        # create graph for holding agent neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # record initial values
        self.step_values(arrays=["locations"])

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # get all neighbors within radius of 2
        self.get_neighbors(self.neighbor_graph, 5)

        # call the following methods that update agent values
        self.decide_cells()

        # add/remove agents from the simulation
        self.update_populations()

        # save multiple forms of information about the simulation at the current step
        self.step_values(arrays=["locations"])
        self.step_image()
        self.temp()
        self.data()

    def end(self):
        """ Overrides the default end method in the Simulation class.
        """
        # make a video from all of the step images
        self.create_video()

    @record_time
    def decide_cells(self):
        for index in range(self.number_agents):
            # see if state is yellow
            if self.states[index] == 0:
                if r.random() < self.prolifratemult * 0.05:
                    if r.random() < self.pRed1 * self.shhC:
                        self.states[index] = 1
                    else:
                        if r.random() < self.pBlue1 * (1 - self.shhC):
                            self.states[index] = 2
                    self.mark_to_hatch(index)
            else:
                if r.random() < self.prolifratemult * 0.05:
                    self.mark_to_hatch(index)
            if r.random() < 0.05 * self.celldeathmult * math.e ** ((self.current_step / self.end_step) ** 2):
                self.mark_to_remove(index)
            if self.localfate:
                if self.states[index] == 1 and (self.blueness / (self.blueness + self.redness) > 0.6):
                    self.states = 2
                    self.mark_to_hatch(index)
                if self.states[index] == 2 and (self.blueness / (self.blueness + self.redness) > 0.6):
                    self.states = 2
                    self.mark_to_hatch(index)

    @record_time
    def move_cells(self):
        for index in range(self.number_agents):
            if self.pressure > 4:
                vec = np.array([0.1 - self.vx, -self.vy, 0])
                norm = normal_vector(vec)
                self.locations[index] += norm * (0.5 * r.random() + 0.5) * math.sqrt(self.vx ** 2 + self.vy ** 2)

            # jiggle cells (do later)

    @record_time
    def update_fields(self):
        self.patches_redness[:] = 3
        self.patches_blueness[:] = 3

        for index in range(self.number_agents):
            # pressure
            x = int(self.locations[index][0])
            y = int(self.locations[index][1])
            self.patches_pressure[y][x] += 1

            # redness
            if self.states[index] == 1:
                self.patches_redness[y][x] += 1
            elif self.states[index] == 2:
                self.patches_blueness[y][x] += 1

            # diffuse pressure 0.5 ten times
            # diffuse redness 0.2 twice
            # diffuse blueness 0.2 twice

            # calculate differential


    @record_time
    def update_populations(self):
        """ Overrides default update_populations method from
            Simulation class.
        """
        # get indices of hatching/dying agents with Boolean mask
        add_indices = np.arange(self.number_agents)[self.hatching]
        remove_indices = np.arange(self.number_agents)[self.removing]

        # count how many added/removed agents
        num_added = len(add_indices)
        num_removed = len(remove_indices)

        # go through each agent array name
        for name in self.array_names:
            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][add_indices]

            # hatch the new agents in radius 1 from reproducing agent
            if name == "locations":
                for index in range(len(copies)):
                    # get new location position
                    new_location = copies[index] + 1 * self.random_vector()

                    # check that the new location is within the space, otherwise use boundary values
                    for i in range(3):
                        if new_location[i] > self.size[i]:
                            copies[index][i] = self.size[i]
                        elif new_location[i] < 0:
                            copies[index][i] = 0
                        else:
                            copies[index][i] = new_location[i]

            # add/remove agent data to/from the arrays
            self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added - num_removed
        print("\tAdded " + str(num_added) + " agents")
        print("\tRemoved " + str(num_removed) + " agents")

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    @classmethod
    def simulation_mode_0(cls, name, output_dir):
        """ Override the default mode 0 method from the simulation
            class.
        """
        # make simulation instance, update name, and add paths
        sim = cls()
        sim.name = name
        sim.set_paths(output_dir)

        # set up the simulation agents and run the simulation
        sim.full_setup()
        sim.run_simulation()
