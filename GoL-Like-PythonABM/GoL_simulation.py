import csv
import sys
import numpy as np

from pythonabm import Simulation, commandline_param, record_time

from sklearn import linear_model
from sklearn.metrics import r2_score



class GoLSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self):
        # initialize the Simulation object
        Simulation.__init__(self)

        # read parameters from YAML file and add them to instance variables
        self.yaml_parameters("templates\\general.yaml")

        # get the following from the commandline
        self.search_radius = commandline_param("-r", int)
        self.kill_below = commandline_param("-kb", int)
        self.kill_above = commandline_param("-ka", int)
        self.num_to_start = commandline_param("-c", int)
        self.move_value = commandline_param("-move", float)
        self.hatch_lower = commandline_param("-hl", int)
        self.hatch_upper = commandline_param("-hu", int)

        # The following array holds the population n at each time step
        self.reg_pop = np.zeros(0, dtype=int)

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation
        self.add_agents(self.num_to_start)

        # create the following agent arrays with initial conditions.
        self.indicate_arrays("locations", "radii", "colors")
        self.locations = np.random.rand(self.number_agents, 3) * self.size
        self.radii = self.agent_array(initial=lambda: 0.5)
        self.colors = np.full((self.number_agents, 3), np.array([255, 255, 255]), dtype=int)

        # create graph for holding agent neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # record initial values
        self.step_values(arrays=["locations"])
        self.agent_count()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # get all neighbors within radius of 2
        self.get_neighbors(self.neighbor_graph, self.search_radius)

        # call the following methods that update agent values
        self.update()
        self.reproduce()
        self.move()

        # add/remove agents from the simulation
        self.update_populations()

        # save multiple forms of information about the simulation at the current step
        self.step_values(arrays=["locations"])
        self.agent_count()
        # self.step_image()
        # self.temp()
        # self.data()

    def end(self):
        """ Overrides the default end method in the Simulation class.
        """
        # make a video from all of the step images
        # self.create_video()

        # if all of the agents did not die after the first step, performs linear regression
        if self.reg_pop.shape[0] > 1:
            self.regression()

    @record_time
    def update(self):
        """ Updates an agent based on the presence of neighbors.
        """
        # determine which agents are being removed
        for index in range(self.number_agents):
            # get number of neighbors
            count = self.neighbor_graph.num_neighbors(index)

            # if meeting the conditions, remove the agent
            if count < self.kill_below or count > self.kill_above:
                self.mark_to_remove(index)

    @record_time
    def move(self):
        """ Assigns new location to agent.
        """
        for index in range(self.number_agents):
            # get new location position
            new_location = self.locations[index] + self.move_value * self.random_vector()

            # check that the new location is within the space, otherwise use boundary values
            for i in range(3):
                if new_location[i] > self.size[i]:
                    self.locations[index][i] = self.size[i]
                elif new_location[i] < 0:
                    self.locations[index][i] = 0
                else:
                    self.locations[index][i] = new_location[i]

    @record_time
    def reproduce(self):
        """ If the agent meets criteria, hatch a new agent.
        """
        # determine which agents are hatching
        for index in range(self.number_agents):
            # get number of neighbors
            count = self.neighbor_graph.num_neighbors(index)

            # if in bounds of hatch thresholds
            if self.hatch_lower < count < self.hatch_upper:
                self.mark_to_hatch(index)

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

    def regression(self):
        """ Performs linear regression on the log(n) or exp(n) vs time, where n
            is the number of living agents at time step t (held in reg_pop)
        """
        # Threshold for linear regression
        reg_thresh = 0.9

        # Placeholder for tracking whether exp(n) would yield an overflow error
        exp_track = False

        # We exclude the initial number of agents from our regression computation
        self.reg_pop = np.delete(self.reg_pop, 0)

        # First, we try to transform the data using log(n) and reshape to 2D array for the regression
        y_adjPop = np.log(self.reg_pop).reshape(-1, 1)
        
        # Placeholder array for the time steps, again reshape to 2D
        x_time = np.arange(1, len(y_adjPop) + 1).reshape(-1, 1)

        # Creates linear regression object from scikit-learn
        regr = linear_model.LinearRegression()

        # Performs linear regression computation
        regr.fit(x_time, y_adjPop)

        # Generates the exp(n) predicted from the linear regression object
        y_pred = regr.predict(x_time)

        # Generates the coefficient of determination (R^2) for the linear regression
        coef_log = r2_score(y_adjPop,y_pred)
        print("log(n) coefficient: ", coef_log)

        # Checks to see if R^2 is above the threshold and names the file accordingly
        if coef_log >= reg_thresh:
            file_name = f"{self.name}_reg_log.csv"

        # If R^2 is not above the threshold, we try exp(n)
        else:
            # We have to check to make sure that the range of values n is not larger than
            # what Python can handle after it is transformed into exp(n)
            if np.ptp(self.reg_pop) >= (np.log(sys.float_info.max) - np.log(sys.float_info.min)):

                # If it is, attempting to compute exp(n) will yield an overflow error. In
                # this case, we simply output the regression from log(n) in a CSV indicating
                # that the regression on exp(n) could not be computed.
                exp_track = True
                # The name of the file indicates that an overflow occurred
                file_name = f"{self.name}_reg_overflow.csv"

            else:
                # Note: we recenter the median of the values n so that python can handle 
                # computing exp(n) without causing an overflow
                # y_adjPop2 = np.exp(self.reg_pop - self.reg_pop[0]).reshape(-1, 1)
                y_adjPop2 = np.exp(self.reg_pop - (np.amax(self.reg_pop)+np.amin(self.reg_pop))/2).reshape(-1, 1)

                # Creates linear regression object from scikit-learn
                # Note that computing this regression might still cause an overflow error
                regr2 = linear_model.LinearRegression()

                # Performs linear regression computation
                regr2.fit(x_time, y_adjPop2)

                # Generates the exp(n) predicted from the linear regression object
                y_pred2 = regr2.predict(x_time)

                # Generates the coefficient of determination (R^2) for the linear regression
                coef_exp = r2_score(y_adjPop2,y_pred2)
                print("exp(n) coefficient: ", coef_exp)
                
                # Checks to see if R^2 is above the threshold and names the file accordingly
                if coef_exp >= reg_thresh:
                    file_name = f"{self.name}_reg_exp.csv"

                else:
                    # If both transformations fail, we assume that the population follows
                    # some "other" trend
                    file_name = f"{self.name}_reg_other.csv"

        with open(self.output_path + file_name, "a", newline="") as file_object:
            
            # create CSV object
            csv_object = csv.writer(file_object)
            # record population transformation log(n) and its associated R^2
            csv_object.writerow(["log", coef_log])
        
            # If an exp(n) could be computed, we record this value as well
            if coef_log < reg_thresh:

                if not exp_track:
                    # record population transformation exp(n) and its associated R^2
                    csv_object.writerow(["exp", coef_exp])
                
                else:
                    # record overflow
                    csv_object.writerow(["exp", "nan"])

    @record_time
    def agent_count(self):
        """ Output the total number of agents as a row in a running CSV file and in the reg_pop
            array for linear regression at the end of the simulation
        """

        # get file name and open the file
        file_name = f"{self.name}_alive-pop.csv"
        with open(self.output_path + file_name, "a", newline="") as file_object:
            # create CSV object
            csv_object = csv.writer(file_object)

            #  Values outputted to CSV
            csv_object.writerow([self.number_agents])

        # The following conditional catches the case that the population decreases to 0 and stops updating
        # the population array reg_pop accordingly.
        if self.number_agents != 0:

            # Values outputted to array reg_pop
            self.reg_pop = np.append(self.reg_pop, self.number_agents)

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
