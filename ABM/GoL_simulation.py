import math
import cv2
import csv
import numpy as np

from pythonabm import Simulation
from pythonabm import *

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
        self.reg_pop = np.zeros((0, 1), dtype=float)

        # The following variable tracks when the population reaches the last step or its asymptote
        # (whichever comes first) in the case that the population is decreasing exponentially
        self.reg_track = self.end_step

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation
        self.add_agents(self.num_to_start)

        # create the following agent arrays with initial conditions.
        self.agent_array("locations", override=np.random.rand(self.number_agents, 3) * self.size)
        self.agent_array("radii", func=lambda: 0.5)

        # create graph for holding agent neighbors
        self.agent_graph("neighbor_graph")

        # record initial values
        self.step_values()
        self.agent_count()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # records step run time and prints the current step and number of agents
        self.info()

        # get all neighbors within radius of 2
        self.get_neighbors("neighbor_graph", self.search_radius)

        # call the following methods that update agent values
        self.update()
        self.reproduce()
        self.move()

        # add/remove agents from the simulation
        self.update_populations()

        # save multiple forms of information about the simulation at the current step
        self.step_values()
        self.agent_count()

    def regression(self):
        """ Performs linear regression on the exp(n) or log(n) vs time, where n
            is the number of living agents at time step t (held in reg_pop)
        """

        # Threshold for linear regression
        reg_thresh = 0.9

        # If the number of living cells increases from step 0 to step 1, we will compute 
        # linear regression using exp(n). Otherwise, we will use log(n) instead

        if self.reg_pop[0] < self.reg_pop[1]:
            # Note: we decrease the values for exp(n) by the starting population so that
            # computing the exp(n) is less prone to cause overflow issues
            y_adjPop = np.exp((self.reg_pop) - self.reg_pop[0])

            x_time = np.array(range(0,self.reg_pop.shape[0])).reshape((-1,1))

            # Creates linear regression object from scikit-learn
            regr = linear_model.LinearRegression()

            # Performs linear regression computation
            regr.fit(x_time, y_adjPop)

            # Generates the exp(n) predicted from the linear regression object
            y_pred = regr.predict(x_time)

            # Generates the coefficient of determination (R^2) for the linear regression
            coef = r2_score(y_adjPop,y_pred)

            # Checks to see if R^2 is above the threshold and names the file accordingly
            if coef >= reg_thresh:
                file_name = f"{self.name}_reg-exp.csv"
            else:
                file_name = f"{self.name}_reg-other.csv"

            with open(self.main_path[:-(len(self.name)) - 1] + file_name, "a", newline="") as file_object:
            
                # create CSV object
                csv_object = csv.writer(file_object)
                csv_object.writerow(["exp",R2])  

        else: 
            # Note: if the population is exponentially decreasing, we only find log(n)
            # up until the end step or the step when the population reaches its asymptote
            # (i.e. reg_track), whichever comes first
            y_adjPop = np.log((self.reg_pop[0:self.reg_track]))

            # Creates an array of time step values up to reg_track, which is either the end
            # step for the simulation or step in which an exponentially decreasing population
            # n reached its asymptote. 

            x_time = np.array(range(0,self.reg_track)).reshape((-1,1))

            # Creates linear regression object from scikit-learn
            regr1 = linear_model.LinearRegression()

            # Performs linear regression computation
            regr1.fit(x_time, y_adjPop)

            # Generates the exp(n) or log(n) predicted from the linear regression object
            y_pred1 = regr1.predict(x_time)

            # Generates the coefficient of determination (R^2) for the linear regression
            coef1 = r2_score(y_adjPop,y_pred1)

            # If the asymptote is encountered before the last step, a second linear regression
            # will be performed on the remaining population values to verify that the population
            # exponentially decreases

            if self.reg_track < self.reg_pop.shape[0]:

                x_rem = np.array(range(self.reg_track,self.reg_pop.shape[0]-1)).reshape((-1,1))

                y_rem = self.reg_pop[self.reg_track:-1]
                print(y_rem)
                
                # Creates linear regression object from scikit-learn
                regr2 = linear_model.LinearRegression()

                # Performs linear regression computation
                regr2.fit(x_rem, y_rem)

                # Generates the exp(n) or log(n) predicted from the linear regression object
                y_pred2 = regr2.predict(x_rem)

                # Generates the coefficient of determination (R^2) for the linear regression
                coef2 = r2_score(y_rem,y_pred2)

                # Checks to see if both R^2 values are above the threshold and names the file accordingly
                if coef1 >= reg_thresh and coef2 >= reg_thresh:
                    file_name = f"{self.name}_reg-log.csv"
                else:
                    file_name = f"{self.name}_reg-other.csv"

                with open(self.main_path[:-(len(self.name)) - 1] + file_name, "a", newline="") as file_object:
            
                    # create CSV object
                    csv_object = csv.writer(file_object)
                    csv_object.writerow(["log",coef1])
                    csv_object.writerow(["asymp",coef2])

            # Otherwise, only the original linear regression on log(n) needs to be checked and recorded
            else:

            # Checks to see if the R^2 is above the threshold and names the file accordingly
                if coef1 >= reg_thresh:
                    file_name = f"{self.name}_reg-log.csv"
                else:
                    file_name = f"{self.name}_reg-other.csv"

                with open(self.main_path[:-(len(self.name)) - 1] + file_name, "a", newline="") as file_object:
            
                    # create CSV object
                    csv_object = csv.writer(file_object)
                    csv_object.writerow(["log",coef1])

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
    def step_image(self, background=(0, 0, 0), origin_bottom=True):
        """ Creates an image of the simulation space. Note the imaging library
            OpenCV uses BGR instead of RGB.

            - background: the color of the background image as BGR
            - origin_bottom: location of origin True -> bottom/left, False -> top/left
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            check_direct(self.images_path)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            image[:, :] = background

            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major = int(scale * self.radii[index])
                minor = int(scale * self.radii[index])

                # draw the agent and a black outline to distinguish overlapping agents
                color = (255, 255, 255)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

    @record_time
    def agent_count(self):
        """ Output the total number of agents as a row in a running CSV file and in the reg_pop
            array for linear regression at the end of the simulation
        """

        # get file name and open the file
        file_name = f"{self.name}_alive-pop.csv"
        with open(self.main_path[:-(len(self.name)) - 1] + file_name, "a", newline="") as file_object:
            # create CSV object
            csv_object = csv.writer(file_object)

            #  Values outputted to CSV
            csv_object.writerow([self.number_agents])

        # The following conditional catches the time step at which a population that is exponentially decreasing
        # reaches is asymptote and records this time step in reg_track

        if (self.reg_pop.shape[0] > 0) and (self.reg_track == self.end_step):
                
            if self.number_agents - self.reg_pop[-1] >= 0:

                self.reg_track = self.current_step - 1

        # Values outputted to array reg_pop
        self.reg_pop = np.append(self.reg_pop,[[self.number_agents]])


    @classmethod
    def start(cls, output_dir):
        """ Configures/runs the model based on the specified
            simulation mode.
        """
        # check that the output directory exists and get the starting parameters for the model
        output_dir = check_output_dir(output_dir)
        name, mode, final_step = starting_params()

        # new simulation
        if mode == 0:
            # first check that new simulation can be made and create simulation output directory
            name = check_existing(name, output_dir, new_simulation=True)

            # now make simulation instance, update name, and add paths
            sim = cls()
            sim.name = name
            sim.set_paths(output_dir)

            # copy model files to simulation directory, ignoring __pycache__ files
            # direc_path = sim.main_path + name + "_copy"
            # shutil.copytree(os.getcwd(), direc_path, ignore=shutil.ignore_patterns("__pycache__"))

            # set up the simulation, run the steps, and create a video from any images
            sim.setup()
            for sim.current_step in range(1, sim.end_step + 1):
                sim.step()

            # If all of the agents did not die after the first step, performs linear regression
            if sim.reg_pop.shape[0] > 1:
                sim.regression()
            
            sim.create_video()

        # previous simulation
        else:
            # check that previous simulation exists
            name = check_existing(name, output_dir, new_simulation=False)

            # continuation
            if mode == 1:
                # load previous simulation object from pickled file
                file_name = output_dir + name + os.sep + name + "_temp.pkl"
                with open(file_name, "rb") as file:
                    sim = pickle.load(file)

                # update paths for the case the simulation is move to new folder
                sim.set_paths(output_dir)

                # iterate through all steps and create a video from any images
                for sim.current_step in range(sim.current_step + 1, final_step + 1):
                    sim.step()

                # Computing linear regression should also be added in this mode

                sim.create_video()

            # images to video
            elif mode == 2:
                # make object for video/path information and create video
                sim = cls()
                sim.name = name
                sim.set_paths(output_dir)
                sim.create_video()

            # zip simulation output
            elif mode == 3:
                # zip a copy of the folder and save it to the output directory
                print("Compressing \"" + name + "\" simulation...")
                shutil.make_archive(output_dir + name, "zip", root_dir=output_dir, base_dir=name)
                print("Done!")
