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
        """ Performs linear regression on the log(n) or exp(n) vs time, where n
            is the number of living agents at time step t (held in reg_pop)
        """
        # Threshold for linear regression
        reg_thresh = 0.9

        # Placeholder for tracking whether exp(n) would yield an overflow error
        exp_track = False

        # We exclude the initial number of agents from our regression computation
        self.reg_pop = np.delete(self.reg_pop,0)

        # First, we try to transform the data using log(n)

        y_adjPop = np.log((self.reg_pop))
        
        # Placeholder array for the time steps

        x_time = np.array(range(1,y_adjPop.shape[0]+1)).reshape((-1,1))

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
                # y_adjPop2 = np.exp(self.reg_pop - self.reg_pop[0])
                y_adjPop2 = np.exp(self.reg_pop - (np.amax(self.reg_pop)+np.amin(self.reg_pop))/2)

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

        with open(self.main_path[:-(len(self.name)) - 1] + file_name, "a", newline="") as file_object:
            
            # create CSV object
            csv_object = csv.writer(file_object)
            # record population transformation log(n) and its associated R^2
            csv_object.writerow(["log",coef_log])
        
            # If an exp(n) could be computed, we record this value as well
            if coef_log < reg_thresh:

                if not exp_track:
                    # record population transformation exp(n) and its associated R^2
                    csv_object.writerow(["exp",coef_exp])
                
                else:
                    # record overflow
                    csv_object.writerow(["exp","nan"])

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

        # The following conditional catches the case that the population decreases to 0 and stops updating
        # the population array reg_pop accordingly.

        if self.number_agents != 0:

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
