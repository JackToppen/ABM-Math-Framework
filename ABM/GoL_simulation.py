import math
import cv2
import csv
import numpy as np

from simulation import Simulation
from backend import *


class GoLSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self, name, output_path):
        # initialize the Simulation object
        Simulation.__init__(self, name, output_path)

        # hold main output directory path
        self.output_path = output_path

        # get the following from the commandline
        self.search_radius = commandline_param("-r", int)
        self.kill_below = commandline_param("-kb", int)
        self.kill_above = commandline_param("-ka", int)
        self.num_to_start = commandline_param("-c", int)
        self.move_value = commandline_param("-move", float)
        self.hatch_lower = commandline_param("-hl", int)
        self.hatch_upper = commandline_param("-hu", int)

    def agent_initials(self):
        """ Overrides the agent_initials() method from the Simulation class.
        """
        # add agents to the simulation
        self.add_agents(self.num_to_start)

        # create the following agent arrays with initial conditions.
        self.agent_array("locations", override=np.random.rand(self.number_agents, 3) * self.size)
        self.agent_array("radii", func=lambda: 0.5)

        # create graph for holding agent neighbors
        self.agent_graph("neighbor_graph")

    def steps(self):
        """ Overrides the steps() method from the Simulation class.
        """
        # record initial count of agents
        self.agent_count_csv()

        # iterate over all steps specified
        for self.current_step in range(self.beginning_step, self.end_step + 1):
            # records step run time and prints the current step and number of agents
            self.info()

            # get all neighbors within radius of 2
            self.get_neighbors("neighbor_graph", self.search_radius)

            # call the following methods that update agent values
            self.update()
            self.reproduce()
            self.move()

            # save multiple forms of information about the simulation at the current step
            self.step_values()
            self.agent_count_csv()
            # self.step_image()
            # self.temp()
            # self.data()

        # ends the simulation by creating a video from all of the step images
        # self.create_video()
        print("Done!")

    @record_time
    def update(self):
        """ Updates an agent based on the presence of neighbors.
        """
        # create boolean array to mark agents to be removed
        agents_to_remove = np.zeros(self.number_agents, dtype=bool)

        # determine which agents are being removed
        for index in range(self.number_agents):
            # get number of neighbors
            count = self.neighbor_graph.num_neighbors(index)

            # if meeting the conditions, remove the agent
            if count < self.kill_below or count > self.kill_above:
                agents_to_remove[index] = 1

        # get indices of agents to remove with a Boolean mask and count how many removed
        indices = np.arange(self.number_agents)[agents_to_remove]
        num_removed = len(indices)

        # go through the agent arrays and remove the indices
        for name in self.agent_array_names:
            # if the array is 1-dimensional, otherwise 2-dimensional
            if self.__dict__[name].ndim == 1:
                self.__dict__[name] = np.delete(self.__dict__[name], indices)
            else:
                self.__dict__[name] = np.delete(self.__dict__[name], indices, axis=0)

        # remove the indices from each graph
        for graph_name in self.graph_names:
            self.__dict__[graph_name].delete_vertices(indices)

        # change total number of agents and print to terminal
        self.number_agents -= num_removed
        print("\tRemoved " + str(num_removed) + " agents")

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
        # create boolean array to mark hatching agents
        agents_to_hatch = np.zeros(self.number_agents, dtype=bool)

        # determine which agents are hatching
        for index in range(self.number_agents):
            # get number of neighbors
            count = self.neighbor_graph.num_neighbors(index)

            # if in bounds of hatch thresholds
            if self.hatch_lower < count < self.hatch_upper:
                agents_to_hatch[index] = 1

        # get indices of the hatching agents with Boolean mask and count how many added
        indices = np.arange(self.number_agents)[agents_to_hatch]
        num_added = len(indices)

        # go through the agent arrays and add indices
        for name in self.agent_array_names:
            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][indices]

            # add the copies to the end of the array, handle if the array is 1-dimensional or 2-dimensional
            if self.__dict__[name].ndim == 1:
                self.__dict__[name] = np.concatenate((self.__dict__[name], copies))
            else:
                self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

        # go through each graph, adding one new vertex at a time
        for graph_name in self.graph_names:
            self.__dict__[graph_name].add_vertices(num_added)

        # change total number of agents and print to terminal
        self.number_agents += num_added
        print("\tAdded " + str(num_added) + " agents")

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
    def agent_count_csv(self):
        """ Output the total number of agents as a row in a
            running CSV file.
        """
        # get file name and open the file
        file_name = f"{self.name}_alive.csv"
        with open(self.output_path + file_name, "a", newline="") as file_object:
            # create CSV object
            csv_object = csv.writer(file_object)

            # write the row with the corresponding values
            csv_object.writerow([self.number_agents])

    @classmethod
    def start(cls):
        """ Configures/runs the model based on the specified
            simulation mode.
        """
        output_dir = check_output_dir()  # read paths.yaml to get/make the output directory
        name, mode = get_name_mode()  # get the name/mode of the simulation

        # new simulation
        if mode == 0:
            # check that new simulation can be made
            name = check_new_sim(name, output_dir)

            # create simulation object
            sim = cls(name, output_dir)

            # add agent arrays to object and run the simulation steps
            sim.agent_initials()
            sim.steps()

        # previous simulation
        else:
            # check that previous simulation exists
            name = check_previous_sim(name, output_dir)

            # continuation
            if mode == 1:
                # load previous simulation object from pickled file
                file_name = output_dir + name + os.sep + name + "_temp.pkl"
                with open(file_name, "rb") as file:
                    sim = pickle.load(file)

                # update the following
                sim.beginning_step = sim.current_step + 1  # update starting step
                sim.end_step = get_final_step()  # update final step

                # run the simulation steps
                sim.steps()

            # images to video
            elif mode == 2:
                # create instance for video/path information and make video
                sim = cls(name, output_dir)
                sim.create_video()

            # zip simulation output
            elif mode == 3:
                # zip a copy of the folder and save it to the output directory
                print("Compressing \"" + name + "\" simulation...")
                shutil.make_archive(output_dir + name, "zip", root_dir=output_dir, base_dir=name)
                print("Done!")
