import math
import cv2
import numpy as np

from simulation import Simulation
from backend import record_time, check_direct


class ExampleSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self, name, output_path):
        # initialize the Simulation object
        Simulation.__init__(self, name, output_path)

    def agent_initials(self):
        """ Overrides the agent_initials() method from the Simulation class.
        """
        # add agents to the simulation
        self.add_agents(self.num_to_start)

        # create the following agent arrays with initial conditions.
        self.agent_array("locations", override=np.random.rand(self.number_agents, 3) * self.size)
        self.agent_array("radii", func=lambda: 0.5)
        self.agent_array("states", func=lambda: True)

        # create graph for holding cell neighbors
        self.agent_graph("neighbor_graph")

    def steps(self):
        """ Overrides the steps() method from the Simulation class.
        """
        # if True, record starting values/image for the simulation
        if self.record_initial_step:
            self.record_initials()

        # iterate over all steps specified
        for self.current_step in range(self.beginning_step, self.end_step + 1):
            # records step run time and prints the current step and number of agents
            self.info()

            # get all neighbors within radius of 2
            self.get_neighbors("neighbor_graph", 2)

            # call the following methods that update agent values
            self.update()
            self.reproduce()
            self.move()

            # save multiple forms of information about the simulation at the current step
            self.step_image()
            self.step_values()
            self.temp()
            self.data()

        # ends the simulation by creating a video from all of the step images
        self.create_video()

    @record_time
    def update(self):
        """ Updates an agent based on the presence of neighbors.
        """
        for index in range(self.number_agents):
            # get neighbors and make variable to hold the number of alive neighbors
            neighbors = self.neighbor_graph.neighbors(index)
            count = 0

            # count how many alive neighbors
            for neighbor in neighbors:
                if self.states[neighbor]:
                    count += 1

            # if no neighbors or more than 3, die
            if count == 0 or count >= 3:
                self.states[index] = False

    @record_time
    def move(self):
        """ Assigns new location to agent.
        """
        for index in range(self.number_agents):
            # get new location vector
            new_location = self.locations[index] + self.random_vector()

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
        # create boolean array to mark hatching agents
        agents_to_hatch = np.zeros(self.number_agents, dtype=bool)

        # determine which agents are hatching
        for index in range(self.number_agents):
            # get neighbors and make variable to hold the number of alive neighbors
            neighbors = self.neighbor_graph.neighbors(index)
            count = 0

            # count how many alive neighbors
            for neighbor in neighbors:
                if self.states[neighbor]:
                    count += 1

            # if alive and exactly 2 neighbors, hatch new agent
            if self.states[index] and count == 2:
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

                if self.states[index]:
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
