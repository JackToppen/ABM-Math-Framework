import random as r
import numpy as np
import math
import cv2
from numba import njit, stencil

from pythonabm import Simulation, record_time, normal_vector, check_direct


@stencil
def laplacian(c):
    return 0.125 * (c[0,1] + c[1,0] + c[0,-1] + c[-1,0] + c[1,1] + c[1,-1] + c[-1,1] + c[-1,-1])


@njit
def diffuse_numba(pressure, redness, blueness):
    for _ in range(10):
        # mirror first and last rows
        pressure[0] = pressure[1]
        pressure[-1] = pressure[-2]

        # mirror first and last columns
        pressure[:, 0] = pressure[:, 1]
        pressure[:, -1] = pressure[:, -2]
        pressure += 0.5 * laplacian(pressure) - 0.5 * pressure
        
    for _ in range(2):
        redness += 0.2 * laplacian(redness)
        blueness += 0.2 * laplacian(blueness)

    return pressure, redness, blueness


def diffuse(pressure, redness, blueness):
    # pad array edges with zeros
    pressure_pad = np.pad(pressure, 1)
    redness_pad = np.pad(redness, 1)
    blueness_pad = np.pad(blueness, 1)

    # perform integration calculation
    p_out, r_out, b_out = diffuse_numba(pressure_pad, redness_pad, blueness_pad)

    # return array without edges
    return p_out[1:-1, 1:-1], r_out[1:-1, 1:-1], b_out[1:-1, 1:-1]


class RibSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """

    def __init__(self):
        # initialize the Simulation object
        Simulation.__init__(self)

        # read parameters from YAML file and add them to instance variables
        self.yaml_parameters("templates/general.yaml")

        # simulation parameters
        self.init_size_mult = 0.55
        self.shh_xport = 12.0
        self.shh_intensity_log = 0.63
        self.pRed1 = 0.4
        self.pBlue1 = 0.4
        self.celldeathmult = 0.00
        self.prolifratemult = 1.0
        self.cdduration = 30
        self.localfate = True
       
        # patch values
        self.shhC = np.zeros((17, 68))
        self.intens = np.zeros((17, 68))
        self.pressure = np.zeros((17, 68))
        self.vx = np.zeros((17, 68))
        self.vy = np.zeros((17, 68))
        self.redness = np.zeros((17, 68))
        self.blueness = np.zeros((17, 68))

        self.video_quality = 2010

    def setup(self):
        """ Matches the "setup" method from the NetLogo Rib-ABM
        """
        # add agents to the simulation
        num_to_start = int(self.init_size_mult * 1200)
        self.add_agents(num_to_start)

        # specify arrays
        self.indicate_arrays("locations", "radii", "colors", "states")

        # create the following agent arrays with initial conditions.
        self.locations = np.random.rand(num_to_start, 3)
        self.radii = self.agent_array(initial=lambda: 0.25)
        self.states = self.agent_array(initial=lambda: 0)
        
        # set shh-intensity
        shh_intensity = math.e ** self.shh_intensity_log
        for i in range(68):
            self.shhC[:, i] = shh_intensity * math.e ** (-1 * (((i + 8) / self.shh_xport) ** 2)) / math.e ** (-1 * (10 / self.shh_xport) ** 2)
        self.intens = 255 * (1 - self.shhC / shh_intensity)

        # adjust locations based on dimensions of space
        self.locations[:, 0] = 14 * math.sqrt(self.init_size_mult) * self.locations[:, 0] + 2
        self.locations[:, 1] = math.sqrt(self.init_size_mult) * (14 * self.locations[:, 1] - 7) + 8

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Matches the "go" method from the NetLogo Rib-ABM
        """
        # call the following methods that update cell and field values
        self.decide_cells()
        self.update_fields()

        # incrementally move cells and update fields
        countsteps = 0
        while np.amax(self.pressure) > 6 and countsteps < 100:
            self.move_cells()
            self.update_fields()
            countsteps += 1

        # add/remove agents from the simulation
        self.update_populations()

        # save multiple forms of information about the simulation at the current step
        self.step_values(arrays=["locations"])
        self.step_image()
        self.temp()
        self.data()

    def end(self):
        """ Make a video from all of the step images
        """
        self.create_video()

    def get_patch_location(self, index):
        """ Return patch indices based on cell location
        """
        return int(self.locations[index][1] + 0.5), int(self.locations[index][0] + 0.5)

    @record_time
    def decide_cells(self):
        # loop over all cells
        for index in range(self.number_agents):
            # get location of patch cell is on
            i, j = self.get_patch_location(index)

            # determine if dividing
            if r.random() < self.prolifratemult * 0.05:
                # see if state is yellow
                if self.states[index] == 0:
                    # determine new state
                    if r.random() < self.pRed1 * self.shhC[i][j]:
                        self.states[index] = 1
                    else:
                        if r.random() < self.pBlue1 * (1 - self.shhC[i][j]):
                            self.states[index] = 2

                # hatch
                self.mark_to_hatch(index)
            
            # determine if dying
            if r.random() < 0.05 * self.celldeathmult * math.e ** ((self.current_step / self.end_step) ** 2):
                self.mark_to_remove(index)

            # if local fate is turned on, potentially change fates
            if self.localfate and (self.blueness[i][j] + self.redness[i][j]) != 0:
                if self.states[index] == 1 and (self.blueness[i][j] / (self.blueness[i][j] + self.redness[i][j]) > 0.6):
                    self.states[index] = 2

                if self.states[index] == 2 and (self.redness[i][j] / (self.blueness[i][j] + self.redness[i][j]) > 0.6):
                    self.states[index] = 1

    @record_time
    def move_cells(self):
        """ Matches the "move-cells" method from the NetLogo Rib-ABM
        """
        # loop over all cells
        for index in range(self.number_agents):
            # get current patch location
            i, j = self.get_patch_location(index)

            # if patch pressure is greater than 4, move in random direction
            if self.pressure[i][j] > 4:
                vec = np.array([-self.vy[i][j], 0.1-self.vx[i][j], 0])
                norm = normal_vector(vec)
                self.locations[index] += norm * (0.5 * r.random() + 0.5) * math.sqrt(self.vx[i][j] ** 2 + self.vy[i][j] ** 2)

        # move cells slightly
        self.jiggle_turtles(0.5)

    def jiggle_turtles(self, jsize):
        """ Matches the "move-cells" method from the NetLogo Rib-ABM
        """
        # loop over all cells
        for index in range(self.number_agents):
            self.locations[index, 0] = min(66.5, max(1.5, self.locations[index, 0] + r.gauss(0, jsize)))
            self.locations[index, 1] = min(15.5, max(0.5, self.locations[index, 1] + r.gauss(0, jsize)))

            if self.locations[index, 1] > 15:
                self.locations[index, 1] = 15 - r.expovariate(1)

            if self.locations[index, 1] < 1:
                self.locations[index, 1] = 1 + r.expovariate(1)

            if self.locations[index, 0] < 2:
                self.locations[index, 0] = 2 + r.expovariate(3)

            if self.locations[index, 0] > 66:
                self.locations[index, 0] = 66 - r.expovariate(3)

    @record_time
    def update_fields(self):
        # set base value for pressure, redness, and blueness
        self.pressure[:] = 0
        self.redness[:] = 3
        self.blueness[:] = 3

        # loop over all cells
        for index in range(self.number_agents):
            # add pressure of cell to current patch
            i, j = self.get_patch_location(index)
            self.pressure[i][j] += 1

            # increase either redness or blueness density to patch based on cell state
            if self.states[index] == 1:
                self.redness[i][j] += 1
            elif self.states[index] == 2:
                self.blueness[i][j] += 1

        # call diffusion method for patch values
        self.pressure, self.redness, self.blueness = diffuse(self.pressure, self.redness, self.blueness)

        # calculate differential
        for i in range(68):
            for j in range(17):
                if j < 2 or j == 16:
                    self.vx[j][i] = 0
                else:
                    self.vx[j][i] = (self.pressure[j+1][i] - self.pressure[j-1][i]) / 2

                if i == 0 or i == 67:
                    self.vy[j][i] = 0
                else:
                    self.vy[j][i] = (self.pressure[j][i+1] - self.pressure[j][i-1]) / 2

    @record_time
    def step_image(self, background=(0, 0, 0), origin_bottom=True):
        """ Creates an image of the simulation space.
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            check_direct(self.images_path)

            # scale image size based on environment
            scale = 60

            # make sure first two columns are black, make image background based on SHH intensity
            self.intens[:, 0:2] = 0
            image = cv2.resize(np.floor(self.intens), scale * np.array(self.size[:2]), interpolation=cv2.INTER_AREA)
            image = np.dstack((image, image, image))

            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates and the axis lengths
                x, y = int(scale * (self.locations[index][0])), int(scale * (self.locations[index][1]))
                major, minor = int(scale * self.radii[index]), int(scale * self.radii[index])

                # get color of cell based on state
                if self.states[index] == 0:
                    color = (0, 255, 255)
                elif self.states[index] == 1:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

