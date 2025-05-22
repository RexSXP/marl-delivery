import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

class Robot: 
    def __init__(self, position): 
        self.position = position
        self.carrying = 0

class Package: 
    def __init__(self, start, start_time, target, deadline, package_id): 
        self.start = start
        self.start_time = start_time
        self.target = target
        self.deadline = deadline
        self.package_id = package_id
        self.status = 'None' # Possible statuses: 'waiting', 'in_transit', 'delivered'

class Environment: 

    def __init__(self, map_file, max_time_steps = 100, n_robots = 5, n_packages=20,
             move_cost=-0.01, delivery_reward=10., delay_reward=1., 
             seed=2025): 
        """ Initializes the simulation environment. :param map_file: Path to the map text file. :param move_cost: Cost incurred when a robot moves (LRUD). :param delivery_reward: Reward for delivering a package on time. """ 
        self.map_file = map_file
        self.grid = self.load_map()
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0]) if self.grid else 0 
        self.move_cost = move_cost 
        self.delivery_reward = delivery_reward 
        self.delay_reward = delay_reward
        self.t = 0 
        self.robots = [] # List of Robot objects.
        self.packages = [] # List of Package objects.
        self.total_reward = 0

        self.n_robots = n_robots
        self.max_time_steps = max_time_steps
        self.n_packages = n_packages

        self.rng = np.random.RandomState(seed)
        self.reset()
        self.done = False
        self.state = None
        
        # For visualization
        self.frames = []
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def load_map(self):
        """
        Reads the map file and returns a 2D grid.
        Assumes that each line in the file contains numbers separated by space.
        0 indicates free cell and 1 indicates an obstacle.
        """
        grid = []
        with open(self.map_file, 'r') as f:
            for line in f:
                # Strip line breaks and split into numbers
                row = [int(x) for x in line.strip().split(' ')]
                grid.append(row)
        return grid
    
    def is_free_cell(self, position):
        """
        Checks if the cell at the given position is free (0) or occupied (1).
        :param position: Tuple (row, column) to check.
        :return: True if the cell is free, False otherwise.
        """
        r, c = position
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return self.grid[r][c] == 0

    def add_robot(self, position):
        """
        Adds a robot at the given position if the cell is free.
        :param position: Tuple (row, column) for the robot's starting location.
        """
        if self.is_free_cell(position):
            robot = Robot(position)
            self.robots.append(robot)
        else:
            raise ValueError("Invalid robot position: must be on a free cell not occupied by an obstacle or another robot.")

    def reset(self):
        """
        Resets the environment to its initial state.
        Clears all robots and packages, and reinitializes the grid.
        """
        self.t = 0
        self.robots = []
        self.packages = []
        self.total_reward = 0
        self.done = False
        self.state = None

        # Reinitialize the grid
        #self.grid = self.load_map(sel)
        # Add robots and packages
        tmp_grid = np.array(self.grid)
        for i in range(self.n_robots):
            # Randomly select a free cell for the robot
            position, tmp_grid = self.get_random_free_cell(tmp_grid)
            self.add_robot(position)
        
        N = self.n_rows
        list_packages = []
        for i in range(self.n_packages):
            # Randomly select a free cell for the package
            start = self.get_random_free_cell_p()
            while True:
                target = self.get_random_free_cell_p()
                if start != target:
                    break
            
            to_deadline = 10 + self.rng.randint(N/2, 3*N)
            if i <= min(self.n_robots, 20):
                start_time = 0
            else:
                start_time = self.rng.randint(1, self.max_time_steps)
            list_packages.append((start_time, start, target, start_time + to_deadline ))

        list_packages.sort(key=lambda x: x[0])
        for i in range(self.n_packages):
            start_time, start, target, deadline = list_packages[i]
            package_id = i+1
            self.packages.append(Package(start, start_time, target, deadline, package_id))

        return self.get_state()
    
    def get_state(self):
        """
        Returns the current state of the environment.
        The state includes the positions of robots and packages.
        :return: State representation.
        """
        selected_packages = []
        for i in range(len(self.packages)):
            if self.packages[i].start_time == self.t:
                selected_packages.append(self.packages[i])
                self.packages[i].status = 'waiting'

        state = {
            'time_step': self.t,
            'map': self.grid,
            'robots': [(robot.position[0] + 1, robot.position[1] + 1,
                        robot.carrying) for robot in self.robots],
            'packages': [(package.package_id, package.start[0] + 1, package.start[1] + 1, 
                          package.target[0] + 1, package.target[1] + 1, package.start_time, package.deadline) for package in selected_packages]
        }
        return state
        

    def get_random_free_cell_p(self):
        """
        Returns a random free cell in the grid.
        :return: Tuple (row, col) of a free cell.
        """
        free_cells = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) \
                      if self.grid[i][j] == 0]
        i = self.rng.randint(0, len(free_cells))
        return free_cells[i]


    def get_random_free_cell(self, new_grid):
        """
        Returns a random free cell in the grid.
        :return: Tuple (row, col) of a free cell.
        """
        free_cells = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) \
                      if new_grid[i][j] == 0]
        i = self.rng.randint(0, len(free_cells))
        new_grid[free_cells[i][0]][free_cells[i][1]] = 1
        return free_cells[i], new_grid

    
    def step(self, actions):
        """
        Advances the simulation by one timestep.
        :param actions: A list where each element is a tuple (move_action, package_action) for a robot.
            move_action: one of 'S', 'L', 'R', 'U', 'D'.
            package_action: '1' (pickup), '2' (drop), or '0' (do nothing).
        :return: The updated state and total accumulated reward.
        """
        r = 0
        if len(actions) != len(self.robots):
            raise ValueError("The number of actions must match the number of robots.")

        #print("Package env: ")
        #print([p.status for p in self.packages])

        # -------- Process Movement --------
        proposed_positions = []
        # For each robot, compute the new position based on the movement action.
        old_pos = {}
        next_pos = {}
        for i, robot in enumerate(self.robots):
            move, pkg_act = actions[i]
            new_pos = self.compute_new_position(robot.position, move)
            # Check if the new position is valid (inside bounds and not an obstacle).
            if not self.valid_position(new_pos):
                new_pos = robot.position  # Invalid moves result in no change.
            proposed_positions.append(new_pos)
            old_pos[robot.position] = i
            next_pos[new_pos] = i

        moved_robots = [0 for _ in range(len(self.robots))]
        computed_moved = [0 for _ in range(len(self.robots))]
        final_positions = [None] * len(self.robots)
        occupied = {}  # Dictionary to record occupied cells.
        while True:
            updated = False
            for i in range(len(self.robots)):
            
                if computed_moved[i] != 0: 
                    continue

                pos = self.robots[i].position
                new_pos = proposed_positions[i]
                can_move = False
                if new_pos not in old_pos:
                    can_move = True
                else:
                    j = old_pos[new_pos]
                    if (j != i) and (computed_moved[j] == 0): # We must wait for the conflict resolve
                        continue
                    # We can decide where the robot can go now
                    can_move = True

                if can_move:
                    # print("Updated: ", i, new_pos)
                    if new_pos not in occupied:
                        occupied[new_pos] = i
                        final_positions[i] = new_pos
                        computed_moved[i] = 1
                        moved_robots[i] = 1
                        updated = True
                    else:
                        new_pos = pos
                        occupied[new_pos] = i
                        final_positions[i] = pos
                        computed_moved[i] = 1
                        moved_robots[i] = 0
                        updated = True

                if updated:
                    break

            if not updated:
                break
        #print("Computed postions: ", final_positions)
        for i in range(len(self.robots)):
            if computed_moved[i] == 0:
                final_positions[i] = self.robots[i].position 
        
        # Update robot positions and apply movement cost when applicable.
        for i, robot in enumerate(self.robots):
            move, pkg_act = actions[i]
            if move in ['L', 'R', 'U', 'D'] and final_positions[i] != robot.position:
                r += self.move_cost
            robot.position = final_positions[i]

        # -------- Process Package Actions --------
        for i, robot in enumerate(self.robots):
            move, pkg_act = actions[i]
            #print(i, move, pkg_act)
            # Pick up action.
            if pkg_act == '1':
                if robot.carrying == 0:
                    # Check for available packages at the current cell.
                    for j in range(len(self.packages)):
                        if self.packages[j].status == 'waiting' and self.packages[j].start == robot.position and self.packages[j].start_time <= self.t:
                            # Pick the package with the smallest package_id.
                            package_id = self.packages[j].package_id
                            robot.carrying = package_id
                            self.packages[j].status = 'in_transit'
                            # print(package_id, 'in transit')
                            break

            # Drop action.
            elif pkg_act == '2':
                if robot.carrying != 0:
                    package_id = robot.carrying
                    target = self.packages[package_id - 1].target
                    # Check if the robot is at the target position.
                    if robot.position == target:
                        # Update package status to delivered.
                        pkg = self.packages[package_id - 1]
                        pkg.status = 'delivered'
                        # Apply reward based on whether the delivery is on time.
                        if self.t <= pkg.deadline:
                            r += self.delivery_reward
                        else:
                            # Example: a reduced reward for late delivery.
                            r += self.delay_reward
                        robot.carrying = 0  
        
        # Increment the simulation timestep.
        self.t += 1

        self.total_reward += r

        done = False
        infos = {}
        if self.check_terminate():
            done = True
            infos['total_reward'] = self.total_reward
            infos['total_time_steps'] = self.t

        return self.get_state(), r, done, infos
    
    def check_terminate(self):
        if self.t == self.max_time_steps:
            return True
        
        for p in self.packages:
            if p.status != 'delivered':
                return False
            
        return True

    def compute_new_position(self, position, move):
        """
        Computes the intended new position for a robot given its current position and move command.
        """
        r, c = position
        if move == 'S':
            return (r, c)
        elif move == 'L':
            return (r, c - 1)
        elif move == 'R':
            return (r, c + 1)
        elif move == 'U':
            return (r - 1, c)
        elif move == 'D':
            return (r + 1, c)
        else:
            return (r, c)

    def valid_position(self, pos):
        """
        Checks if the new position is within the grid and not an obstacle.
        """
        r, c = pos
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        if self.grid[r][c] == 1:
            return False
        return True

    def render(self, save_frame=False):
        """
        Visualizes the environment using matplotlib, including robots, packages, and targets.
        Optionally saves the frame.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Define colors for visualization elements
        cmap_colors = ['white', 'black'] # 0: Free, 1: Obstacle
        cmap = ListedColormap(cmap_colors)
        bounds = [-0.5, 0.5, 1.5]
        norm = plt.Normalize(bounds[0], bounds[-1])

        # Display the grid map
        ax.imshow(self.grid, cmap=cmap, norm=norm, extent=[-0.5, self.n_cols - 0.5, self.n_rows - 0.5, -0.5])
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, self.n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.n_rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Plot robots
        robot_colors = plt.cm.get_cmap('tab10', self.n_robots)
        for i, robot in enumerate(self.robots):
            ax.plot(robot.position[1], robot.position[0], 'o', color=robot_colors(i), markersize=15, label=f'Robot {i}')
            ax.text(robot.position[1], robot.position[0], f'R{i}', 
                    fontsize=10, ha='center', va='center', color='white', weight='bold')
        
        # Plot packages and targets
        package_colors = plt.cm.get_cmap('Set2', self.n_packages)
        for pkg in self.packages:
            if pkg.status != 'delivered':
                # Plot package start location (waiting or in_transit)
                ax.plot(pkg.start[1], pkg.start[0], 's', color=package_colors(pkg.package_id - 1), markersize=10, label=f'Package {pkg.package_id}')
                ax.text(pkg.start[1], pkg.start[0], f'P{pkg.package_id}', 
                        fontsize=8, ha='center', va='center', color='black')

            if pkg.status != 'delivered':
                 # Plot package target location
                ax.plot(pkg.target[1], pkg.target[0], 'X', color=package_colors(pkg.package_id - 1), markersize=10, label=f'Target {pkg.package_id}')
                ax.text(pkg.target[1], pkg.target[0], f'T{pkg.package_id}', 
                        fontsize=8, ha='center', va='center', color='black')

        # Set axis labels and title
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title('Multi-Agent Package Delivery Simulation')
        ax.set_aspect('equal', adjustable='box')

        # Add time step and total reward text boxes
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, f'Time Step: {self.t}\nTotal Reward: {self.total_reward:.2f}', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        # Create custom legend elements
        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='Wall'),
            Patch(facecolor='white', edgecolor='black', label='Free Space'),
        ]
        # Add robot legend entries (using dummy lines)
        for i in range(self.n_robots):
             legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Robot {i}', 
                                   markerfacecolor=robot_colors(i), markersize=10))
        # Add package legend entries (using dummy lines)
        for i in range(self.n_packages):
             legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label=f'Package {i+1} Start', 
                                   markerfacecolor=package_colors(i), markersize=8))
             legend_elements.append(plt.Line2D([0], [0], marker='X', color='w', label=f'Package {i+1} Target', 
                                   markerfacecolor=package_colors(i), markersize=8))

        # Place legend outside the plot
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        if save_frame:
            # Save the frame
            frame_path = os.path.join(self.results_dir, f"frame_{self.t:04d}.png")
            plt.savefig(frame_path, bbox_inches='tight', dpi=100)
            self.frames.append(frame_path)
        
        plt.close(fig)
        
        # Also print text-based information
        print(f"\nTime step: {self.t}")
        print(f"Total reward: {self.total_reward:.2f}")
        print("\nRobot positions and packages:")
        for i, robot in enumerate(self.robots):
            print(f"Robot {i}: Position {robot.position}, Carrying package {robot.carrying}")
        print("\nUndelivered packages:")
        for pkg in self.packages:
            if pkg.status != 'delivered':
                print(f"Package {pkg.package_id}: Status={pkg.status}, Start={pkg.start}, Target={pkg.target}")
        print("-" * 50)

    def save_gif(self, filename="simulation.gif"):
        """
        Saves the collected frames as a GIF animation.
        """
        import imageio
        frames = []
        for frame_path in self.frames:
            frames.append(imageio.imread(frame_path))
        
        gif_path = os.path.join(self.results_dir, filename)
        imageio.mimsave(gif_path, frames, fps=5)
        
        # Clean up individual frames
        for frame_path in self.frames:
            os.remove(frame_path)
        
        print(f"\nSimulation GIF saved to: {gif_path}")
        return gif_path

if __name__=="__main__":
    env = Environment('map.txt', 10, 2, 5)
    state = env.reset()
    # print("Initial State:", state)
    # print("Initial State:")
    env.render()

    # Agents
    # Initialize agents
    from greedyagent import GreedyAgents as Agents
    agents = Agents()   # You should define a default parameters here
    agents.init_agents(state) # You have a change to init states which can be used or not. Depend on your choice
    print("Agents initialized.")
    
    # Example actions for robots
    list_actions = ['S', 'L', 'R', 'U', 'D']
    n_robots = len(state['robots'])
    done = False
    t = 0
    while not done:
        actions = agents.get_actions(state) 
        state, reward, done, infos = env.step(actions)
    
        print("\nState after step:")
        env.render()
        # print(f"Reward: {reward}, Done: {done}, Infos: {infos}")
        # print("Total Reward:", env.total_reward)
        # print("Time step:", env.t)
        # print("Packages:", state['packages'])
        # print("Robots:", state['robots'])

        # For debug purpose
        t += 1
        if t == 100:
            break