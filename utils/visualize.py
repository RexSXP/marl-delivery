import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.widgets import Button, Slider
import matplotlib as mpl
import time
from collections import defaultdict

class DeliveryVisualizer:
    def __init__(self, env):
        self.env = env
        self.map = np.array(env.grid)
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.robot_markers = []
        self.package_markers = {}  # Track all packages by ID
        self.time_text = None
        self.reward_text = None
        self.state_history = []  # Store all states for playback
        self.reward_history = []  
        
        self.colors = {
            'wall': 'black',
            'free': 'white',
            'robot': 'blue',
            'package_waiting': 'green',
            'package_transit': 'orange',
            'package_delivered': 'red',
            'target': 'purple'
        }
        
        # Add animation control variables
        self.animation_speed = 0.001  # seconds between frames
        self.animation_running = True
        self.pause_button = None
        self.speed_slider = None
        
        # Initialize the visualization
        self.setup_plot()
        self.setup_controls()
        
    def setup_plot(self):
        """Initialize the plot with the map"""
        # Create colormap for the grid
        cmap = ListedColormap(['white', 'black'])
        
        # Plot the grid
        self.ax.imshow(self.map, cmap=cmap)
        
        # Add grid lines
        self.ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        
        # Set axis labels and title
        self.ax.set_title('Multi-Agent Package Delivery Simulation')
        self.ax.set_xlabel('Column')
        self.ax.set_ylabel('Row')
        
        # Create text elements for time and reward
        self.time_text = self.ax.text(0.02, 0.95, 'Time Step: 0', 
                                     transform=self.ax.transAxes, 
                                     fontsize=12, fontweight='bold', 
                                     bbox=dict(facecolor='white', alpha=0.7))
        self.reward_text = self.ax.text(0.02, 0.90, 'Total Reward: 0.00', 
                                       transform=self.ax.transAxes,
                                       fontsize=12, fontweight='bold',
                                       bbox=dict(facecolor='white', alpha=0.7))
        
        # Add a legend to explain map symbols
        self._create_legend()

    def setup_controls(self):
        """Add animation control buttons and sliders"""
        # Make room for controls by adjusting the main plot
        self.fig.subplots_adjust(bottom=0.15)
        
        # Create axes for the buttons and slider
        pause_ax = self.fig.add_axes([0.45, 0.05, 0.1, 0.04])
        speed_ax = self.fig.add_axes([0.2, 0.05, 0.2, 0.03])
        
        # Create pause/play button
        self.pause_button = Button(pause_ax, 'Pause')
        self.pause_button.on_clicked(self.toggle_pause)
        
        # Create speed slider (0.1 = fast, 2.0 = slow)
        self.speed_slider = Slider(speed_ax, 'Speed', 0.1, 2.0, 
                                  valinit=self.animation_speed, valstep=0.1)
        self.speed_slider.on_changed(self.update_speed)
        
        # Set up keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def toggle_pause(self, event):
        """Toggle the animation between running and paused states"""
        self.animation_running = not self.animation_running
        self.pause_button.label.set_text('Play' if not self.animation_running else 'Pause')
        self.fig.canvas.draw_idle()
        
    def update_speed(self, val):
        """Update the animation speed based on slider"""
        self.animation_speed = val
        
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == ' ':  # Spacebar
            self.toggle_pause(event)
        elif event.key == 'right':  # Right arrow - advance one frame when paused
            if not self.animation_running:
                self.step_animation()
                
    def step_animation(self):
        """Advance the animation by one frame when paused"""
        # Can be implemented if you want step-by-step animation while paused
        pass

    def _create_legend(self):
        """Create a legend explaining map symbols"""
        from matplotlib.patches import Patch, Circle
        from matplotlib.lines import Line2D
        
        # Make room for the legend by adjusting the figure layout
        self.fig.subplots_adjust(right=0.8)
        
        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='Wall'),
            Patch(facecolor='white', edgecolor='gray', label='Free Space'),
            Circle((0, 0), radius=1, facecolor=self.colors['robot'], edgecolor='black', label='Robot'),
            Patch(facecolor=self.colors['package_waiting'], edgecolor='black', alpha=0.5, label='Waiting Package'),
            Patch(facecolor=self.colors['package_transit'], edgecolor='black', alpha=0.5, label='Package in Transit'),
            Patch(facecolor=self.colors['package_delivered'], edgecolor='black', alpha=0.5, label='Delivered Package'),
            Patch(facecolor=self.colors['target'], edgecolor='black', alpha=0.5, label='Target Location'),
            Line2D([0], [0], marker='$R0$', color='w', markerfacecolor='black', markersize=15, label='Robot ID'),
            Line2D([0], [0], marker='$P1$', color='w', markerfacecolor='black', markersize=15, label='Package ID'),
            Line2D([0], [0], marker='$T1$', color='w', markerfacecolor='black', markersize=15, label='Target ID')
        ]
        
        # Place the legend outside the axes, to the right
        self.ax.legend(handles=legend_elements, 
                      loc='center left', 
                      bbox_to_anchor=(1.02, 0.5), 
                      fontsize=10,
                      title="Map Legend",
                      frameon=True)
        
    def update_visualization(self, state, reward=0, actions=None):
        """Update the visualization with the current state"""
        self.state_history.append(state.copy())  # Store state for replay
        self.reward_history.append(reward)
        # Store actions along with the state for replay
        if actions:
            if not hasattr(self, 'action_history'):
                self.action_history = []
            self.action_history.append(actions)
        self.update_display(state, reward, actions)
        plt.pause(0.1)  # Short pause to update display
        
    def update_display(self, state, reward, actions=None):
        """Update the display with the current state and planned actions"""
        # Clear previous markers
        for marker in self.robot_markers:
            marker.remove()
        self.robot_markers = []
        
        # Update time and reward text
        self.time_text.set_text(f'Time Step: {state["time_step"]}')
        
        # Use just the current reward value, not the sum
        self.reward_text.set_text(f'Total Reward: {reward}')
        
        # Add centered large timestep indicator that fades
        # timestep_big = self.ax.text(0.5, 0.5, f'{state["time_step"]}', 
        #                           transform=self.ax.transAxes,
        #                           fontsize=50, alpha=0.3, color='blue',
        #                           ha='center', va='center')
        # self.robot_markers.append(timestep_big)
        
        # Update robot positions
        for i, robot in enumerate(state['robots']):
            row, col, carrying = robot[0]-1, robot[1]-1, robot[2]
            
            # Draw the robot circle
            robot_circle = Circle((col, row), 0.3, color=self.colors['robot'], 
                                  alpha=0.7, zorder=3)
            self.ax.add_patch(robot_circle)
            self.robot_markers.append(robot_circle)
            
            # Add robot ID text
            robot_text = self.ax.text(col, row, f'R{i}', 
                                     ha='center', va='center', color='white',
                                     fontsize=8, zorder=4)
            self.robot_markers.append(robot_text)
            
            # Add movement direction indicator if actions are provided
            if actions and i < len(actions):
                move_direction = actions[i][0]  # First element is the movement direction
                
                # Define arrow endpoints based on direction
                dx, dy = 0, 0
                if move_direction == 'U':
                    dx, dy = 0, -0.5
                elif move_direction == 'D':
                    dx, dy = 0, 0.5
                elif move_direction == 'L':
                    dx, dy = -0.5, 0
                elif move_direction == 'R':
                    dx, dy = 0.5, 0
                    
                # Only draw arrow if robot is actually moving (not staying still)
                if move_direction != 'S':
                    arrow = self.ax.arrow(col, row, dx, dy, head_width=0.2, 
                                         head_length=0.2, fc='yellow', ec='black', 
                                         zorder=5, alpha=0.8)
                    self.robot_markers.append(arrow)
            
            # Add carried package indicator if carrying
            if carrying > 0:
                carried_text = self.ax.text(col, row-0.4, f'P{carrying}', 
                                           ha='center', va='center', color='black',
                                           fontsize=7, zorder=4)
                self.robot_markers.append(carried_text)
        
        # Process current packages
        self.process_packages(state)
        
        # Refresh the canvas
        self.fig.canvas.draw()
        
    def process_packages(self, state):
        """Process package visualization with respect to time"""
        # Get current time step
        current_time = state['time_step']
        
        # Get packages from the environment's internal state
        env_packages = self.env.packages
        
        # First, hide any package markers that shouldn't be visible
        # (those with start time > current_time and not yet in transit/delivered)
        for pkg_id in list(self.package_markers.keys()):
            pkg = next((p for p in env_packages if p.package_id == pkg_id), None)
            if pkg and pkg.start_time > current_time and pkg.status == 'None':
                # Remove the visual elements if they exist
                marker = self.package_markers[pkg_id]
                marker['pickup'].remove()
                marker['target'].remove()
                marker['pickup_text'].remove()
                marker['target_text'].remove()
                del self.package_markers[pkg_id]
        
        # Process only packages that should be visible now
        for pkg in env_packages:
            pkg_id = pkg.package_id
            
            # Skip packages that haven't appeared yet
            if pkg.start_time > current_time and pkg.status == 'None':
                continue
                
            start_row, start_col = pkg.start[0], pkg.start[1]
            target_row, target_col = pkg.target[0], pkg.target[1]
            status = pkg.status  # 'None', 'waiting', 'in_transit', or 'delivered'
            
            # Create marker if it doesn't exist
            if pkg_id not in self.package_markers:
                # Pickup location marker
                pickup_marker = Rectangle((start_col-0.4, start_row-0.4), 0.8, 0.8,
                                        color=self.colors['package_waiting'], alpha=0.5, zorder=2)
                self.ax.add_patch(pickup_marker)
                
                # Target location marker
                target_marker = Rectangle((target_col-0.4, target_row-0.4), 0.8, 0.8, 
                                         color=self.colors['target'], alpha=0.3, zorder=1)
                self.ax.add_patch(target_marker)
                
                # Package ID at pickup
                pickup_text = self.ax.text(start_col, start_row, f'P{pkg_id}',
                                          ha='center', va='center', color='black',
                                          fontsize=7, zorder=2)
                
                # Package ID at target
                target_text = self.ax.text(target_col, target_row, f'T{pkg_id}',
                                          ha='center', va='center', color='black',
                                          fontsize=7, zorder=1)
                
                self.package_markers[pkg_id] = {
                    'pickup': pickup_marker,
                    'target': target_marker,
                    'pickup_text': pickup_text,
                    'target_text': target_text,
                    'status': status
                }
            
            # Update color based on status
            marker = self.package_markers[pkg_id]
            current_status = marker['status']
            
            if status != current_status:
                marker['status'] = status
                if status == 'in_transit':
                    marker['pickup'].set_color(self.colors['package_transit'])
                elif status == 'delivered':
                    marker['target'].set_color(self.colors['package_delivered'])
                elif status == 'waiting':
                    marker['pickup'].set_color(self.colors['package_waiting'])
    
    def run_animation(self, env, agents, steps=100):
        """Run the animation by simulating the environment"""
        state = env.reset()
        agents.init_agents(state)
        
        # Clear any existing visualizations
        self.package_markers = {}
        self.robot_markers = []
        
        done = False
        t = 0
        final_reward = 0
        
        while not done and t < steps:
            # Check if animation is paused
            while not self.animation_running:
                plt.pause(0.1)
                if not plt.fignum_exists(self.fig.number):
                    return {}
            
            # Get planned actions for current state
            actions = agents.get_actions(state)
            
            # Display current state with planned actions
            self.update_visualization(state, round(env.total_reward, 2), actions)
            
            # Pause to let user see planned moves
            plt.pause(self.animation_speed)
            
            # Execute step in environment
            next_state, reward, done, infos = env.step(actions)
            
            # Update state for next iteration
            state = next_state
            t += 1
            
            # Store final reward when done
            if done:
                final_reward = infos.get('total_reward', env.total_reward)
                # Update visualization one last time with the final reward
                self.update_visualization(state, round(final_reward, 2), actions)
            
            # Check if window was closed
            if not plt.fignum_exists(self.fig.number):
                return {}
                
        return infos
    
    def save_animation(self, filename='delivery_simulation2.gif', fps=5):
        """Save the animation as a video file"""
        def animate(i):
            if i < len(self.state_history):
                self.update_display(self.state_history[i], 
                                   self.reward_history[i] if i < len(self.reward_history) else 0)
            return self.robot_markers + list(self.package_markers.values())
        
        ani = animation.FuncAnimation(self.fig, animate, frames=len(self.state_history),
                                      interval=1000/fps, blit=False)
        ani.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved to {filename}")

# Helper function to run the visualization
def visualize_delivery(map_file='map5.txt', num_agents=5, n_packages=10, max_steps=100, seed=2025):
    from env import Environment
    from agentversion2 import AgentsVersion2 as Agents
    
    env = Environment(map_file=map_file, max_time_steps=max_steps,
                      n_robots=num_agents, n_packages=n_packages,
                      seed=seed)
    
    agents = Agents()
    visualizer = DeliveryVisualizer(env)
    
    infos = visualizer.run_animation(env, agents, steps=max_steps)
    
    print("Simulation complete!")
    print(f"Total reward: {infos.get('total_reward', 'N/A')}")
    print(f"Total time steps: {infos.get('total_time_steps', 'N/A')}")
    
    # Save animation to file
    visualizer.save_animation()
    
    # Keep the plot window open
    plt.show()

# Run the visualization if script is executed directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning for Delivery")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for reproducibility")
    parser.add_argument("--max_time_steps", type=int, default=100, help="Maximum time steps for the environment")
    parser.add_argument("--map", type=str, default="map5.txt", help="Map name")

    args = parser.parse_args()
    np.random.seed(args.seed)
    visualize_delivery(
        map_file=args.map,
        num_agents=args.num_agents,
        n_packages=args.n_packages,
        max_steps=args.max_steps,
        seed=args.seed
    )