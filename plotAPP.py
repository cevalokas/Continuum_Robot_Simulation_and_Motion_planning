import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from Kinematics.curve_V2 import curve
from reinforcement_learning.RL_model import RL_model


class PlotApp:
    def __init__(self, master):
        # Initialize the main application window
        self.master = master
        self.master.title("Continuum Motion Simulation")
        self.curve2_endpoints = []
        self.create_widgets()  # Call the function to create GUI elements


    def create_widgets(self):
        # Create and organize the layout of the control panel and plotting area
        self.create_control_frame()
        self.create_entries()
        self.create_plot_canvas()


    def create_control_frame(self):
        # Create a frame on the right side to hold control buttons and inputs
        self.control_frame = ttk.Frame(self.master)
        self.control_frame.pack(side=tk.RIGHT)
        
        # Buttons for updating plot and clearing data
        ttk.Button(self.control_frame, text="Update", command=self.update_plot).grid(row=10, columnspan=8, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Clear Endpoint motion trajectory", command=self.Clear_Endpoint_traj).grid(row=11, columnspan=8, padx=5, pady=5)
        ttk.Button(self.control_frame, text="RL_model",  command=lambda: self.create_RL()).grid(row=12, columnspan=8, padx=5, pady=5)
    

    def create_RL(self):
        # Create an instance of the reinforcement learning model and run it
        RL_instance = RL_model(self)
        RL_instance.run()


    def create_entries(self):
        # Create input fields for entering posture angles and other parameters
        posture_labels = ["Phi", "Theta"]  # Labels for posture angles
        self.posture_entries = {}
        for i in range(3):  # Create input fields for three curves
            for j, label in enumerate(posture_labels):
                ttk.Label(self.control_frame, text=f"{label}{i} (float):").grid(row=j, column=i*2, padx=5, pady=5)
                entry = ttk.Entry(self.control_frame)
                entry.grid(row=j, column=i*2+1, padx=5, pady=5)
                entry.insert(0, "0")  # Default value is 0
                self.posture_entries[f"{label.lower()}{i}"] = entry

        # Create input field for length
        ttk.Label(self.control_frame, text="Length (float):").grid(row=3, column=0, padx=5, pady=5)
        self.length_entry = ttk.Entry(self.control_frame)
        self.length_entry.grid(row=3, column=1, padx=5, pady=5)
        self.length_entry.insert(0, "1")  # Default length is 1

        # Checkbox and input fields for target point coordinates
        self.target_var = tk.BooleanVar()
        ttk.Checkbutton(self.control_frame, text="Add Target", variable=self.target_var).grid(row=5, column=1)
        target_labels = ["X", "Y", "Z", "Radius"]
        self.target_entries = {}
        for i, label in enumerate(target_labels):
            ttk.Label(self.control_frame, text=f"{label} (float):").grid(row=i+6, column=0)
            entry = ttk.Entry(self.control_frame)
            entry.grid(row=i+6, column=1, padx=5)
            entry.insert(0, "1.5" if label != "Radius" else "40")  # Default values for target coordinates
            self.target_entries[label.lower()] = entry


    def create_plot_canvas(self):
        # Create the matplotlib figure and canvas for plotting curves
        self.fig = Figure(figsize=(12, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')  # 3D plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.update_curve()  # Initial curve plot


    def update_curve(self):
        # Update the curve plot with the current parameter values
        self.ax.clear()  # Clear the previous plot
        self.set_ax_properties()
        self.plot_curves()
        self.ax.legend()  # Add legend to the plot


    def set_ax_properties(self):
        # Set the properties of the 3D axis
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.grid(True)  # Enable grid


    def plot_curves(self):
        # Calculate and plot the curves based on the input parameters
        length = float(self.length_entry.get())
        t = np.linspace(0, length, 100)  # Generate 100 points along the length
        end_point = [0, 0, 0]  # Initial endpoint
        matrix = np.eye(3)  # Identity matrix for initial rotation
        colors = ['blue', 'red', 'green']  # Colors for the curves
        labels = ['curve_0', 'curve_1', 'curve_2']  # Labels for the curves

        s_phi = 0  # Initial s_phi value
        for i in range(3):  # Loop over the three curves
            phi = float(self.posture_entries[f"phi{i}"].get())  # Get phi value
            theta = float(self.posture_entries[f"theta{i}"].get())  # Get theta value
            curve_data, end_point, matrix, s_phi = curve(t, end_point, matrix, phi, theta, s_phi)  # Compute the curve
            self.ax.plot(*curve_data, label=labels[i])  # Plot the curve
            self.ax.scatter(*curve_data[:, -1], color=colors[i], s=20)  # Plot the endpoint of the curve
            if i == 2:
                self.curve2_endpoints.append(curve_data[:, -1])  # Store the endpoint of the third curve

        # Re-plot all stored endpoints of the third curve
        for ep in self.curve2_endpoints:
            self.ax.scatter(*ep, color='green', s=20)
        
        # Plot the target point if the checkbox is selected
        if self.target_var.get():
            x = float(self.target_entries['x'].get())
            y = float(self.target_entries['y'].get())
            z = float(self.target_entries['z'].get())
            radius = float(self.target_entries['radius'].get())

            # Draw the target point
            self.ax.scatter(x, y, z, color='yellow', s=(radius*10)) 


    def update_plot(self):
        # Update the plot by redrawing the curves and target
        self.update_curve()
        self.canvas.draw()


    def Clear_Endpoint_traj(self):
        # Clear the stored endpoints of the third curve and update the plot
        self.curve2_endpoints = []
        self.update_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
