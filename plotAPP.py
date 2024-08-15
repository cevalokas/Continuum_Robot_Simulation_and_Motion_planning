import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from Kinematics.curve_V2 import curve
from reinforcement_learning.RL_model import RL_model


class PlotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Continuum Motion Simulation")
        self.curve2_endpoints = []
        self.create_widgets()


    def create_widgets(self):
        self.create_control_frame()
        self.create_entries()
        self.create_plot_canvas()


    def create_control_frame(self):
        self.control_frame = ttk.Frame(self.master)
        self.control_frame.pack(side=tk.RIGHT)
        
        # Buttons for updating plot and clearing data
        ttk.Button(self.control_frame, text="Update", command=self.update_plot).grid(row=10, columnspan=8, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Clear Endpoint motion trajectory", command=self.Clear_Endpoint_traj).grid(row=11, columnspan=8, padx=5, pady=5)
        ttk.Button(self.control_frame, text="RL_model",  command=lambda: self.create_RL()).grid(row=12, columnspan=8, padx=5, pady=5)
    

    def create_RL(self):
        RL_instance = RL_model(self)
        RL_instance.run()


    def create_entries(self):
        posture_labels = ["Phi", "Theta"]
        self.posture_entries = {}
        for i in range(3):  # Three curves
            for j, label in enumerate(posture_labels):
                ttk.Label(self.control_frame, text=f"{label}{i} (float):").grid(row=j, column=i*2, padx=5, pady=5)
                entry = ttk.Entry(self.control_frame)
                entry.grid(row=j, column=i*2+1, padx=5, pady=5)
                entry.insert(0, "0")
                self.posture_entries[f"{label.lower()}{i}"] = entry

        ttk.Label(self.control_frame, text="Length (float):").grid(row=3, column=0, padx=5, pady=5)
        self.length_entry = ttk.Entry(self.control_frame)
        self.length_entry.grid(row=3, column=1, padx=5, pady=5)
        self.length_entry.insert(0, "1")

        self.target_var = tk.BooleanVar()
        ttk.Checkbutton(self.control_frame, text="Add Target", variable=self.target_var).grid(row=5, column=1)
        target_labels = ["X", "Y", "Z", "Radius"]
        self.target_entries = {}
        for i, label in enumerate(target_labels):
            ttk.Label(self.control_frame, text=f"{label} (float):").grid(row=i+6, column=0)
            entry = ttk.Entry(self.control_frame)
            entry.grid(row=i+6, column=1, padx=5)
            entry.insert(0, "1.5" if label != "Radius" else "40")
            self.target_entries[label.lower()] = entry


    def create_plot_canvas(self):
        self.fig = Figure(figsize=(12, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.update_curve()


    def update_curve(self):
        self.ax.clear()
        self.set_ax_properties()
        self.plot_curves()
        self.ax.legend()


    def set_ax_properties(self):
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.grid(True)


    def plot_curves(self):
        length = float(self.length_entry.get())
        t = np.linspace(0, length, 100)
        end_point = [0, 0, 0]
        matrix = np.eye(3)
        colors = ['blue', 'red', 'green']
        labels = ['curve_0', 'curve_1', 'curve_2']

        s_phi = 0
        for i in range(3):
            phi = float(self.posture_entries[f"phi{i}"].get())
            theta = float(self.posture_entries[f"theta{i}"].get())
            curve_data, end_point, matrix, s_phi = curve(t, end_point, matrix, phi, theta, s_phi)
            self.ax.plot(*curve_data, label=labels[i])
            self.ax.scatter(*curve_data[:, -1], color=colors[i], s=20)
            if i == 2:
                self.curve2_endpoints.append(curve_data[:, -1])

        for ep in self.curve2_endpoints:
            self.ax.scatter(*ep, color='green', s=20)
        
        if self.target_var.get():
            x = float(self.target_entries['x'].get())
            y = float(self.target_entries['y'].get())
            z = float(self.target_entries['z'].get())
            radius = float(self.target_entries['radius'].get())

            # Draw the target point
            self.ax.scatter(x, y, z, color='yellow', s=(radius*10)) 


    def update_plot(self):
        self.update_curve()
        self.canvas.draw()


    def Clear_Endpoint_traj(self):
        self.curve2_endpoints = []
        self.update_plot()




if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()