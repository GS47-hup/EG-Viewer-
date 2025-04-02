import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import argparse
from PyQt5 import QtWidgets, QtCore
import ecg_viewer
from real_ecg_classifier import RealEcgClassifier

class ECGSimulator:
    """
    Class for simulating real ECG data for the ECG Viewer application
    """
    def __init__(self, ecg_file='Real ECG.csv', sample_index=None, is_normal=None):
        self.ecg_file = ecg_file
        self.sample_index = sample_index
        self.is_normal = is_normal
        self.data = None
        self.sample = None
        self.label = None
        self.time_values = None
        self.ecg_values = None
        self.simulator_active = False
        self.sample_rate = 250  # Hz
        
    def load_data(self):
        """Load the ECG data from the file"""
        print(f"Loading ECG data from {self.ecg_file}...")
        
        try:
            # Load the data
            self.data = pd.read_csv(self.ecg_file, header=None)
            
            # If specific sample_index is provided, use it
            if self.sample_index is not None:
                if self.sample_index >= len(self.data):
                    print(f"Error: Sample index {self.sample_index} is out of range. File has {len(self.data)} samples.")
                    return False
                
                # Extract the sample and label
                self.sample = self.data.iloc[self.sample_index, :-1].values
                self.label = self.data.iloc[self.sample_index, -1]
                print(f"Using sample {self.sample_index} with label {'Normal' if self.label == 0 else 'Abnormal'}")
                
            # If specific type (normal/abnormal) is requested, filter and select randomly
            elif self.is_normal is not None:
                target_label = 0 if self.is_normal else 1
                filtered_data = self.data[self.data.iloc[:, -1] == target_label]
                
                if len(filtered_data) == 0:
                    print(f"Error: No {'normal' if self.is_normal else 'abnormal'} samples found in the file.")
                    return False
                
                # Select a random sample from the filtered data
                random_index = np.random.randint(0, len(filtered_data))
                self.sample_index = filtered_data.index[random_index]
                self.sample = filtered_data.iloc[random_index, :-1].values
                self.label = target_label
                print(f"Randomly selected {'normal' if self.is_normal else 'abnormal'} sample (index {self.sample_index})")
                
            else:
                # Select a random sample from the entire dataset
                self.sample_index = np.random.randint(0, len(self.data))
                self.sample = self.data.iloc[self.sample_index, :-1].values
                self.label = self.data.iloc[self.sample_index, -1]
                print(f"Randomly selected sample {self.sample_index} with label {'Normal' if self.label == 0 else 'Abnormal'}")
            
            # Create time values (assuming 250 Hz sampling rate)
            length = len(self.sample)
            self.time_values = np.arange(length) * (1000 / self.sample_rate)  # in ms
            self.ecg_values = self.sample
            
            print(f"Successfully loaded ECG sample with {length} data points")
            return True
            
        except Exception as e:
            print(f"Error loading ECG data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def plot_sample(self, save_path=None):
        """Plot the selected ECG sample"""
        if self.sample is None:
            print("No ECG sample loaded. Call load_data() first.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_values, self.ecg_values)
        plt.title(f"ECG Sample {self.sample_index} - {'Normal' if self.label == 0 else 'Abnormal'}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Sample plot saved to {save_path}")
        else:
            plt.show()

    def simulate_to_ecg_viewer(self, viewer, speed_factor=1.0):
        """
        Simulate the ECG data to the ECG Viewer application
        
        Args:
            viewer: ECG Viewer instance
            speed_factor: Speed factor (1.0 = real-time, 2.0 = twice as fast)
        """
        if self.sample is None:
            print("No ECG sample loaded. Call load_data() first.")
            return
        
        try:
            # Prepare for simulation
            self.simulator_active = True
            
            # Calculate delay between points based on speed factor
            delay = (1.0 / self.sample_rate) / speed_factor  # in seconds
            
            # Inform user
            print(f"Starting ECG simulation with speed factor {speed_factor}x (delay={delay*1000:.2f}ms)")
            print(f"ECG type: {'Normal' if self.label == 0 else 'Abnormal'}")
            print("Press Ctrl+C to stop the simulation")
            
            # Simulate the ECG data
            index = 0
            while self.simulator_active and index < len(self.ecg_values):
                value = self.ecg_values[index]
                time_ms = self.time_values[index]
                
                # Add the value to the ECG viewer
                viewer.add_data_point(time_ms, value)
                
                # Update the viewer
                QtWidgets.QApplication.processEvents()
                
                # Sleep to simulate real-time data
                time.sleep(delay)
                
                # Increment index
                index += 1
                
                # Loop back to start when reaching the end
                if index >= len(self.ecg_values):
                    index = 0
                    print("Restarting ECG simulation from beginning")
            
            print("ECG simulation finished")
            
        except KeyboardInterrupt:
            print("ECG simulation stopped by user")
            self.simulator_active = False
        except Exception as e:
            print(f"Error during ECG simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            self.simulator_active = False

    def stop_simulation(self):
        """Stop the ECG simulation"""
        self.simulator_active = False
        print("Stopping ECG simulation")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simulate real ECG data for the ECG Viewer')
    parser.add_argument('--file', type=str, default='Real ECG.csv',
                        help='Path to the ECG data file')
    parser.add_argument('--index', type=int, default=None,
                        help='Specific sample index to use')
    parser.add_argument('--normal', action='store_true',
                        help='Use a random normal ECG sample')
    parser.add_argument('--abnormal', action='store_true',
                        help='Use a random abnormal ECG sample')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speed factor (1.0 = real-time, 2.0 = twice as fast)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot the sample, do not simulate')
    args = parser.parse_args()
    
    # Handle contradicting options
    if args.normal and args.abnormal:
        print("Error: Cannot specify both --normal and --abnormal")
        return 1
    
    # Initialize ECG simulator
    simulator = ECGSimulator(
        ecg_file=args.file,
        sample_index=args.index,
        is_normal=args.normal if args.normal else (False if args.abnormal else None)
    )
    
    # Load the data
    if not simulator.load_data():
        return 1
    
    # If plot-only is specified, just plot the sample and exit
    if args.plot_only:
        simulator.plot_sample()
        return 0
    
    # Initialize ECG viewer
    app = QtWidgets.QApplication(sys.argv)
    viewer = ecg_viewer.ECGViewer()
    viewer.setWindowTitle("ECG Viewer - Simulating Real ECG")
    viewer.show()
    
    # Start simulation in a separate thread to keep UI responsive
    from threading import Thread
    simulation_thread = Thread(
        target=simulator.simulate_to_ecg_viewer,
        args=(viewer, args.speed),
        daemon=True
    )
    simulation_thread.start()
    
    # When the viewer is closed, stop the simulation
    app.aboutToQuit.connect(simulator.stop_simulation)
    
    # Run the application
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main()) 