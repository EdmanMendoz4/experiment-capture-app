from pynput import keyboard
import os
import re

class KeyboardHandler:
    """Handles keyboard input with state management."""
    
    def __init__(self):
        self.stop_pressed = False
    
    def on_press_start(self, key):
        try:
            if key == keyboard.Key.enter:
                return False  # Stop listener
        except AttributeError:
            pass
    
    def on_press_stop(self, key):
        try:
            if key == keyboard.Key.esc:
                self.stop_pressed = True
                return False  # Stop listener
        except AttributeError:
            pass


# Utility function to get the next available cell ID based on existing folders 
        
def get_next_cell_id(project_path: str) -> int:
    """Scans project_path for 'Cell_X' folders and returns the next ID."""
    try:
        if not os.path.exists(project_path):
            return 1
        
        cell_folders = []
        for item in os.listdir(project_path):
            match = re.match(r'Cell_(\d+)', item)
            if match:
                cell_folders.append(int(match.group(1)))
        
        return max(cell_folders) + 1 if cell_folders else 1
    except Exception as e:
        print(f"Warning: Could not scan project path ({e}). Starting at Cell_1.")
        return 1