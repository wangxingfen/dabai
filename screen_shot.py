import pyautogui
import os

def take_screenshot(output_path=None):
    """
    Take a screenshot and save it to a file.
    
    Args:
        output_path (str, optional): Path where screenshot will be saved. 
                                    If not provided, saves to current directory 
                                    with timestamp filename.
    
    Returns:
        str: Path to the saved screenshot file.
    """
    # Take screenshot using pyautogui
    screenshot = pyautogui.screenshot()
    
    # Generate default path if none provided
    if output_path is None:
        output_path = f"screenshot.png"
    
    # Save screenshot to file
    screenshot.save(output_path)
    
    # Return the path where screenshot was saved
    return os.path.abspath(output_path)