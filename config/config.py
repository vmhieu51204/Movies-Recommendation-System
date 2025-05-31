import os
import argparse

config_parameters = {
    "path": None,
}
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.txt')

def write_config(file_path=CONFIG_FILE_PATH):
    """
    Writes the current configuration parameters to a text file.
    Each parameter is written on a new line in KEY=VALUE format.
    """
    with open(file_path, 'w') as f:
        f.write("# Project Configuration\n")
        f.write("# Generated automatically by config.py\n\n")
        for key, value in config_parameters.items():
            f.write(f"{key}={value}\n")
    print(f"Configuration saved to: {file_path}")


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="enter the path to the folder with the dataset csv files",
                        type=str)
    args = parser.parse_args()
    path = args.path

    global config_parameters
    config_parameters['path'] = path
    write_config()


def read_config(file_path=CONFIG_FILE_PATH):
    """
    Reads configuration parameters from a text file and returns them as a dictionary.
    Assumes each parameter is on a new line in KEY=VALUE format.
    Lines starting with '#' are treated as comments and ignored.
    """
    read_parameters = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        read_parameters[key.strip()] = value.strip()
                    except ValueError:
                        print(f"Warning: Skipping malformed line in config file: {line}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {file_path}. Returning empty configuration.")
    return read_parameters

if __name__ == '__main__':

    set_config()


