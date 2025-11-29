import sys, os
import pandas as pd

def get_filenames(DIRECTORY):
  """
  Returns a list of filenames within a given directory.

  Command example:
    python get_filenames.py 'alfalfa'
  """
  # Get filenames
  path = os.getcwd() + "\\dataset\\" + DIRECTORY + "\\"
  files = pd.DataFrame([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))], columns=['name'])

  # Add filenames to CSV file
  files.to_csv(DIRECTORY + '_filenames.csv', index=False)

# Run main function
if __name__ == "__main__":
  DIR_NAME = sys.argv[1]
  get_filenames(DIR_NAME)