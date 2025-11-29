import os, sys

if __name__ == '__main__':
  """
  Used to quickly rename files within a given directory. The name of the folder is used as each image name accompanied by a number. For example: alfalfa0.jpg.

  Command example:
    python img_rename.py 'alfalfa'
  """
  # Set directory and path variables
  DIR_NAME = sys.argv[1]
  path = os.getcwd() + "\\dataset\\" + DIR_NAME + "\\"
  
  # Go through each file
  for count, filename in enumerate(os.listdir(path)):
    new_name = DIR_NAME.lower() + str(count) + ".jpg"
    src = path + filename # Old name
    dst = path + new_name # New name

    # Rename files
    os.rename(src, dst)