import os, cv2

# Create resize_imgs function
def resize_imgs():
  """
  Used to quickly resize images to 600 width x auto height. Must be run inside the root folder that contains an existing dataset and resized folder. Additionally, resized folder must contain class subfolders already.

  Command example:
    python resize_images.py
  """
  # Set initial variables
  completed = [
  'alfalfa', 'allium', 'borage', 'burdock', 'calendula', 'cattail', 'chickweed', 'chicory',
  'chive_blossom', 'coltsfoot', 'common_mallow', 'common_milkweed', 'common_vetch', 'common_yarrow',
  'coneflower', 'cow_parsley', 'cowslip', 'crimson_clover', 'crithmum_maritimum', 'daisy',
  'dandelion', 'fennel', 'fireweed', 'gardenia', 'garlic_mustard', 'geranium', 'ground_ivy',
  'harebell', 'henbit', 'knapweed', 'meadowsweet', 'mullein', 'pickerelweed', 'ramsons', 'red_clover'
  ]
  root = os.getcwd() + "\\dataset\\"
  new_location = os.getcwd() + "\\resized\\"

  # Get each file within dataset directory
  for path, _, files in os.walk(root):
    # Loop through each file and get the names
    for name in files:
    	# Get folder name
    	sub = path.split('\\')[-1]
      
    	if sub not in completed:
    		# Set path locations
    		new_loc_path = os.path.join(new_location, sub, name)
    		img_path = os.path.join(path, name)

    		# Open image
    		img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    		print(img_path)

    		# Calculate sizing
    		width = 600
    		height = (float(img.shape[0]))
    		width_percent = (width / float(img.shape[1]))
    		height = int(height * float(width_percent))

    		# Resize and save image
    		img_resize = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    		cv2.imwrite(new_loc_path, img_resize)

# Run main function
if __name__ == "__main__":
	resize_imgs()