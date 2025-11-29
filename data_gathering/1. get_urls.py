from flickrapi import FlickrAPI
import pandas as pd
import sys

# Set API variables
KEY = '[API KEY]'
SECRET = '[API SECRET]'

# Create URL function
def get_urls(IMAGE_TAG, MAX_COUNT):
	"""
	Used to get a list of image URLs from Flickr based on
	the specified IMAGE_TAG and MAX_COUNT.

	Parameters:
		IMAGE_TAG (string) = type of image url
		MAX_COUNT (int) = url count to get

	Command example:
		python get_urls.py "alfalfa flower" 500
	"""
	# Set variables
	flickr = FlickrAPI(KEY, SECRET)
	photos = flickr.walk(text=IMAGE_TAG, tag_mode='all', extras='url_o', per_page=50, sort='relevance', media='photos')
	count, urls = 0, {}

	print(f"Retrieving {MAX_COUNT} urls for {IMAGE_TAG}...")
	# Loop through images
	for photo in photos:
		# Check count is lower than max
		if count < MAX_COUNT:
			# Try getting URL
			url = photo.get('url_o')
			if url is not None:
				urls[count + 1] = url # Add to url dict
				count += 1 # Increment count
		else:
			break
	
	# Check url count in file
	print(f"Urls retrieved. Obtained {len(urls)} urls out of {MAX_COUNT}.")
	
	# Put links into csv file
	urls = pd.DataFrame(urls.items(), columns=['id', 'url'])
	print("Importing urls into new file...")
	filename = "_".join(IMAGE_TAG.lower().split(" "))
	urls.to_csv(filename + "_urls.csv", index=False)
	print(f"Conversion complete. New file created: {filename}_urls.csv.")

# Run main
if __name__=='__main__':
	# Set variables
	IMAGE_TAG = sys.argv[1]
	MAX_COUNT = int(sys.argv[2])

	# Run function
	get_urls(IMAGE_TAG, MAX_COUNT)