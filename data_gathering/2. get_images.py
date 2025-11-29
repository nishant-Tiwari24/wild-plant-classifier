import csv, requests, os, sys, time

# Create get images function
def get_images(FILENAME):
  """
	Used to download Flickr images from a csv file
  of Flickr image URLs.

	Parameters:
		FILENAME (string) = csv filename

	Command example:
		python get_images.py "alfalfa_flower_urls.csv"
	"""
  urls = []
  success_count = 0

  # Open csv file
  with open(FILENAME, newline="") as csvfile:
    doc = csv.reader(csvfile, delimiter=",")
    # Add links to urls list
    for row in doc:
      if row[1].startswith("https"):
        urls.append(row[1])
  
  # Create a new directory
  path = os.getcwd() + "\dataset\\"
  if not os.path.isdir(os.path.join(path, FILENAME.split("_")[0])):
    os.mkdir(path + FILENAME.split("_")[0])

  start = time.time()
  # Download images from urls list
  for idx, url in enumerate(urls):
    print(f"Downloading {idx+1} of {len(urls)}", end="")
    try:
      resp = requests.get(urls[idx], stream=True)
      path_to_write = os.path.join(path, FILENAME.split("_")[0], urls[idx].split("/")[-1])
      outfile = open(path_to_write, 'wb')
      outfile.write(resp.content)
      outfile.close()
      print(f" -> complete.")
      success_count += 1
    except:
      print(f"-> failed. Url: {url}")
  
  # Calculate time taken
  end = time.time()
  mins, secs = divmod(end - start, 60)

  # Output completed message
  print(f"{success_count} of {len(urls)} images successfully downloaded. Time taken: {int(mins):0>2}:{secs:05.2f} minutes and seconds.")

# Run main
if __name__=='__main__':
  FILE_NAME = sys.argv[1]
  get_images(FILE_NAME)