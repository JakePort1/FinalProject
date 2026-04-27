"""
Use this file to import the data, 

May need to install kagglehub library, via: 
    /Library/Developer/CommandLineTools/usr/bin/python3 -m pip install kagglehub 
in terminal 

It may be smarter to just store the data in the git repository, but it may be too big too? We'll see lol

"""

import kagglehub
path = kagglehub.dataset_download("adityajn105/flickr8k")

print("Path to dataset files:", path)
