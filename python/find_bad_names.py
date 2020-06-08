import os

for root, directories, filenames in os.walk('./'):
     for filename in filenames:
        if '#' in filename:
            print os.path.join(root,filename)