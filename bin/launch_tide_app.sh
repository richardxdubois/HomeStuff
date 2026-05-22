#!/bin/bash
# Launch the Bokeh tide app
export PATH="/Users/richarddubois/opt/miniconda3/envs/python311/bin:$PATH"
cd /Users/richarddubois/Code/Home/HomeStuff/python  # Adjust to your path
bokeh serve tides_app.py --show
