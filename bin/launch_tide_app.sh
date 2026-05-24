#!/bin/bash
# Launch the Bokeh tide app
export PATH="/Users/richarddubois/opt/miniconda3/envs/python311/bin:$PATH"
#cd /Users/richarddubois/Code/Home/HomeStuff/python  # Adjust to your path
#bokeh serve tides_app.py &

osascript -e 'tell application "Terminal" to do script "cd /Users/richarddubois/Code/Home/HomeStuff/python && /Users/richarddubois/opt/miniconda3/envs/python311/bin/bokeh serve tides_app.py"'
# Save the PID
BOKEH_PID=$!

# Wait a moment for server to start
sleep 2

# Open in new Chrome window
open -na "Google Chrome" --args --new-window "http://localhost:5006/tides_app"

# Wait for bokeh process to finish
wait $BOKEH_PID

# Exit cleanly
exit 0
