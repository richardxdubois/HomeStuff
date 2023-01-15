from pyhubitat import MakerAPI
from functools import partial
from bokeh.layouts import row, layout, column
from bokeh.models.widgets import TextInput, Dropdown, Slider, Button, Div
from bokeh.plotting import figure, output_file, show, save, curdoc
from astral import LocationInfo
from astral.location import Location
from astral.sun import sun
import datetime

kwds = {}
with open("/Users/richarddubois/Code/Home/hubitat_connect.txt") as f:
    for line in f:
        (key, val) = line.split()
        kwds[key] = val

access_token = kwds["token"]
access_url = kwds["url"]

ph = MakerAPI(access_token, access_url)
devices = ph.list_devices()
buttons = []
b_map = {}

print("completed getting list of devices")

# get sunset (lights set to turn on at sunset - 75 mins)

city = LocationInfo("San Francisco", "Pacific", "US/Pacific", 37.775, -122.42)
sf = Location(city)
s = sun(city.observer, date=datetime.date(2023,1,1), tzinfo=sf.timezone)
sunset = s["sunset"]
evening_on = sunset - datetime.timedelta(minutes=75)

print("Sunset today at", sunset)

incr = 0
for d in devices:
    id = d['id']
    status = ph.device_status(id)
    switch_val = status["switch"]["currentValue"]
    # lazy way to tell that a device is not a dimmer
    try:
        level_val = status["level"]["currentValue"]
    except KeyError:
        level_val = "NA"

    print(d["label"], id, switch_val, level_val)

    button_state = "danger"
    if switch_val == "on":
        button_state = "success"
    buttons.append(Button(label=d["label"], button_type=button_state))
    b_map[incr] = id
    incr += 1


def update_button(who):
    id = b_map[who]
    d_status = ph.device_status(id)
    switch_val = d_status["switch"]["currentValue"]
    if switch_val == "off":
        buttons[who].button_type = "success"
        ph.send_command(id, "on")
    else:
        buttons[who].button_type = "danger"
        ph.send_command(id, "off")


for i, b in enumerate(buttons):
    b.on_click(partial(update_button, i))

refresh_button = Button(label="Refresh", button_type="success")


def refresh_cb():
    for id in b_map:
        who = b_map[id]
        d_status = ph.device_status(who)
        switch_val = d_status["switch"]["currentValue"]
        if switch_val == "on":
            buttons[id].button_type = "success"
        else:
            buttons[id].button_type = "danger"


refresh_button.on_click(refresh_cb)


exit_button = Button(label="Exit", button_type="danger")


def exit_cb():
    exit(0)


exit_button.on_click(exit_cb)

del_div = Div(text="Run on: " + datetime.datetime.now().strftime("%Y-%m-%d") + " Evening on at " +
                   evening_on.strftime("%H:%M:%S"))

m = layout(del_div, column(buttons), refresh_button, exit_button)

curdoc().add_root(m)
curdoc().title = "Hubitat Dashboard"
