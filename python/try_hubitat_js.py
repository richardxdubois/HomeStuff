from pyhubitat import MakerAPI
from functools import partial
from bokeh.layouts import row, layout, column
from bokeh.models.widgets import TextInput, Dropdown, Slider, Button, Div
from bokeh.plotting import figure, output_file, show, save, curdoc
from bokeh.models.callbacks import CustomJS
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


qjs1 = """
    var h1 = hub + "devices" + "/" + device
    var h2 = "?access_token=" + token
    var h3 = h1 + h2
    //console.log(h3)
    var qhttp = new XMLHttpRequest();
    qhttp.open("GET",h3)
    qhttp.overrideMimeType('json');
    qhttp.onreadystatechange = function () {
      if (qhttp.readyState === qhttp.DONE && qhttp.status === 200) {
       //console.log(qhttp.response);
       var r = JSON.parse(qhttp.response)
       var dev = r.name
       //console.log(dev, r.name, r)
       var offset = 0
        if (dev.indexOf("Switch") != -1) {
            offset = 4
            }
        if (dev.indexOf("Outlet") != -1) {
            offset = 0
            }
        if (dev.indexOf("Room") != -1) {
            offset = 2
            }

       var switch_value = r.attributes[offset].currentValue
       if (switch_value == 'on') {
         button.button_type = "danger"
         }
       else {
         button.button_type = "success"
        }
       //console.log(cb_obj.origin.properties.button_type.spec)
           var xhttp = new XMLHttpRequest();
     var set_state = "on"
    // for some reason the button_type doesn't seem to refresh, so assume it will change state
     if (button.button_type == "danger") {
        set_state = "off"
    }
    var command = h1 + '/' + set_state + h2
    //console.log(command)
    xhttp.open("GET", command, true)
    xhttp.send()

      }
    }
    qhttp.send()
"""


update_button_js = []
for i, b in enumerate(buttons):
    #    b.on_click(partial(update_button_js, i))
    #js3 = qjs1 + qjs2 + js1
    update_button_js.append(CustomJS(args=dict(button=buttons[i], device=b_map[i], hub=access_url,
                                               token=access_token), code=qjs1))
    b.js_on_click(update_button_js[i])

refresh_button = Button(label="Refresh", button_type="warning")


def refresh_cb():
    for id in b_map:
        who = b_map[id]
        d_status = ph.device_status(who)
        switch_val = d_status["switch"]["currentValue"]
        if switch_val == "on":
            buttons[id].button_type = "success"
        else:
            buttons[id].button_type = "danger"


js_refresh = """
    var h1 = hub + "/devices/"
    var h2 = "?access_token=" + token
    const rhttp = []
    for (let i in b_map) {
      var id = b_map[i]
      var h3 = h1 + id + h2
      //console.log(h3)
      rhttp.push(new XMLHttpRequest());
      rhttp[i].open("GET",h3)
      rhttp[i].overrideMimeType('json');
      rhttp[i].onreadystatechange = function () {
        if (rhttp[i].readyState === rhttp[i].DONE && rhttp[i].status === 200) {
         //console.log(rhttp[i].response);
         var r = JSON.parse(rhttp[i].response)
         //console.log(r)
         var dev = r.name
         var offset = 0
         if (dev.indexOf("Switch") != -1) {
            offset = 4
            }
         if (dev.indexOf("Outlet") != -1) {
            offset = 0
            }
        if (dev.indexOf("Room") != -1) {
            offset = 2
            }
        var switch_value = r.attributes[offset].currentValue
        //console.log(dev, switch_value)
         if (switch_value == 'on') {
           buttons[i].button_type = "success"
          }
         else {
           buttons[i].button_type = "danger"
          }
        }
      }
      var now = new Date()
      div.text = "Refreshed at: " + now
      rhttp[i].send()
    }
"""

del_div = Div(text="Run on: " + datetime.datetime.now().strftime("%Y-%m-%d") + " Evening on at " +
                   evening_on.strftime("%H:%M:%S"))

refresh_cb_js = CustomJS(args=(dict(b_map=b_map, buttons=buttons, div=del_div, hub=access_url,
                        token=access_token)), code=js_refresh)

refresh_button.js_on_click(refresh_cb_js)

m = layout(del_div, column(buttons), refresh_button)

output_file("try_hubitat_js.html")
save(m, title="Hubitat JS Dashboard")
