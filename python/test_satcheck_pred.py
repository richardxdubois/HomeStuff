import requests
from datetime import datetime, timezone, date
from astropy.time import Time

#utc_time_str = "2025-11-26 08:05:46"
#utc_time_str = "2025-11-26 07:47:54"
#utc_time_str = "2024-08-21 13:51:54"
#utc_time_str = "2025-03-15 21:03:12"
utc_time_str = "2026-02-20 07:18:00"

utc_Time = Time(utc_time_str)
jd = utc_Time.jd

start_jd = jd
stop_jd = start_jd + 0.0002

# .2 is 4.8 hours
step_jd = 0.0001

observer_latitude =  -30.244714  # Rubin Latitude
observer_longitude =  -70.747658  # Rubin Longitude
observer_elevation = 2622  # Rubin Elevation (meters)

satellite_list = [
    "GLAST"
]

"""
observer_latitude = 32
observer_longitude = -110
observer_elevation = 0

satellite_list = [
    "STARLINK-30109"
    ]
"""

visible_satellites = []
not_visible_satellites = []

for satellite in satellite_list:
    response = requests.get(
        f"https://satchecker.cps.iau.org/ephemeris/name-jdstep/?name={satellite}&elevation={observer_elevation}&latitude={observer_latitude}&longitude={observer_longitude}&startjd={start_jd}&stopjd={stop_jd}&stepjd={step_jd}&min_altitude=-90",
        timeout=10,
    )
    info = response.json()
    print(response.url)
    #print(info)
    fields = info["fields"]
    altitude_index = fields.index("altitude_deg")
    ra_index = fields.index("right_ascension_deg")
    dec_index = fields.index("declination_deg")
    # Iterate through each satellite's data
    for satellite_data in info["data"]:
        # Extract the satellite's name and altitude using the fields' indexes
        satellite_name = satellite_data[fields.index("name")]
        altitude_deg = satellite_data[altitude_index]
        ra_deg = satellite_data[ra_index]
        dec_deg = satellite_data[dec_index]
        print("alt=",altitude_deg, "ra=",ra_deg, "dec=",dec_deg)

        # Check if the satellite is visible
        if altitude_deg > 0:
            visible_satellites.append(satellite_data)
        else:
            not_visible_satellites.append(satellite_data)

fields = info["fields"]
data = info["data"]

print("\n")
print("Visible Satellites: ")
for satellite in visible_satellites:
    satellite_info = dict(zip(fields, satellite, strict=True))

    print(satellite_info.get("name"))
    print("Altitude (degrees): ", satellite_info.get("altitude_deg"))
    print("Azimuth (degrees): ", satellite_info.get("azimuth_deg"))
    print("Time: ", satellite_info.get("julian_date"))
    print("--------------------------------------")

print("\n\n")

print("Not Currently Visible: ")
for satellite in not_visible_satellites:
    satellite_info = dict(zip(fields, satellite, strict=True))
    print(satellite_info["name"])
    print(satellite_info["julian_date"])
    print("-------------------------")
