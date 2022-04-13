#!/usr/bin/env python3
"""Script that displays the upcoming launch"""
import requests
import time


if __name__ == '__main__':
    upcoming_url = 'https://api.spacexdata.com/v4/launches/upcoming'
    launches = requests.get(upcoming_url).json()
    curr_time = time.time()
    soonest_time = launches[0]['date_unix']
    st = 0
    for i in range(len(launches)):
        launch_time = launches[i]['date_unix']
        if launch_time < soonest_time and launch_time > curr_time:
            st = i
            soonest_time = launch_time
    soonest_launch = launches[st]
    launch_name = soonest_launch['name']
    date = soonest_launch['date_local']
    rocket_num = soonest_launch['rocket']
    launchpad_num = soonest_launch['launchpad']
    url_rocket = 'https://api.spacexdata.com/v4/rockets/' + rocket_num
    url_launchpad = 'https://api.spacexdata.com/v4/launchpads/' + launchpad_num
    rocket_name = requests.get(url_rocket).json()['name']
    launchpad_name = requests.get(url_launchpad).json()['name']
    launchpad_locality = requests.get(url_launchpad).json()['locality']
    print('{} ({}) {} - {} ({})'.format(launch_name, date,
                                        rocket_name, launchpad_name,
                                        launchpad_locality))
