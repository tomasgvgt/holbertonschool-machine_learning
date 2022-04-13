#!/usr/bin/env python3
"""Script to search for number of launches for each rocket."""
import requests

if __name__ == '__main__':
    response = requests.get('https://api.spacexdata.com/v4/launches')
    launches = response.json()
    r_launches = {}
    for launch in launches:
        rocket_num = launch['rocket']
        if r_launches.get(rocket_num, 0) == 0:
            r_launches[rocket_num] = 1
        else:
            r_launches[rocket_num] += 1
    launches_list = []
    for rocket_num in r_launches.keys():
        rocket_url = 'https://api.spacexdata.com/v4/rockets/' + rocket_num
        rocket_response = requests.get(rocket_url)
        rocket_name = rocket_response.json()['name']
        num_launches = r_launches[rocket_num]
        launches_list.append((rocket_name, num_launches))
    launches_list.sort(key=lambda tup: tup[0])
    launches_list.sort(key=lambda tup: tup[1], reverse=True)
    for rocket_name, num_launches in launches_list:
        print('{}: {}'.format(rocket_name, num_launches))
