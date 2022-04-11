#!/usr/bin/env python3
"""Script to request Github API for users location."""
import requests
import time
from sys import argv


if __name__ == '__main__':
    address = argv[-1]

    params = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(address, params=params)
    if response.status_code == 200:
        print(response.json()['location'])

    elif response.status_code == 403:
        limit = int(response.headers['X-Ratelimit-Reset'])
        now = time.time()
        minutes = int((limit - now) / 60)

        print('Reset in {} min'.format(minutes))

    elif response.status_code == 404:
        print("Not found")
