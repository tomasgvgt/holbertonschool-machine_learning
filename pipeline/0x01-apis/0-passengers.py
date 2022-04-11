#!/usr/bin/env python3
"""
By using the Swapi API,
create a method that returns the list of ships that can hold
a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """
    Returns the list of ships that can hold a given number of passengers
    Don’t forget the pagination
    If no ship available, return an empty list.
    """
    starships = requests.get("https://swapi-api.hbtn.io/api/starships").json()
    ships = []
    while starships['next'] is not None:
        for ship in starships['results']:
            ship_passengers = ship.get('passengers', 0).replace(',', '')
            try:
                ship_passengers = int(ship_passengers)
            except ValueError:
                continue
            if ship_passengers >= passengerCount:
                ships.append(ship['name'])
        starships = requests.get(starships['next']).json()
    return ships
