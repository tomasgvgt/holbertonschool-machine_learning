#/usr/bin/env pythont3
import requests
"""
By using the Swapi API,
create a method that returns the list of ships that can hold
a given number of passengers
"""

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
            try:
                ship_passengers = int(ship['passengers'])
            except:
                pass
            if ship_passengers >= passengerCount:
                ships.append(ship['name'])
        starships = requests.get(starships['next']).json()
    return ships
