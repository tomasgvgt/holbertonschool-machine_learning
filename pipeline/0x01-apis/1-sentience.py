#!/usr/bin/env python3
"""
By using the Swapi API, create a method that returns
the list of names of the home planets of all sentient species.
"""
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets
    of all sentient species.
    """
    species = requests.get("https://swapi-api.hbtn.io/api/species/").json()
    planet_names = []
    while species['next'] is not None:
        for specie in species['results']:
            designation = specie.get('designation', None)
            classification = specie.get('classification', None)
            if designation == "sentient" or classification == "sentient":
                if specie['homeworld'] is not None:
                    planet = requests.get(specie['homeworld']).json()
                    planet_names.append(planet['name'])
                else:
                    continue
    planet_names = list(dict.fromkeys(planet_names))
    planet_names.sort()
    return planet_names
