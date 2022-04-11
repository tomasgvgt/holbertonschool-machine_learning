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
    sentient_species_urls = []
    homeworld_urls = []
    planet_names = []
    while species['next'] is not None:
        for specie in species['results']:
            if specie['designation'] == "sentient":
                for people in specie['people']:
                    sentient_species_urls.append(people)
        species = requests.get(species['next']).json()
    for people_url in sentient_species_urls:
        subject = requests.get(people_url).json()
        homeworld_urls.append(subject['homeworld'])
    for planet_url in homeworld_urls:
        planet = requests.get(planet_url).json()
        planet_names.append(planet['name'])
    planet_names = list(dict.fromkeys(planet_names))
    print(len(planet_names))
    return planet_names
