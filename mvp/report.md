# MVP Report: AI Drone Signal-Loss Mitigation System

## Executive Summary
 ### Problem:
 With the increasing use of drones in the modern day, drone pilots can lose communication with their drones while flying through high interference zones, areas with low signal or complex terrain, resulting in mission failure, and the drone activating failsafe measures.
 ### Project Goal: 
 Create an AI assisted drone safety system that monitors flight data to predict if the projected flight path is safe to continue on or should be altered then notifys the pilot of a suggested course of action.
## User & Use Case
### User:
Commercial drone pilots: Drones are frequently used in the construction, agriculture, delivery, and real estate industries to inspect, survey, deliver packages, and capture film.  
  
Military drone pilots: More and more air force missions are being carried out by Unmanned Aeriel Vehicles for surveillance and strike missions.
### Use Case:
Any situation where the drone is flying far enough away, low enough to the terrain, or through enough interference where signal loss is a major risk.
- Military surveillance missions are frequently across vast distances which would cause a signal loss to be catastrophic for the mission.  
- Commercial inspection of infastructure like bridges buidings or powerlines are typically located in high interference areas.
- delivery drones fly through dense urban areas where interference is high

Example use case: 
A drone pilot for the military is conducting a surveillance mission along a pre planned route while in route the drone sends flight and communication data to the pilot that indicates increased signal interference the data is cleaned of unneccesary information and sent to an AI model which predicts whether the trends of the data indicate a major problem ahead or if it is safe to continue. The model then communicates this prediction to the pilot by recommending an action such as slow down, re route the flight path, or return to base
## System Design
For the purpose of testing a drone simulation was utilized in this project to simulate multiple different drone flight paths with the option of adding interference obstacles along the flight path  


ArduPilot Simulation
         |
         V

         |
         V

         |
         V

         |
         V

         |
         V

## Data

## Models

## Evaluation

## Limitations & Risks

## Next Steps
