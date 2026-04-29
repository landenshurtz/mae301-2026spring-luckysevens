# MVP Report: AI Drone Signal-Loss Mitigation System

## Executive Summary
 ### Problem:
 With the increasing use of drones in the modern day, drone pilots can lose communication with their drones while flying through high interference zones, areas with low signal or complex terrain, resulting in mission failure, and the drone activating failsafe measures.
 ### Project Goal: 
 Create an AI assisted drone safety system that monitors flight data to predict if the projected flight path is safe to continue on or should be altered then notifys the pilot of a suggested course of action.
### MVP
Our MVP includes an ArduPilot Drone simulator that ouputs MAVLink telemetry Data, a proxy system that allows us to simulate communication obstacles along the drone flight path, a mission control AI backend that interprets and evaluates the communication data, assigns risk scores, decides on the best course of action then recommends that action to the pilot, and a data refiner that cleans flight data for the purpose of training a linear regression model that predicts future communication interferences based off the previous few seconds of data.

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
A drone pilot for the military is conducting a surveillance mission along a pre planned route while in route the drone sends flight and communication data to mission control the data is processed and sent to an AI model which predicts whether the trends of the data indicate a high interference area ahead or if it is safe to continue. The model then communicates this prediction to the pilot by recommending an action such as slow down, re route the flight path, or return to base.
## System Design

- ArduPilot Simulation  

- Proxy/ interference layer  

- Mission Control/ AI Backend

- Risk Scoring and Decision logic

- API endpoint
  
### Main Components 
The ArduPilot simulation outputs MAVLink telemetry data used to train the AI model and test the overall project  

The proxy system sits between the drone simulation and the mission control AI backend. Its purpose is to simulate interference obstacles such as packet loss, delay, and other impairments.  

Mission control is the main brain of the overall system it is resposnible for extracting drone state information(latitude, longitude, heading, battery percentage, etc.), parsing MAVLink messages, calculating communication risk scores, deciding on a recommended course of action and then outputting that decision to the user.  
- Risk scoring and decision logic: the system assigns risk scores to communication features as they come in (low, medium, high, and critical) and increases this score as it detects more severe problems such as high packet loss, high latency, or high jitter. these risk scores are then used to decide on the best course of action( continue, monitor, slow down, loiter and climb, return to launch, and resume auto)  
- API endpoint: the API endpoint publishes the drone system status, risk score, risk level and recommended action to the user



## Data
Our team collected data for this system through the use of a ArduPilot simulation environment. This simulation environment outputs MAVLink telemetry messages similar to how a real drone would allowing the team to test drone behavior easily without needing a physical drone.  
This project focuses on specifically communication risk variables that relate to whether the drone is likely to lose signal 
- risk score
- packet loss percentage
- latency
- jitter
- missed heartbeats
- packets dropped in a row
- dropout duration
- dropped packets
- returned packets
- heartbeat interval

The raw data is preprocessed using dataRefiner to remove unnecessary variables, such as windspeed, roll, pitch, and bearing. This simplifies the dataset and creates cleaner, more useful data for training the model. Cleaning the data involves removing empty fields and specific columns that contain unnecessary information. Simplifying the data involves downsampling the data to quarter second intervals and averaging certain columns over the previous second that contain an abnormal amount of noise.



## Models
The project contains two models 
- A linear regression model
- A real time rule based AI in mission control

### Linear Regression model  
The linear regression model is trained using the cleaned data files in order to predict future packet behavior based off of recent communicaton and flight data. This model gives the project a machine learning based foundation for prediciting communication problems.

### Rule Based AI backend   
The AI backend is a real-time model that uses a rule based scoring system to evaluate if the communication data it recieves risk level is low (stable), medium (link is degrading), high (signal loss is more likely), or critical (signal loss is highly likely). Based off the computed risk score the model recommends actions such as continue, monitor, slow down, loiter and climb, return to launch, and resume auto. These actions are specifically designed so they can help the drone or pilot take action before communication is lost. Additionally it explains to the pilot what is the main factor causing the risk of signal loss (elevated packet loss, high latency, etc.). This way the pilot not only knows that they need to take action to prevent communication loss but they also know why they need to take action.     
## Evaluation
### Current methods  
The current design utilizes predicition error metrics in the linear regression model to predict future packet values
### Comparison  
The most simple way to evaluate the teams design is to compare the results of flights through multiple different environments of an AI assisted pilot and a pilot not using AI assistance. Using metrics such as how often did signal loss occur, how long did the flight take, did the AI warn the pilot of any high risk areas, was the failsafe activated, did the drone reach its intended target.
### Qualitative 
A simple qualitative evaluation would be is the information presented to the pilot easy to understand and useful in helping them complete their mission.
## Limitations & Risks
### Simulation  
The use of the ArduPilot drone simulator is convenient for testing and training the models however simulations are always a little different from real world conditions because there is so many different factors effecting a drone in the real world that would be very difficult to fully simulate. Therefore a version of this design will require additional real world testing before widespread release.
### Models  
Linear Regression: The linear regression model being used is relatively simple and explainable which might struggle when presented with complex non linear communication loss patterns.
AI backend: In its current state the AI backend is rule based rather than a fully trained deep learning model.
### Integration 
The teams current project contains many different systems that all must communicate with each other accurately because there is so many different systems integrating them all to work together in real time will pose a reasonable technical challenge.
### Risks  
Risks associated with this project primarily boil down to incorrect predictions such as the system underestimating risk, warning the pilot to late, or recommending the pilot an action that fails to improve or worsens the communication problem.

## Next Steps
Next steps for the design include testing more advanced models integrating the predicition model into mission control, creating a visual display for the user that displays all relevant info they might need, and implementing real drone test data.  
The final goal is to design an AI drone safety assistant that monitors drone communication risk in real time and explains and advises the pilot on the best course of action in the event of weakening signal or triggering fail safe actions in the event of total signal failure.
