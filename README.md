
1. SIMULATION
   Need to create a linux virtual environment and install ubuntu since ArduPilo truns on linux and not windows (also protects your computer)
   To run the simulation you will need to install certain python libraries (can't remember but definitly pymavlink, check dependencies in the project)
2. INTERFACES
   14550 should be simulation and 14560 should be proxy (traffic from mission control and the simulation go through), ai proxy and mission control should be on the same level wiht the interference proxy inbetween it and the sim
3. Running a simulation
   Start ardupilot with sim vehicle(autocopter)
   wait for Ardupilot to load
   Run proxy's (AI and interference)
   Run mission control
   It'll output in validation_run.txt
