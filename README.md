
1. SIMULATION
   Need to create a linux virtual environment and install ubuntu since ArduPilo truns on linux and not winodws (also protects your computer)
   To run the simulation you will need to instal ceritan python libraries (cant remebver off the the dome but definitly pymavlink)
2. INTERFACES
   I need to work on this still, 14550 should be simulation ans 14560 should be proxy (trafic from mission control and the sim go through), ai proxy and mission control should be on the same level wiht the interference proxy inbetween it and the sim
4. Running a simulation
   Start ardupilot with sim vehicle(autocopter)
   wait for Ardupilot to load
   Run proxy's (AI and interference)
   Run mission control (Parker will probably want to program a auto run/export feature
5. Data
   extract/parse as whatever file you think is best (.txt seems best), with the data you think is most important for the model (time, hearatbeats, packet count etc.)
