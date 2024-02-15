# VWs-Chrono
This repo combines the VWs with Chrono simulation environment. We have several steps to put the recorded .bag data from real world into chrono simulation environment. 

### (1) Readbag.py

In this file, we use recorded elevation_map.bag from VWs to extract the elevation data. Then, we resize the data into $360 \times 360$. For the 'nan' value, we perform linear interpolation. After that, we normalize the filled data and create the .bmp image.

### (2) RigidHeightMap.json

After getting the elevation_data_linear.bmp, we use Json files to create the rigid rock terrain in the chrono environment. In the Json file, you can customize own size and height range. For example, the real world's map size is $2.88 \times 2.88$ meters and height range is from 0 to 1.33.

### (3) RockSimple.py

We create our own VWs simulated model from RCCar demo in the chrono. The size of chassis from X-axis is 0.63 meters. We utilize the PID control to make the VWs run on the rock terrain.

### (4) RocksVehicleJson.py

This file is 1:1 HMMWV vehicle model in the rock terrain from Json files.

### (5) RocksAPIJson.py

This file is 1:1 HMMWV vehicle model in the rock terrain from API provided by chrono.
