import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os

def main():

    my_feda = veh.WheeledVehicle(vehicle_file, contact_method)
    my_feda.Initialize(chrono.ChCoordsysD(initLoc, initRot))

    my_feda.SetChassisVisualizationType(chassis_vis_type)
    my_feda.SetChassisRearVisualizationType(chassis_vis_rear_type)
    my_feda.SetSuspensionVisualizationType(suspension_vis_type)
    my_feda.SetSteeringVisualizationType(steering_vis_type)
    my_feda.SetWheelVisualizationType(wheel_vis_type)

    # Create and initialize the tires
    for axle in my_feda.GetAxles() :
        for wheel in axle.GetWheels() :
            tire = veh.ReadTireJSON(tire_file)
            my_feda.InitializeTire(tire, wheel, veh.VisualizationType_MESH)

    # Create and initialize the powertrain system
    engine = veh.ReadEngineJSON(engine_file)
    transmission = veh.ReadTransmissionJSON(transmission_file)
    powertrain = veh.ChPowertrainAssembly(engine, transmission)
    my_feda.InitializePowertrain(powertrain)

    my_feda.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.VIS_Contacts)

    init_speed = 1
    # Set initial speed
    my_feda.GetChassisBody().SetPos_dt(chrono.ChVectorD(init_speed, 0, 0))
    # Set gravity
    my_feda.GetSystem().Set_G_acc(chrono.ChVectorD(0, 0, -9.81))

    # Create the path-follower, cruise-control driver
    # Use a parameterized ISO double lane change (to left)
    path = veh.DoubleLaneChangePath(initLocPath, 13.5, 4.0, 11.0, 50.0, True)
    driver = veh.ChPathFollowerDriver(my_feda, path, "my_path", target_speed)
    driver.GetSteeringController().SetLookAheadDistance(5)
    driver.GetSteeringController().SetGains(0.8, 0, 0)
    driver.GetSpeedController().SetGains(0.4, 0, 0)
    driver.Initialize()

    # Create the terrain
    terrain = veh.RigidTerrain(my_feda.GetSystem(), rigidterrain_file)
    terrain.Initialize()

    # Create the vehicle Irrlicht interface
    vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
    vis.SetWindowTitle('VW-Rocks')
    vis.SetWindowSize(1280, 1024)
    vis.SetChaseCamera(trackPoint, 6.0, 0.5)
    vis.Initialize()
    vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
    vis.AddLightDirectional()
    vis.AddSkyBox()
    vis.AttachVehicle(my_feda)

    # Simulation loop
    my_feda.EnableRealtime(True)
    while vis.Run() :
        # Render scene
        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        # Get driver inputs
        driver_inputs = driver.GetInputs()

        # Update modules (process inputs from other modules)
        time = my_feda.GetSystem().GetChTime()
        driver.Synchronize(time)
        terrain.Synchronize(time)
        my_feda.Synchronize(time, driver_inputs, terrain)
        vis.Synchronize(time, driver_inputs)

        # Advance simulation for one timestep for all modules
        driver.Advance(step_size)
        terrain.Advance(step_size)
        my_feda.Advance(step_size)
        vis.Advance(step_size)

        # Extract vehicle state
        chassis = my_feda.GetChassis()
        pos = chassis.GetPos()  # Position
        rot = chassis.GetRot()   # Orientation (as a quaternion)
        cur_vel = chassis.GetBody().GetPos_dt()  # Velocity

        # Convert the quaternion to roll, pitch, and yaw
        euler_angles_vec = chrono.Q_to_Euler123(rot)
        euler_angles = [euler_angles_vec.x, euler_angles_vec.y, euler_angles_vec.z]
        roll = euler_angles[0]
        pitch = euler_angles[1]
        yaw = euler_angles[2]

        # Extract vehicle actions
        steering = driver_inputs.m_steering
        throttle = driver_inputs.m_throttle
        brake = driver_inputs.m_braking

        # Print or log the state and action
        print(f"Time: {time:.2f}, "
              f"Position: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}), "
              f"Orientation (RPY): ({roll:.4f}, {pitch:.4f}, {yaw:.4f}), "
              f"Velocity: ({cur_vel.x:.4f}, {cur_vel.y:.4f}, {cur_vel.z:.4f})")
        print(f"Steering: {steering:.4f}, Throttle: {throttle:.4f}, Brake: {brake:.4f}")

        # with open('Vehicle_Output.txt', 'a') as file:
        #     output_string = (
        #         f"Time: {time:.2f}, "
        #         f"Position: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}), "
        #         f"Orientation (RPY): ({roll:.4f}, {pitch:.4f}, {yaw:.4f}), "
        #         f"Velocity: ({cur_vel.x:.4f}, {cur_vel.y:.4f}, {cur_vel.z:.4f})\n"
        #         f"Steering: {steering:.4f}, Throttle: {throttle:.4f}, Brake: {brake:.4f}\n\n"
        #     )
        #     file.write(output_string)

    return 0

veh.SetDataPath("")
# Terain JSON specification file
Directory = "Data"
Filename = "RigidHeightMapOriginal.json"
rigidterrain_file = os.path.join(Directory, Filename)

# HMMWV specification files (vehicle, powertrain, and tire models)
vehicle_file = veh.GetDataFile('Data/HMMWV/HMMWV_Vehicle_4WD.json')
engine_file = veh.GetDataFile('Data/HMMWV/HMMWV_EngineSimpleMap.json')
transmission_file = veh.GetDataFile('Data/HMMWV/HMMWV_AutomaticTransmissionSimpleMap.json')
tire_file = veh.GetDataFile("Data/HMMWV/HMMWV_RigidTire.json")

# Initial vehicle location and orientation
initLoc = chrono.ChVectorD(-20, 0, 1)
initLocPath = chrono.ChVectorD(-20, 0, 1)
initRot = chrono.ChQuaternionD(1, 0, 0, 0)

# Vehicle target speed (cruise-control)
target_speed = 6

# Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
chassis_vis_type = veh.VisualizationType_MESH
chassis_vis_rear_type = veh.VisualizationType_PRIMITIVES
suspension_vis_type =  veh.VisualizationType_PRIMITIVES
steering_vis_type = veh.VisualizationType_MESH
wheel_vis_type = veh.VisualizationType_MESH

# Point on chassis tracked by the camera
trackPoint = chrono.ChVectorD(0.0, 0.0, 1.4)

# Contact method
contact_method = chrono.ChContactMethod_NSC

# Simulation step sizes
step_size = 2e-3

main()
