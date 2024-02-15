import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os
import math as m
from pychrono.vehicle import RCCar

class ModifiedRCCar(RCCar):
    def __init__(self):
        super().__init__()  
        self.drive_type = "FWD"  # Set drive type to All Wheel Drive

    def SetDriveType(self, drive_type):
        self.drive_type = drive_type

def main():
    veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

    car = ModifiedRCCar()
    car.SetContactMethod(contact_method)
    car.SetChassisCollisionType(chassis_collision_type)
    car.SetChassisFixed(False) 
    car.SetInitPosition(chrono.ChCoordsysD(initLoc, initRot))
    car.SetTireType(tire_model)
    car.SetTireStepSize(tire_step_size)
    car.SetDriveType(drive_type)
    car.SetMaxMotorVoltageRatio(0.16)
    car.SetStallTorque(0.3)
    car.SetTireRollingResistance(0.06)
    car.Initialize()

    car.SetChassisVisualizationType(chassis_vis_type)
    car.SetSuspensionVisualizationType(suspension_vis_type)
    car.SetSteeringVisualizationType(steering_vis_type)
    car.SetWheelVisualizationType(wheel_vis_type)
    car.SetTireVisualizationType(tire_vis_type)

    init_speed = 3
    # Set initial speed
    car.GetChassisBody().SetPos_dt(chrono.ChVectorD(init_speed, 0, 0))
    # Set gravity
    car.GetSystem().Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
    # Lock Differential: make same wheel speed
    car.LockAxleDifferential(0, True)
    car.LockAxleDifferential(1, True)

    # Create the path-follower, cruise-control driver
    # Use a parameterized ISO double lane change (to left)
    path = veh.DoubleLaneChangePath(initLocPath, 13.5, 4.0, 11.0, 50.0, True)
    driver = veh.ChPathFollowerDriver(car.GetVehicle(), path, "my_path", target_speed)
    driver.GetSteeringController().SetLookAheadDistance(5)
    driver.GetSteeringController().SetGains(0.8, 0, 0)
    driver.GetSpeedController().SetGains(0.4, 0, 0)
    driver.Initialize()

    veh.SetDataPath("")
    # Create the customed terrain
    terrain = veh.RigidTerrain(car.GetSystem(), rigidterrain_file)
    terrain.Initialize()

    # # Create the rigid terrain
    # patch_mat = chrono.ChMaterialSurfaceNSC()
    # patch_mat.SetFriction(0.9)
    # patch_mat.SetRestitution(0.01)
    # terrain = veh.RigidTerrain(car.GetSystem())
    # patch = terrain.AddPatch(patch_mat, 
    #     chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.QUNIT), 
    #     100, 100)

    # patch.SetTexture(veh.GetDataFile("terrain/textures/concrete.jpg"), 200, 200)
    # patch.SetColor(chrono.ChColor(0.8, 0.8, 0.5))
    # terrain.Initialize()

    # Create the vehicle Irrlicht interface
    vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
    vis.SetWindowTitle('RC-Rocks')
    vis.SetWindowSize(1280, 1024)
    vis.SetChaseCamera(trackPoint, 6.0, 0.5)
    vis.Initialize()
    vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
    vis.AddLightDirectional()
    vis.AddSkyBox()
    vis.AttachVehicle(car.GetVehicle())

    # Simulation loop
    car.GetVehicle().EnableRealtime(True)
 
    while vis.Run() :
        time = car.GetSystem().GetChTime()

        #End simulation
        if (time >= t_end):
            vis.Quit()

        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        # Get driver inputs
        driver_inputs = driver.GetInputs()

        # Update modules (process inputs from other modules)
        driver.Synchronize(time)
        terrain.Synchronize(time)
        car.Synchronize(time, driver_inputs, terrain)
        vis.Synchronize(time, driver_inputs)

        # Advance simulation for one timestep for all modules
        driver.Advance(step_size)
        terrain.Advance(step_size)
        car.Advance(step_size)
        vis.Advance(step_size)

        # Extract vehicle state
        chassis = car.GetChassis()
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

        # #Log the state and action to file
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

# Terrain JSON specification file
Directory = "Data"
Filename = "RigidHeightMap.json"
rigidterrain_file = os.path.join(Directory, Filename)

# Initial vehicle and path location and orientation
initLoc = chrono.ChVectorD(-17.25, -0.3, 1.1) #forward-big/backward-small, left-big/right-small, height
initLocPath = chrono.ChVectorD(-17.25, -0.3, 1.1)
initRot = chrono.ChQuaternionD(1, 0, 0, 0)

# Vehicle target speed (cruise-control)
target_speed = 8

# Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
chassis_vis_type = veh.VisualizationType_PRIMITIVES
suspension_vis_type =  veh.VisualizationType_PRIMITIVES
steering_vis_type = veh.VisualizationType_PRIMITIVES
wheel_vis_type = veh.VisualizationType_PRIMITIVES
tire_vis_type = veh.VisualizationType_PRIMITIVES

# # Collision type for chassis (PRIMITIVES, MESH, or NONE)
chassis_collision_type = veh.CollisionType_MESH

# Steering type (PITMAN_ARM or PITMAN_ARM_SHAFTS)
steering_type = veh.SteeringTypeWV_PITMAN_ARM_SHAFTS

# Type of tire model (RIGID, PAC02)
tire_model = veh.TireModelType_RIGID

# Drive type (FWD, RWD, or AWD)
drive_type = veh.DrivelineTypeWV_AWD

# Point on chassis tracked by the camera
trackPoint = chrono.ChVectorD(3.2, 0, 0) #forward-big/backward-small

# Contact method
contact_method = chrono.ChContactMethod_NSC

# Simulation step sizes
step_size = 2e-3
tire_step_size = 1e-3

# Simulation end time
t_end = 1000

main()
