import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os

def main():
    veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

    my_feda = veh.HMMWV_Full()
    my_feda.SetContactMethod(contact_method)
    my_feda.SetChassisCollisionType(chassis_collision_type)
    my_feda.SetChassisFixed(False) 
    my_feda.SetInitPosition(chrono.ChCoordsysD(initLoc, initRot))
    my_feda.SetEngineType(veh.EngineModelType_SIMPLE_MAP)
    my_feda.SetTransmissionType(veh.TransmissionModelType_SIMPLE_MAP)
    my_feda.SetTireType(tire_model)
    my_feda.SetTireStepSize(tire_step_size)
    my_feda.Initialize()

    my_feda.SetChassisVisualizationType(chassis_vis_type)
    my_feda.SetSuspensionVisualizationType(suspension_vis_type)
    my_feda.SetSteeringVisualizationType(steering_vis_type)
    my_feda.SetWheelVisualizationType(wheel_vis_type)
    my_feda.SetTireVisualizationType(tire_vis_type)

    init_speed = 2
    # Set initial speed
    my_feda.GetChassisBody().SetPos_dt(chrono.ChVectorD(init_speed, 0, 0))
    # Set gravity
    my_feda.GetSystem().Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
    my_feda.LockAxleDifferential(0, True)
    my_feda.LockAxleDifferential(1, True)

    # Create the path-follower, cruise-control driver
    # Use a parameterized ISO double lane change (to left)
    path = veh.DoubleLaneChangePath(initLocPath, 13.5, 4.0, 11.0, 50.0, True)
    driver = veh.ChPathFollowerDriver(my_feda.GetVehicle(), path, "my_path", target_speed)
    driver.GetSteeringController().SetLookAheadDistance(5)
    driver.GetSteeringController().SetGains(0.8, 0, 0)
    driver.GetSpeedController().SetGains(0.4, 0, 0)
    driver.Initialize()

    veh.SetDataPath("")

    # Create the terrain
    terrain = veh.RigidTerrain(my_feda.GetSystem(), rigidterrain_file)
    # Non-Smooth Contact: Rigid
    if (contact_method == chrono.ChContactMethod_NSC):
        patch_mat = chrono.ChMaterialSurfaceNSC()
        patch_mat.SetFriction(0.6)
        patch_mat.SetRestitution(0.01)
    # Smoothed Particle Hydrodynamics: soft materials
    elif (contact_method == chrono.ChContactMethod_SMC):
        patch_mat = chrono.ChMaterialSurfaceSMC()
        patch_mat.SetFriction(0.9)
        # coefficient of restitution
        patch_mat.SetRestitution(0.01)
        patch_mat.SetYoungModulus(2e7)

    terrain.Initialize()

    # Create the vehicle Irrlicht interface
    vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
    vis.SetWindowTitle('HMMWV-Rocks')
    vis.SetWindowSize(1280, 1024)
    vis.SetChaseCamera(trackPoint, 6.0, 0.5)
    vis.Initialize()
    vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
    vis.AddLightDirectional()
    vis.AddSkyBox()
    vis.AttachVehicle(my_feda.GetVehicle())

    # Simulation loop
    my_feda.GetVehicle().EnableRealtime(True)

    step_number = 0
 
    while vis.Run() :
        time = my_feda.GetSystem().GetChTime()

        #End simulation
        if (time >= t_end):
            break

        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        # Get driver inputs
        driver_inputs = driver.GetInputs()

        # Update modules (process inputs from other modules)
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

        # Increment frame number
        step_number += 1

    return 0

# Terain JSON specification file
Directory = "Data"
Filename = "RigidHeightMapOriginal.json"
rigidterrain_file = os.path.join(Directory, Filename)

# Initial vehicle location and orientation
initLoc = chrono.ChVectorD(-20, -3, 1.5)
initLocPath = chrono.ChVectorD(-20, -3, 1.5)
initRot = chrono.ChQuaternionD(1, 0, 0, 0)

# Vehicle target speed (cruise-control)
target_speed = 6

# Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
chassis_vis_type = veh.VisualizationType_MESH
suspension_vis_type =  veh.VisualizationType_PRIMITIVES
steering_vis_type = veh.VisualizationType_PRIMITIVES
wheel_vis_type = veh.VisualizationType_MESH
tire_vis_type = veh.VisualizationType_MESH 

# # Collision type for chassis (PRIMITIVES, MESH, or NONE)
chassis_collision_type = veh.CollisionType_MESH

# Type of powertrain models (SHAFTS, SIMPLE)
engine_model = veh.EngineModelType_SHAFTS
transmission_model = veh.TransmissionModelType_SHAFTS

# Drive type (FWD, RWD, or AWD)
drive_type = veh.DrivelineTypeWV_AWD

# Steering type (PITMAN_ARM or PITMAN_ARM_SHAFTS)
steering_type = veh.SteeringTypeWV_PITMAN_ARM

# Type of tire model (RIGID, PAC02)
tire_model = veh.TireModelType_RIGID

# Point on chassis tracked by the camera
trackPoint = chrono.ChVectorD(0.0, 0.0, 1.4)

# Contact method
contact_method = chrono.ChContactMethod_NSC

# Simulation step sizes
step_size = 2e-3
tire_step_size = step_size

# Simulation end time
t_end = 10000

# Time interval between two render frames
render_step_size = 1.0 / 5000;  # FPS = 5000

main()
