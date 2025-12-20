# Gazebo Simulation for Humanoid Robots

## Introduction to Gazebo

Gazebo is a powerful physics-based simulation environment that enables the testing and validation of robotic systems before deploying to real hardware. For humanoid robotics, Gazebo provides a realistic environment to test complex multi-degree-of-freedom robots with accurate physics simulation, sensor models, and environmental interaction.

> **Key Concept**: Gazebo facilitates safe, fast, cost-effective testing and experimentation before expensive hardware deployment, allowing for comprehensive validation of complex scenarios that would be impractical or unsafe to test on physical hardware.

Gazebo uses the Open Dynamics Engine (ODE), Bullet Physics, or DART for physics simulation, providing realistic interactions between robots and their environment. For humanoid robots, this includes accurate modeling of contact forces, friction, and dynamic behaviors that are crucial for stable locomotion and manipulation.

## Gazebo Architecture and Components

### Core Components

Gazebo consists of several interconnected components:

1. **Gazebo Server**: The physics simulation backend that handles physics calculations, sensor processing, and model updates
2. **Gazebo Client**: The visualization interface that renders the simulation environment
3. **Model Database**: A repository of pre-built models and environments that can be included in simulations
4. **Plugin System**: Extensible interface for custom physics, sensors, and controllers

### Simulation Loop

Gazebo operates on a simulation loop that includes:

1. **Physics Update**: Calculating forces, torques, and applying them to objects
2. **Sensor Update**: Processing sensor data based on the current state of the world
3. **Model Update**: Updating the state of all simulated objects
4. **Visualization**: Rendering the current state to the user interface

## Setting Up Humanoid Simulation in Gazebo

### Installing Gazebo and ROS 2 Integration

First, ensure you have Gazebo installed with ROS 2 integration:

```bash
# Install Gazebo Garden (or newer version)
sudo apt install ros-humble-gazebo-*

# Install ROS 2 Gazebo bridge
sudo apt install ros-humble-gazebo-ros-pkgs
```

### Creating a Basic Simulation Environment

Here's how to create a simple simulation world file:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Include a sky -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Add your humanoid robot -->
    <!-- This will be spawned via ROS 2 launch files -->
    
    <!-- Add obstacles for testing -->
    <model name="obstacle_1">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
  </world>
</sdf>
```

## Integrating Humanoid Robots with Gazebo

### Spawning Robots into Gazebo

To spawn your humanoid robot into Gazebo, you'll need to configure it with Gazebo-specific elements in your URDF. Here's an example:

```xml
<!-- Add this to your URDF file for Gazebo integration -->
<gazebo>
  <!-- Robot-wide settings -->
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find my_robot_description)/config/my_robot_controllers.yaml</parameters>
  </plugin>
</gazebo>

<!-- For each link, define Gazebo-specific properties -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>1.0</kd>        <!-- Contact damping -->
</gazebo>

<!-- For joints with controllers, define control interfaces -->
<gazebo>
  <plugin name="left_shoulder_pitch_controller" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_robot</robotNamespace>
  </plugin>
</gazebo>
```

### Controller Configuration

Create a controller configuration file (e.g., `config/my_robot_controllers.yaml`):

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    left_arm_controller:
      type: effort_controllers/JointGroupEffortController

    right_arm_controller:
      type: effort_controllers/JointGroupEffortController

    torso_controller:
      type: effort_controllers/JointGroupEffortController

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_pitch
      - left_shoulder_yaw
      - left_shoulder_roll
      - left_elbow

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_pitch
      - right_shoulder_yaw
      - right_shoulder_roll
      - right_elbow

torso_controller:
  ros__parameters:
    joints:
      - torso_yaw
      - torso_pitch
      - torso_roll
```

## Launching Simulation

### Creating a Launch File

Create a launch file to bring up Gazebo with your humanoid robot:

```python
# launch/humanoid_simulation.launch.py
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_robot_description = get_package_share_directory('my_robot_description')
    
    # World file
    world = os.path.join(
        get_package_share_directory('my_robot_description'),
        'worlds',
        'humanoid_world.sdf'
    )
    
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py'),
        ),
        launch_arguments={'world': world}.items(),
    )
    
    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_humanoid_robot',
            '-x', '0', '-y', '0', '-z', '1.0'  # Initial position
        ],
        output='screen'
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'use_sim_time': True}],
    )
    
    # Joint state publisher (for GUI control during testing)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        joint_state_publisher_gui,
    ])
```

## Advanced Simulation Features for Humanoids

### Physics Parameters for Humanoid Simulation

Tuning physics parameters is crucial for realistic humanoid simulation:

```xml
<!-- In your world file -->
<world name="humanoid_world">
  <!-- Physics engine configuration -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>  <!-- Smaller for more stability -->
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    
    <!-- ODE-specific parameters -->
    <ode>
      <solver>
        <type>quick</type>
        <iters>10</iters>  <!-- More iterations for stability -->
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>0.000001</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>100</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>
</world>
```

### Sensor Integration in Simulation

Humanoid robots require various sensors that must be accurately simulated:

```xml
<!-- Adding a camera to the head of the robot -->
<gazebo reference="head">
  <sensor name="camera" type="camera">
    <pose>0.05 0 0 0 0 0</pose>
    <camera name="head_camera">
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees in radians -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>head</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
    </plugin>
  </sensor>
</gazebo>

<!-- Adding an IMU to the torso for balance sensing -->
<gazebo reference="base_link">
  <sensor name="imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <pose>0 0 0.1 0 0 0</pose>
  </sensor>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <frame_name>base_link</frame_name>
    <topic>imu/data</topic>
  </plugin>
</gazebo>

<!-- Adding force/torque sensors to feet for balance control -->
<gazebo reference="left_foot">
  <sensor name="left_foot_ft_sensor" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
  </sensor>
  <plugin name="left_foot_ft_plugin" filename="libgazebo_ros_ft_sensor.so">
    <frame_name>left_foot</frame_name>
    <topic>left_foot/ft_sensor</topic>
  </plugin>
</gazebo>
```

## Control Strategies for Humanoid Simulation

### Joint Position Control

For basic testing of humanoid robot kinematics:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import math
import time

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Create publisher for joint trajectory commands
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, 
            '/my_robot/left_arm_controller/joint_trajectory', 
            10
        )
        
        # Timer to publish trajectory commands
        self.timer = self.create_timer(2.0, self.move_to_position)
        self.get_logger().info('Humanoid controller initialized')
        
    def move_to_position(self):
        # Create a joint trajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            'left_shoulder_pitch', 'left_shoulder_yaw', 'left_shoulder_roll',
            'left_elbow', 'left_wrist_pitch', 'left_wrist_yaw'
        ]
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [0.1, 0.2, 0.0, -0.5, 0.0, 0.0]  # Target joint positions
        point.velocities = [0.0] * len(point.positions)  # Zero velocity at target
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0
        
        trajectory.points.append(point)
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.header.frame_id = 'base_link'
        
        # Publish the trajectory
        self.trajectory_pub.publish(trajectory)
        self.get_logger().info(f'Published trajectory to joints: {trajectory.joint_names}')

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Balance Control Simulation

Implementing basic balance control for bipedal simulation:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3, WrenchStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy import signal

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')
        
        # Subscriptions
        self.imu_sub = self.create_subscription(
            Imu, '/my_robot/imu/data', self.imu_callback, 10)
        self.l_foot_force_sub = self.create_subscription(
            WrenchStamped, '/my_robot/left_foot/ft_sensor', 
            self.left_force_callback, 10)
        self.r_foot_force_sub = self.create_subscription(
            WrenchStamped, '/my_robot/right_foot/ft_sensor', 
            self.right_force_callback, 10)
            
        # Publishers for joint control
        self.l_leg_cmd_pub = self.create_publisher(
            Float64MultiArray, 
            '/my_robot/left_leg_controller/commands', 
            10)
        self.r_leg_cmd_pub = self.create_publisher(
            Float64MultiArray, 
            '/my_robot/right_leg_controller/commands', 
            10)
        
        # Initialize state
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.left_foot_force = Vector3()
        self.right_foot_force = Vector3()
        
        # PID parameters for balance control
        self.kp = 20.0  # Proportional gain
        self.kd = 5.0   # Derivative gain
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.balance_control_loop)
        
    def imu_callback(self, msg):
        # Convert quaternion to roll, pitch, yaw
        quat = msg.orientation
        self.current_roll, self.current_pitch, self.current_yaw = self.quat_to_euler(
            quat.w, quat.x, quat.y, quat.z)
    
    def left_force_callback(self, msg):
        self.left_foot_force = msg.wrench.force
        
    def right_force_callback(self, msg):
        self.right_foot_force = msg.wrench.force
        
    def quat_to_euler(self, w, x, y, z):
        # Convert quaternion to Euler angles
        import math
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def balance_control_loop(self):
        # Simple balance control based on IMU data
        target_roll = 0.0
        target_pitch = 0.0
        
        # Calculate errors
        roll_error = target_roll - self.current_roll
        pitch_error = target_pitch - self.current_pitch
        
        # PID control (simplified)
        roll_cmd = self.kp * roll_error
        pitch_cmd = self.kp * pitch_error
        
        # Publish control commands (simplified)
        left_cmd = Float64MultiArray()
        left_cmd.data = [roll_cmd, pitch_cmd, 0.0, 0.0, 0.0, 0.0]  # hip, knee, ankle, etc.
        self.l_leg_cmd_pub.publish(left_cmd)
        
        right_cmd = Float64MultiArray()
        right_cmd.data = [roll_cmd, pitch_cmd, 0.0, 0.0, 0.0, 0.0]
        self.r_leg_cmd_pub.publish(right_cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = BalanceController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Simulation Techniques

### Domain Randomization

To help bridge the sim-to-real gap, domain randomization can be used:

```xml
<!-- Randomizing physics parameters to make controllers more robust -->
<gazebo reference="left_upper_arm">
  <mu1>0.1 0.5</mu1>  <!-- Randomize friction coefficient -->
  <mu2>0.1 0.5</mu2>
  <material>Gazebo/Green</material>
  <kp>500000.0 2000000.0</kp>  <!-- Randomize stiffness -->
  <kd>0.5 2.0</kd>              <!-- Randomize damping -->
</gazebo>
```

### Sensor Noise Modeling

Realistic sensor noise is crucial for humanoid simulation:

```xml
<gazebo reference="head_camera">
  <sensor name="camera" type="camera">
    <camera name="head_camera">
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
  </sensor>
</gazebo>

<gazebo reference="imu">
  <sensor name="imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

## Simulation Workflows for Humanoid Development

### Development Cycle

1. **Design Phase**: Create URDF model with accurate inertial properties
2. **Simulation Phase**: Test control algorithms in Gazebo simulation
3. **Validation Phase**: Verify behavior in simulation with various scenarios
4. **Deployment Phase**: Transfer to real hardware with appropriate adjustments

### Testing Complex Scenarios

Gazebo allows for testing dangerous or expensive scenarios safely:

```bash
# Test walking on uneven terrain
ros2 launch my_robot_description uneven_terrain.launch.py

# Test manipulation with various objects
ros2 launch my_robot_description manipulation_test.launch.py

# Test navigation with obstacles
ros2 launch my_robot_description navigation_test.launch.py
```

## Performance Considerations

### Optimizing Simulation Speed

For humanoid robots with many degrees of freedom:

1. **Use simplified collision geometries** during development
2. **Adjust physics parameters** to balance accuracy and performance
3. **Use appropriate update rates** for different sensors
4. **Consider disabling visualization** for batch tests

### Real-time Factor

Monitor the real-time factor to ensure simulation is running efficiently:

```bash
# Check real-time factor in Gazebo
gz stats

# Optimize if real-time factor is too low
# Adjust physics parameters, simplify models, or reduce update rates
```

## Best Practices for Humanoid Simulation

### 1. Accurate Inertial Properties
Ensure URDF models have realistic inertial properties to match real-world behavior.

### 2. Appropriate Control Rates
Match control rates in simulation to what's achievable on real hardware.

### 3. Thorough Testing
Test not just nominal behaviors but also failure cases and edge conditions.

### 4. Gradual Complexity
Start with simple tests and gradually increase complexity.

### 5. Validation Against Real Data
Compare simulation results with real hardware data when possible.

## Troubleshooting Common Issues

### Collision Detection Problems
- Ensure collision geometries are properly defined
- Check for overlapping geometries
- Adjust contact parameters if needed

### Controller Instability
- Reduce control gains if experiencing oscillations
- Ensure physics update rate is sufficient for control rate
- Check joint limits and safety controllers

### Visualization Issues
- Ensure Gazebo client can connect to the server
- Check graphics drivers and hardware acceleration
- Verify model file paths are correct

## Conclusion

Gazebo simulation is an indispensable tool for developing humanoid robots, providing a safe, cost-effective environment to test complex behaviors before deploying to real hardware. For humanoid robotics specifically, Gazebo enables testing of balance, locomotion, and manipulation behaviors with realistic physics simulation.

Successfully simulating humanoid robots requires attention to detail in physics parameters, sensor modeling, and control integration. When done correctly, the simulation environment serves as a powerful tool for developing robust humanoid systems that can be confidently deployed to real hardware.

![Gazebo Humanoid Simulation](/img/gazebo-simulation-humanoid.jpg)
*Image Placeholder: Screenshot of a humanoid robot being simulated in Gazebo environment*

---

## Key Takeaways

- Gazebo provides realistic physics simulation for testing humanoid robots safely
- Proper integration with ROS 2 enables seamless control and sensor data flow
- Physics parameters and sensor noise must be carefully tuned for realistic simulation
- Balance and locomotion controllers can be tested extensively in simulation
- Domain randomization helps bridge the sim-to-real gap
- Proper simulation setup is essential for successful real-world deployment

## Next Steps

In the following chapter, we'll explore Unity as an alternative simulation environment, focusing on its visualization capabilities and different approaches to humanoid robot simulation.