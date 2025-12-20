# URDF for Humanoid Robots

## Understanding URDF: The Robot Description Format

The Unified Robot Description Format (URDF) is an XML-based format used to describe robotic systems in ROS. For humanoid robots, URDF is essential for defining the robot's physical structure, kinematic properties, visual appearance, and collision characteristics.

> **Key Concept**: URDF defines the physical and kinematic properties of a robot, enabling simulation, visualization, and control of robotic systems in ROS environments.

URDF files describe a robot as a collection of links connected by joints, forming a kinematic tree. For humanoid robots, this tree typically starts with a base link (torso) and branches out to form limbs with multiple degrees of freedom.

## URDF Structure for Humanoid Robots

### Basic Components

A humanoid URDF contains several key elements:

1. **Links**: Represent rigid bodies of the robot
2. **Joints**: Define connections between links with specific degrees of freedom
3. **Visual Elements**: Define how the robot appears in simulation and visualization
4. **Collision Elements**: Define collision properties for physics simulation
5. **Inertial Properties**: Define mass, center of mass, and inertia tensor for each link

### Link Definition

Each link in a humanoid robot requires several properties:

```xml
<link name="left_upper_arm">
  <!-- Inertial properties for physics simulation -->
  <inertial>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <mass value="2.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
  </inertial>
  
  <!-- Visual properties for rendering -->
  <visual>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/left_upper_arm.dae"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 0.8 1.0"/>
    </material>
  </visual>
  
  <!-- Collision properties for physics simulation -->
  <collision>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
  </collision>
</link>
```

### Joint Definition

Joints connect links and define their relative motion:

```xml
<joint name="left_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_upper_arm"/>
  <origin xyz="0.1 0.2 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-1.57" upper="1.57" effort="100.0" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Complete Humanoid Robot Example

Here's an example of a simplified humanoid robot structure:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.7 0.7 0.7 1.0"/>
  </material>
  
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0.3" />
      <inertia  ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 0" />
      <inertia  ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1" />
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="grey"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.6"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="10.0" velocity="1.0"/>
  </joint>
  
  <!-- Left Arm -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 0.15" />
      <inertia  ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001" />
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="green"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.15 0.3"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="1.0"/>
  </joint>
  
  <!-- Additional limbs would follow a similar pattern -->
  
</robot>
```

## Xacro: Enhancing URDF with Macros

Xacro (XML Macros) is an extension to URDF that allows for more complex and maintainable robot descriptions using macros and variables:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  
  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="arm_mass" value="1.0" />
  <xacro:property name="arm_length" value="0.3" />
  <xacro:property name="arm_radius" value="0.04" />
  
  <!-- Macro for an arm segment -->
  <xacro:macro name="arm_segment" params="side parent_link position_x position_y joint_limit_lower joint_limit_upper">
    <link name="${side}_upper_arm">
      <inertial>
        <mass value="${arm_mass}" />
        <origin xyz="0 0 ${arm_length/2}" />
        <inertia  ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001" />
      </inertial>
      
      <visual>
        <origin xyz="0 0 ${arm_length/2}"/>
        <geometry>
          <cylinder length="${arm_length}" radius="${arm_radius}"/>
        </geometry>
        <material name="green"/>
      </visual>
      
      <collision>
        <origin xyz="0 0 ${arm_length/2}"/>
        <geometry>
          <cylinder length="${arm_length}" radius="${arm_radius}"/>
        </geometry>
      </collision>
    </link>
    
    <joint name="${side}_shoulder_pitch" type="revolute">
      <parent link="${parent_link}"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${position_x} ${position_y} 0.3"/>
      <axis xyz="1 0 0"/>
      <limit lower="${joint_limit_lower}" upper="${joint_limit_upper}" effort="20.0" velocity="1.0"/>
    </joint>
  </xacro:macro>
  
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0.3" />
      <inertia  ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Use the macro to create arms -->
  <xacro:arm_segment side="left" parent_link="base_link" position_x="0.15" position_y="0.15" joint_limit_lower="-1.57" joint_limit_upper="1.57"/>
  <xacro:arm_segment side="right" parent_link="base_link" position_x="0.15" position_y="-0.15" joint_limit_lower="-1.57" joint_limit_upper="1.57"/>
  
</robot>
```

## Advanced URDF Features for Humanoid Robots

### Transmission Elements

For simulation and control, URDF can include transmission elements that define how joints are actuated:

```xml
<transmission name="left_shoulder_pitch_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_shoulder_pitch">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_shoulder_pitch_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Elements

When using URDF with Gazebo simulation, additional elements can be included:

```xml
<gazebo reference="left_upper_arm">
  <material>Gazebo/Green</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>1.0</kd>
</gazebo>
```

## Kinematic and Dynamic Considerations

### Denavit-Hartenberg Parameters

For complex humanoid robots, kinematic chains can be defined using Denavit-Hartenberg parameters, though URDF uses a different approach with joint origins and axes.

### Center of Mass and Stability

Humanoid robots require careful attention to center of mass for stability:

```xml
<!-- Example of how center of mass affects stability -->
<inertial>
  <mass value="10.0" />
  <origin xyz="0 0 0.3" />  <!-- Lower CoM for more stability -->
  <inertia  ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
</inertial>
```

## ROS 2 Integration

### Robot State Publisher

To publish the robot state based on joint values:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

class SimpleStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')
        
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        self.joint_names = ['joint1', 'joint2', 'joint3']  # Define based on your robot
        self.joint_values = [0.0, 0.0, 0.0]  # Current joint positions
    
    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_values
        
        # Publish joint states
        self.joint_state_pub.publish(msg)
```

## Humanoid-Specific Challenges

### Balance and Locomotion

Humanoid robots face unique challenges related to balance and bipedal locomotion:

1. **Zero Moment Point (ZMP)**: Critical for maintaining balance during walking
2. **Center of Mass Control**: Essential for stable movement
3. **Dynamic Walking**: Requires sophisticated control algorithms

### Degrees of Freedom

Humanoid robots typically have many degrees of freedom (DOF), which requires:
- Sophisticated inverse kinematics
- Careful trajectory planning
- Advanced control algorithms

## URDF Best Practices

### 1. Consistent Naming
Use a consistent naming convention for links and joints (e.g., `left_arm_shoulder_pitch`).

### 2. Appropriate Inertial Properties
Ensure inertial properties match the physical robot as closely as possible for accurate simulation.

### 3. Collision vs. Visual Geometry
Use simplified collision geometry for performance while maintaining detailed visual geometry for rendering.

### 4. Hierarchical Organization
Organize complex robots into logical groups and use Xacro macros to avoid repetition.

### 5. Documentation
Comment your URDF files to explain the structure and any non-obvious design decisions.

## Validation Tools

ROS provides several tools for validating URDF files:

1. **check_urdf**: Checks the structure of a URDF file
2. **RViz**: Visualizes the robot in 3D
3. **Gazebo**: Tests the robot in a physics simulation environment

```bash
# Check URDF validity
check_urdf /path/to/robot.urdf

# Visualize in RViz
ros2 run rviz2 rviz2

# Launch in Gazebo
ros2 launch gazebo_ros gazebo.launch.py
```

## Conclusion

URDF is fundamental to humanoid robotics development in the ROS ecosystem. It provides the specification needed for simulation, visualization, and control of complex multi-degree-of-freedom robots. Understanding how to properly define the physical and kinematic properties of a humanoid robot in URDF is essential for successful robotic applications.

Properly designed URDF files enable accurate simulation, efficient collision detection, and effective control of humanoid robots. Combined with ROS 2's communication infrastructure, URDF provides the foundation for developing sophisticated embodied AI systems.

![Humanoid Robot URDF](/img/humanoid-urdf-structure.jpg)
*Image Placeholder: Diagram showing a humanoid robot's URDF structure with links, joints, and kinematic chains*

---

## Key Takeaways

- URDF defines the physical and kinematic properties of robots in XML format
- For humanoid robots, URDF includes links, joints, visual properties, collision properties, and inertial properties
- Xacro enhances URDF with macros and variables for more maintainable robot descriptions
- Transmission elements connect joints to actuators for simulation and control
- Humanoid robots require special attention to kinematic chains and balance considerations
- Validation tools help ensure URDF files are correctly structured
- Proper URDF design is essential for accurate simulation and control

## Next Steps

In the following chapter, we'll explore how to simulate humanoid robots in Gazebo, using the URDF models we've created to test and validate our robot designs in a physics-based environment.