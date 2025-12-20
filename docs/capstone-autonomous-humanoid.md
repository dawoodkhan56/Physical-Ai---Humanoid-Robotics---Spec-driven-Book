# Capstone: The Autonomous Humanoid System

## Introduction: Bringing It All Together

The autonomous humanoid system represents the ultimate integration of all the concepts explored throughout this book. This capstone project combines the robotic nervous system (ROS 2), digital twin environment (simulation), AI-robot brain (NVIDIA Isaac), vision-language-action capabilities, and conversational interfaces into a unified system capable of operating independently in real-world environments.

> **Key Concept**: The autonomous humanoid system integrates all components—sensing, perception, reasoning, action, and communication—into a cohesive platform that can receive voice commands, plan actions, navigate obstacles, identify objects, and manipulate them in simulation before deployment to real hardware.

This capstone project demonstrates how the individual modules covered in previous chapters work together to create a sophisticated embodied AI system. The system will be able to:
- Receive voice commands through conversational interfaces
- Process and understand these commands using VLA systems
- Plan navigation paths while avoiding obstacles
- Identify and locate objects in its environment
- Manipulate objects with its humanoid arms and hands
- Maintain coherent conversations while performing tasks

## System Architecture Overview

### High-Level System Components

The autonomous humanoid system consists of several interconnected subsystems that work together:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Autonomous Humanoid System                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│  │  Perception     │    │  Reasoning &     │    │  Action &        │    │
│  │  Subsystem      │    │  Planning        │    │  Control         │    │
│  │                 │    │  Subsystem       │    │  Subsystem       │    │
│  │ • Vision (RGB-D)│    │ • VLA Processing │    │ • Motion         │    │
│  │ • Audio Input   │    │ • Natural        │    │   Control        │    │
│  │ • IMU/Sensors   │    │   Language       │    │ • Manipulation   │    │
│  │ • 3D Mapping    │    │ • Task Planning  │    │ • Locomotion     │    │
│  └─────────────────┘    │ • Path Planning  │    │ • Safety Control │    │
│                         └──────────────────┘    └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
               │                         │                      │
               ▼                         ▼                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     Control & Coordination Layer                    │
    │  • ROS 2 Middleware for component communication                     │
    │  • Behavior Trees for task orchestration                            │
    │  • State Management for system awareness                            │
    │  • Safety Monitoring & Emergency Control                            │
    └─────────────────────────────────────────────────────────────────────┘
```

### Integration Architecture

The system follows a modular architecture built on ROS 2, with specialized packages for each function:

1. **Sensing Layer**: Collects data from various sensors (cameras, microphones, IMU, joint encoders)
2. **Perception Layer**: Processes sensor data to extract meaningful information
3. **Cognition Layer**: Interprets commands, plans actions, makes decisions
4. **Action Layer**: Executes physical actions and movements
5. **Communication Layer**: Handles human-robot interaction

## Hardware and Software Requirements

### Hardware Specifications

For a full-featured humanoid robot, consider these specifications:

```yaml
# Hardware requirements for autonomous humanoid system
humanoid_robot:
  # General specifications
  height: "1.5 meters"
  weight: "30 kg (approximate)"
  battery_life: "4 hours continuous operation"
  operating_system: "Real-time Linux (Ubuntu 20.04/22.04 LTS)"
  
  # Computing platform (recommended)
  computer:
    type: "NVIDIA Jetson Orin AGX or equivalent"
    cpu: "ARM Cortex-A78AE (8-core)"
    gpu: "NVIDIA Ampere architecture, 2048 CUDA cores"
    memory: "32GB LPDDR5"
    storage: "64GB eMMC + microSD slot"
    power: "up to 60W (MAX mode)"
  
  # Alternative computing platform
  alternative_computer:
    type: "NVIDIA RTX-equipped workstation"
    cpu: "Intel i7 / AMD Ryzen 7 (8+ cores)"
    gpu: "RTX 3080 / RTX 4080 or better"
    memory: "32GB DDR4+"
    storage: "1TB NVMe SSD"
  
  # Actuation system
  actuators:
    count: "28+ degrees of freedom"
    type: "smart servos with position/velocity/current feedback"
    torque: "varies by joint (hip: 50 Nm, ankle: 20 Nm, etc.)"
    communication: "CAN bus or EtherCAT"
  
  # Sensor system
  sensors:
    cameras:
      - location: "head, stereo pair"
        resolution: "1280x720 or higher"
        fov: "90 degrees"
        type: "global shutter for minimal motion distortion"
      - location: "chest, depth camera"
        type: "Intel Realsense D435i or similar"
        depth_range: "0.2m to 10m"
    
    audio:
      microphones: "4+ microphone array for beamforming"
      speakers: "2W stereo speakers"
      
    inertial:
      imu: "3-axis gyroscope, 3-axis accelerometer"
      inclinometer: "for balance monitoring"
      
    tactile:
      hand_sensors: "force/torque sensors in wrists"
      foot_sensors: "6-axis force/torque sensors in feet"
```

### Software Stack

```yaml
# Software requirements for autonomous humanoid system
software_stack:
  # Core robotics framework
  ros2:
    distribution: "Humble Hawksbill"
    packages: 
      - "navigation2"
      - "moveit2"
      - "vision_opencv"
      - "image_transport"
      - "tf2"
  
  # NVIDIA Isaac ecosystem
  isaac:
    - "isaac_ros_common"
    - "isaac_ros_visual_slam"
    - "isaac_ros_pose_graph"
    - "isaac_ros_realsense"
    - "isaac_ros_occupancy_grid_localizer"
  
  # AI and perception
  ai_frameworks:
    - "PyTorch 2.0+"
    - "TensorRT 8.6+"  # For optimized inference
    - "OpenCV 4.8+"    # For computer vision
    - "transformers"   # For NLP components
    - "espnet"         # For speech processing
  
  # Simulation and development
  simulation:
    - "Isaac Sim (or Gazebo Garden)"
    - "RViz2"
    - "rqt"
  
  # Communication and deployment
  deployment_tools:
    - "Docker"
    - "NVIDIA Container Toolkit"
    - "Git LFS for large model files"
```

## Core System Implementation

### Main Control Node

The main control node orchestrates the entire autonomous humanoid system:

```python
#!/usr/bin/env python3
"""
Main control node for the autonomous humanoid system.
This node coordinates all subsystems to achieve autonomous behavior.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, Imu, JointState
from nav_msgs.msg import Odometry, Path
from builtin_interfaces.msg import Time
from action_msgs.msg import GoalStatus

from humanoid_interfaces.msg import (
    HumanoidCommand, 
    HumanoidStatus, 
    ObjectDetection, 
    HumanoidAction
)
from humanoid_interfaces.action import (
    NavigateToPose,
    ManipulateObject,
    SpeakText
)

import numpy as np
import threading
import time
import json
from typing import Dict, List, Optional, Any


class AutonomousHumanoidController(Node):
    def __init__(self):
        super().__init__('autonomous_humanoid_controller')
        
        # QoS profiles
        self.qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Subscribers - receiving data from all subsystems
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/head_camera/rgb/image_raw', self.camera_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/head_camera/depth/image_raw', self.depth_callback, 10)
        
        # Command and status publishers
        self.status_pub = self.create_publisher(
            HumanoidStatus, '/humanoid/status', self.qos_profile)
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(
            String, '/tts/input', 10)
        
        # Action clients for complex behaviors
        self.nav_client = self.create_client(
            NavigateToPose, '/humanoid/navigate_to_pose')
        self.manip_client = self.create_client(
            ManipulateObject, '/humanoid/manipulate_object')
        self.speak_client = self.create_client(
            SpeakText, '/humanoid/speak_text')
        
        # Wait for action servers to be available
        self.nav_client.wait_for_service()
        self.manip_client.wait_for_service()
        self.speak_client.wait_for_service()
        
        # Internal state
        self.robot_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0]),  # quaternion
            'joint_positions': {},
            'joint_velocities': {},
            'battery_level': 100.0,
            'active_action': None,
            'last_command_time': self.get_clock().now(),
            'emergency_stop': False,
            'system_health': 'nominal'
        }
        
        # Task queue for autonomous behavior
        self.task_queue = []
        self.current_task = None
        
        # Control loop timer
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # Safety timer - runs at higher frequency for critical safety checks
        self.safety_timer = self.create_timer(0.01, self.safety_loop)
        
        # Initialize state
        self.initialize_system()
        
        self.get_logger().info('Autonomous Humanoid Controller initialized')
    
    def initialize_system(self):
        """Initialize the humanoid system and check all components"""
        self.get_logger().info('Initializing autonomous humanoid system...')
        
        # Check sensor availability
        sensors_available = self.check_sensor_availability()
        if not sensors_available:
            self.get_logger().error('Critical sensors not available, stopping initialization')
            self.robot_state['system_health'] = 'degraded'
            return False
        
        # Initialize perception components
        self.initialize_perception()
        
        # Initialize navigation components
        self.initialize_navigation()
        
        # Publish initial status
        self.publish_status()
        
        self.get_logger().info('Autonomous humanoid system initialized successfully')
        self.robot_state['system_health'] = 'nominal'
        return True
    
    def check_sensor_availability(self) -> bool:
        """Check if critical sensors are available"""
        # In practice, you'd check for actual sensor data
        # For demonstration, we'll assume sensors are available
        return True
    
    def initialize_perception(self):
        """Initialize perception components"""
        self.get_logger().info('Initializing perception components...')
        # Initialize object detection
        # Initialize SLAM system
        # Initialize person detection
        pass
    
    def initialize_navigation(self):
        """Initialize navigation components"""
        self.get_logger().info('Initializing navigation components...')
        # Initialize costmap
        # Initialize path planner
        # Set initial position
        pass
    
    def odom_callback(self, msg: Odometry):
        """Handle odometry updates"""
        self.robot_state['position'] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y, 
            msg.pose.pose.position.z
        ])
        
        self.robot_state['orientation'] = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
    
    def joint_state_callback(self, msg: JointState):
        """Handle joint state updates"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.robot_state['joint_positions'][name] = msg.position[i]
            if i < len(msg.velocity):
                self.robot_state['joint_velocities'][name] = msg.velocity[i]
    
    def imu_callback(self, msg: Imu):
        """Handle IMU data for balance and orientation"""
        # Check for excessive tilting - safety consideration
        orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        
        # Convert quaternion to roll/pitch for balance check
        roll, pitch, yaw = self.quaternion_to_euler(orientation)
        
        # Safety check: if robot is tilting too much, trigger emergency stop
        tilt_threshold = 0.5  # radians
        if abs(roll) > tilt_threshold or abs(pitch) > tilt_threshold:
            self.get_logger().warn('Excessive tilt detected, triggering safety protocol')
            self.emergency_stop()
    
    def camera_callback(self, msg: Image):
        """Handle camera image for visual perception"""
        # In a real system, this would trigger visual processing
        # For now, just log that image was received
        timestamp = self.get_clock().now().seconds_nanoseconds()
        self.get_logger().debug(f'Camera image received at {timestamp}')
    
    def depth_callback(self, msg: Image):
        """Handle depth image for 3D perception"""
        # In a real system, this would be used for obstacle detection
        # For now, just log that depth image was received
        timestamp = self.get_clock().now().seconds_nanoseconds()
        self.get_logger().debug(f'Depth image received at {timestamp}')
    
    def control_loop(self):
        """Main control loop running at 10Hz"""
        # Update current time
        current_time = self.get_clock().now()
        self.robot_state['last_command_time'] = current_time
        
        # Check if we have tasks to process
        if self.task_queue and not self.current_task:
            self.process_next_task()
        
        # Monitor system health
        self.monitor_system_health()
        
        # Publish status
        self.publish_status()
        
        # Process any queued commands
        self.process_command_queue()
    
    def safety_loop(self):
        """Critical safety loop running at 100Hz"""
        # Check for emergency conditions
        if self.robot_state['emergency_stop']:
            self.emergency_stop()
            return
        
        # Check joint limits
        self.check_joint_limits()
        
        # Check for collisions (would use costmap in real system)
        self.check_for_collisions()
    
    def process_next_task(self):
        """Process the next task in the queue"""
        if not self.task_queue:
            return
        
        task = self.task_queue.pop(0)
        self.current_task = task
        
        task_type = task.get('type', '')
        
        if task_type == 'navigation':
            self.execute_navigation_task(task)
        elif task_type == 'manipulation':
            self.execute_manipulation_task(task)
        elif task_type == 'conversation':
            self.execute_conversation_task(task)
        else:
            self.get_logger().warn(f'Unknown task type: {task_type}')
            self.current_task = None
    
    def execute_navigation_task(self, task: Dict):
        """Execute a navigation task"""
        goal_pose = task.get('goal_pose', {})
        if not goal_pose:
            self.get_logger().error('Navigation task missing goal pose')
            return
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_pose.get('x', 0.0)
        goal_msg.pose.pose.position.y = goal_pose.get('y', 0.0)
        goal_msg.pose.pose.position.z = goal_pose.get('z', 0.0)
        
        # Send navigation goal
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_completed_callback)
    
    def navigation_completed_callback(self, future):
        """Handle navigation completion"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info('Navigation goal accepted')
                # Wait for result
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(self.navigation_result_callback)
        except Exception as e:
            self.get_logger().error(f'Navigation goal failed: {e}')
            self.current_task = None
    
    def navigation_result_callback(self, future):
        """Handle navigation result"""
        try:
            result = future.result().result
            status = future.result().status
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Navigation completed successfully')
            else:
                self.get_logger().warn(f'Navigation failed with status: {status}')
        except Exception as e:
            self.get_logger().error(f'Navigation result error: {e}')
        
        self.current_task = None
    
    def execute_manipulation_task(self, task: Dict):
        """Execute a manipulation task"""
        object_name = task.get('object_name', '')
        action = task.get('action', '')
        
        if not object_name or not action:
            self.get_logger().error('Manipulation task missing required parameters')
            return
        
        # Create manipulation goal
        goal_msg = ManipulateObject.Goal()
        goal_msg.object_name = object_name
        goal_msg.action = action
        
        # Send manipulation goal
        future = self.manip_client.send_goal_async(goal_msg)
        future.add_done_callback(self.manipulation_completed_callback)
    
    def execute_conversation_task(self, task: Dict):
        """Execute a conversation task"""
        text_to_speak = task.get('text', '')
        
        goal_msg = SpeakText.Goal()
        goal_msg.text = text_to_speak
        
        future = self.speak_client.send_goal_async(goal_msg)
        future.add_done_callback(self.speech_completed_callback)
    
    def speech_completed_callback(self, future):
        """Handle speech completion"""
        try:
            result = future.result().result
            self.get_logger().info(f'Speech completed: {result.success}')
        except Exception as e:
            self.get_logger().error(f'Speech execution failed: {e}')
        
        self.current_task = None
    
    def manipulation_completed_callback(self, future):
        """Handle manipulation completion"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(self.manipulation_result_callback)
        except Exception as e:
            self.get_logger().error(f'Manipulation goal failed: {e}')
            self.current_task = None
    
    def manipulation_result_callback(self, future):
        """Handle manipulation result"""
        try:
            result = future.result().result
            status = future.result().status
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Manipulation completed successfully')
            else:
                self.get_logger().warn(f'Manipulation failed with status: {status}')
        except Exception as e:
            self.get_logger().error(f'Manipulation result error: {e}')
        
        self.current_task = None
    
    def check_joint_limits(self):
        """Check if any joints are near their limits"""
        limit_threshold = 0.1  # radians from joint limit
        
        for joint_name, position in self.robot_state['joint_positions'].items():
            # In practice, you'd have joint limits for each joint
            # For demonstration, we'll use generic limits
            if abs(position) > (3.14159 - limit_threshold):  # Near joint limit
                self.get_logger().warn(f'Joint {joint_name} near limit: {position}')
                # Trigger safety response
                self.emergency_stop()
    
    def check_for_collisions(self):
        """Check for potential collisions using sensor data"""
        # This would integrate with costmap and local planner
        # For demonstration, we'll use simulated data
        pass
    
    def emergency_stop(self):
        """Emergency stop the robot"""
        self.robot_state['emergency_stop'] = True
        self.get_logger().error('EMERGENCY STOP ACTIVATED')
        
        # Send zero velocity commands
        zero_twist = Twist()
        self.cmd_vel_pub.publish(zero_twist)
        
        # Stop all actions
        self.current_task = None
        self.task_queue.clear()
    
    def monitor_system_health(self):
        """Monitor overall system health"""
        # Check various subsystems
        battery_status = self.check_battery_level()
        sensor_status = self.check_sensor_health()
        communication_status = self.check_communication_health()
        
        # Update system health based on checks
        if not battery_status or not sensor_status or not communication_status:
            self.robot_state['system_health'] = 'degraded'
        else:
            self.robot_state['system_health'] = 'nominal'
    
    def check_battery_level(self) -> bool:
        """Check if battery level is acceptable"""
        # In a real system, this would interface with battery monitoring
        # For simulation, return True
        return True
    
    def check_sensor_health(self) -> bool:
        """Check if sensors are operating normally"""
        # In a real system, this would check sensor data quality
        # For simulation, return True
        return True
    
    def check_communication_health(self) -> bool:
        """Check if communication systems are healthy"""
        # In a real system, this would check network connectivity
        # For simulation, return True
        return True
    
    def publish_status(self):
        """Publish current robot status"""
        status_msg = HumanoidStatus()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.header.frame_id = 'base_link'
        
        # Fill status message with current state
        status_msg.position.x = float(self.robot_state['position'][0])
        status_msg.position.y = float(self.robot_state['position'][1])
        status_msg.position.z = float(self.robot_state['position'][2])
        
        status_msg.orientation.x = float(self.robot_state['orientation'][0])
        status_msg.orientation.y = float(self.robot_state['orientation'][1])
        status_msg.orientation.z = float(self.robot_state['orientation'][2])
        status_msg.orientation.w = float(self.robot_state['orientation'][3])
        
        status_msg.battery_level = float(self.robot_state['battery_level'])
        status_msg.system_health = self.robot_state['system_health']
        status_msg.emergency_stop = self.robot_state['emergency_stop']
        status_msg.active_task = str(self.current_task) if self.current_task else 'none'
        
        self.status_pub.publish(status_msg)
    
    def quaternion_to_euler(self, q: np.array) -> tuple:
        """Convert quaternion to euler angles (roll, pitch, yaw)"""
        import math
        
        # Normalize quaternion
        q_norm = q / np.linalg.norm(q)
        w, x, y, z = q_norm
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def process_command_queue(self):
        """Process any pending high-level commands"""
        # This would interface with the VLA and conversational systems
        # to process user commands received through various interfaces
        pass


def main(args=None):
    rclpy.init(args=args)
    
    controller = AutonomousHumanoidController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down autonomous humanoid controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Task Planning and Execution System

The task planning system coordinates complex behaviors across multiple subsystems:

```python
"""
Task planning and execution system for autonomous humanoid robot.
This system takes high-level commands and breaks them down into
executable tasks across navigation, manipulation, and communication subsystems.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from humanoid_interfaces.action import PerformTask
from humanoid_interfaces.msg import HumanoidAction, ObjectDetection
from humanoid_interfaces.srv import GetKnownObjects, GetNavigationMap

from typing import Dict, List, Any, Optional
import json
import time
import threading
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskStep:
    """A single step in a task plan"""
    step_id: str
    action_type: str  # 'navigation', 'manipulation', 'speech', 'perception'
    parameters: Dict[str, Any]
    description: str
    required_subsystems: List[str]
    priority: int = 1  # Lower number = higher priority


class TaskPlanner(Node):
    def __init__(self):
        super().__init__('task_planner')
        
        # Action server for task execution
        self._action_server = ActionServer(
            self,
            PerformTask,
            'perform_task',
            execute_callback=self.execute_task,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        # Service clients for perception and navigation
        self.get_known_objects_cli = self.create_client(
            GetKnownObjects, 'get_known_objects')
        self.get_navigation_map_cli = self.create_client(
            GetNavigationMap, 'get_navigation_map')
        
        # Internal state
        self.current_task = None
        self.task_queue = []
        self.known_objects = {}
        self.navigation_map = None
        
        # Planning database
        self.skill_library = self._initialize_skill_library()
        
        self.get_logger().info('Task planner initialized')
    
    def _initialize_skill_library(self) -> Dict[str, List[TaskStep]]:
        """Initialize the library of known skills and their step sequences"""
        return {
            'fetch_object': [
                TaskStep(
                    step_id='scan_environment',
                    action_type='perception',
                    parameters={'scan_area': 'current_room'},
                    description='Scan environment for object',
                    required_subsystems=['vision', 'object_detection']
                ),
                TaskStep(
                    step_id='navigate_to_object',
                    action_type='navigation',
                    parameters={'target_type': 'object', 'target_name': '{object_name}'},
                    description='Navigate to object location',
                    required_subsystems=['navigation', 'localization']
                ),
                TaskStep(
                    step_id='grasp_object',
                    action_type='manipulation',
                    parameters={'object': '{object_name}', 'grasp_type': 'top_grasp'},
                    description='Grasp the object',
                    required_subsystems=['manipulation', 'tactile_sensing']
                ),
                TaskStep(
                    step_id='bring_to_user',
                    action_type='navigation',
                    parameters={'target_type': 'location', 'target_name': 'user'},
                    description='Return to user with object',
                    required_subsystems=['navigation', 'localization']
                ),
                TaskStep(
                    step_id='deliver_object',
                    action_type='manipulation',
                    parameters={'action': 'release'},
                    description='Release object for user',
                    required_subsystems=['manipulation', 'tactile_sensing']
                )
            ],
            'navigate_to_location': [
                TaskStep(
                    step_id='get_location_coordinates',
                    action_type='perception',
                    parameters={'location_name': '{location_name}'},
                    description='Get coordinates for named location',
                    required_subsystems=['spatial_memory', 'localization']
                ),
                TaskStep(
                    step_id='plan_path',
                    action_type='navigation',
                    parameters={'start': 'current', 'goal': '{coordinates}'},
                    description='Plan path to goal location',
                    required_subsystems=['path_planning']
                ),
                TaskStep(
                    step_id='execute_navigation',
                    action_type='navigation',
                    parameters={'path': '{computed_path}'},
                    description='Execute planned navigation',
                    required_subsystems=['motion_control', 'obstacle_avoidance']
                )
            ],
            'answer_question': [
                TaskStep(
                    step_id='analyze_question',
                    action_type='language',
                    parameters={'question': '{question_text}'},
                    description='Analyze the question semantics',
                    required_subsystems=['natural_language_understanding']
                ),
                TaskStep(
                    step_id='search_knowledge_base',
                    action_type='reasoning',
                    parameters={'query': '{analyzed_question}'},
                    description='Search for relevant information',
                    required_subsystems=['knowledge_base', 'reasoning']
                ),
                TaskStep(
                    step_id='generate_response',
                    action_type='language',
                    parameters={'information': '{search_results}'},
                    description='Generate natural language response',
                    required_subsystems=['natural_language_generation']
                ),
                TaskStep(
                    step_id='speak_response',
                    action_type='speech',
                    parameters={'text': '{generated_response}'},
                    description='Speak the response to user',
                    required_subsystems=['text_to_speech']
                )
            ]
        }
    
    def goal_callback(self, goal_request):
        """Accept or reject a goal request"""
        self.get_logger().info(f'Received task request: {goal_request.task_name}')
        
        # Check if we know how to perform this task
        if goal_request.task_name not in self.skill_library:
            self.get_logger().error(f'Unknown task: {goal_request.task_name}')
            return GoalResponse.REJECT
        
        # Check if robot is available
        if self.current_task is not None:
            # Could queue the task instead of rejecting
            self.get_logger().info('Robot busy, queuing task')
            self.task_queue.append(goal_request)
            return GoalResponse.ACCEPT
        
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT
    
    async def execute_task(self, goal_handle):
        """Execute a high-level task"""
        self.get_logger().info(f'Executing task: {goal_handle.request.task_name}')
        
        # Set current task
        self.current_task = goal_handle.request.task_name
        
        feedback_msg = PerformTask.Feedback()
        result_msg = PerformTask.Result()
        
        try:
            # Get the task plan
            task_plan = self._get_task_plan(goal_handle.request)
            if not task_plan:
                result_msg.success = False
                result_msg.message = f'Could not generate plan for task: {goal_handle.request.task_name}'
                goal_handle.succeed()
                return result_msg
            
            total_steps = len(task_plan)
            completed_steps = 0
            
            # Execute each step in the plan
            for step in task_plan:
                if goal_handle.is_cancel_requested:
                    result_msg.success = False
                    result_msg.message = 'Task cancelled'
                    goal_handle.canceled()
                    self.current_task = None
                    return result_msg
                
                # Update feedback
                feedback_msg.current_step = step.description
                feedback_msg.progress = float(completed_steps) / total_steps
                goal_handle.publish_feedback(feedback_msg)
                
                # Execute the step
                step_success = await self._execute_task_step(step, goal_handle.request.parameters)
                
                if not step_success:
                    self.get_logger().error(f'Step failed: {step.description}')
                    result_msg.success = False
                    result_msg.message = f'Step failed: {step.description}'
                    goal_handle.succeed()
                    self.current_task = None
                    return result_msg
                
                completed_steps += 1
                self.get_logger().info(f'Completed step: {step.description}')
            
            # Task completed successfully
            result_msg.success = True
            result_msg.message = f'Task {goal_handle.request.task_name} completed successfully'
            goal_handle.succeed()
            
        except Exception as e:
            self.get_logger().error(f'Task execution failed: {e}')
            result_msg.success = False
            result_msg.message = f'Task execution failed: {str(e)}'
            goal_handle.succeed()
        
        self.current_task = None
        return result_msg
    
    def _get_task_plan(self, request) -> Optional[List[TaskStep]]:
        """Generate a task plan from the request"""
        task_name = request.task_name
        parameters = request.parameters
        
        if task_name not in self.skill_library:
            return None
        
        # Get the base plan
        base_plan = self.skill_library[task_name]
        
        # Instantiate parameters
        instantiated_plan = []
        for step in base_plan:
            instantiated_params = self._instantiate_parameters(step.parameters, parameters)
            instantiated_step = TaskStep(
                step_id=step.step_id,
                action_type=step.action_type,
                parameters=instantiated_params,
                description=self._instantiate_parameters(step.description, parameters),
                required_subsystems=step.required_subsystems,
                priority=step.priority
            )
            instantiated_plan.append(instantiated_step)
        
        return instantiated_plan
    
    def _instantiate_parameters(self, template: Any, parameters: Dict[str, Any]) -> Any:
        """Replace template placeholders with actual values"""
        if isinstance(template, str):
            # Replace placeholders like {object_name} with actual values
            import re
            def replace_match(match):
                key = match.group(1)
                return str(parameters.get(key, match.group(0)))
            
            return re.sub(r'\{([^}]+)\}', replace_match, template)
        elif isinstance(template, dict):
            # Recursively process dictionary values
            result = {}
            for k, v in template.items():
                result[k] = self._instantiate_parameters(v, parameters)
            return result
        elif isinstance(template, list):
            # Recursively process list items
            return [self._instantiate_parameters(item, parameters) for item in template]
        else:
            return template
    
    async def _execute_task_step(self, step: TaskStep, task_parameters: Dict[str, Any]) -> bool:
        """Execute a single task step"""
        self.get_logger().info(f'Executing step: {step.description}')
        
        # Check if required subsystems are available
        if not self._check_subsystem_availability(step.required_subsystems):
            self.get_logger().error(f'Required subsystems not available: {step.required_subsystems}')
            return False
        
        # Execute based on action type
        if step.action_type == 'navigation':
            return await self._execute_navigation_step(step, task_parameters)
        elif step.action_type == 'manipulation':
            return await self._execute_manipulation_step(step, task_parameters)
        elif step.action_type == 'speech':
            return await self._execute_speech_step(step, task_parameters)
        elif step.action_type == 'perception':
            return await self._execute_perception_step(step, task_parameters)
        elif step.action_type == 'language':
            return await self._execute_language_step(step, task_parameters)
        elif step.action_type == 'reasoning':
            return await self._execute_reasoning_step(step, task_parameters)
        else:
            self.get_logger().error(f'Unknown action type: {step.action_type}')
            return False
    
    def _check_subsystem_availability(self, subsystems: List[str]) -> bool:
        """Check if required subsystems are available"""
        # In a real implementation, this would check actual subsystem status
        # For now, assume all subsystems are available
        return True
    
    async def _execute_navigation_step(self, step: TaskStep, task_parameters: Dict[str, Any]) -> bool:
        """Execute a navigation step"""
        # This would interface with the navigation subsystem
        # For demonstration, we'll simulate the operation
        target_type = step.parameters.get('target_type', 'coordinates')
        target_name = step.parameters.get('target_name', '')
        target_coordinates = step.parameters.get('coordinates', [0, 0, 0])
        
        if target_type == 'object' and target_name:
            # Find object in environment
            object_pose = await self._find_object_pose(target_name)
            if object_pose is None:
                self.get_logger().error(f'Could not find object: {target_name}')
                return False
            target_coordinates = [object_pose.x, object_pose.y, object_pose.z]
        
        self.get_logger().info(f'Navigating to {target_type} {target_name or target_coordinates}')
        
        # Simulate navigation
        import asyncio
        await asyncio.sleep(2.0)  # Simulate navigation time
        
        return True
    
    async def _find_object_pose(self, object_name: str) -> Optional[Any]:
        """Find the pose of an object in the environment"""
        # In a real system, this would query the object detection system
        # For simulation, return a fixed pose
        return type('Pose', (), {'x': 1.0, 'y': 0.5, 'z': 0.0})()
    
    async def _execute_manipulation_step(self, step: TaskStep, task_parameters: Dict[str, Any]) -> bool:
        """Execute a manipulation step"""
        object_name = step.parameters.get('object', 'object')
        action = step.parameters.get('action', 'grasp')
        grasp_type = step.parameters.get('grasp_type', 'top_grasp')
        
        self.get_logger().info(f'Performing {action} on {object_name} using {grasp_type}')
        
        # Simulate manipulation
        import asyncio
        await asyncio.sleep(3.0)  # Simulate manipulation time
        
        return True
    
    async def _execute_speech_step(self, step: TaskStep, task_parameters: Dict[str, Any]) -> bool:
        """Execute a speech step"""
        text = step.parameters.get('text', 'Hello')
        
        self.get_logger().info(f'Speaking: {text}')
        
        # In a real system, this would interface with TTS
        # For simulation, just wait
        import asyncio
        await asyncio.sleep(len(text) / 10)  # Simulate speaking time
        
        return True
    
    async def _execute_perception_step(self, step: TaskStep, task_parameters: Dict[str, Any]) -> bool:
        """Execute a perception step"""
        scan_area = step.parameters.get('scan_area', 'current')
        
        self.get_logger().info(f'Scanning area: {scan_area}')
        
        # Simulate environment scanning
        import asyncio
        await asyncio.sleep(1.0)  # Simulate scanning time
        
        # Update known objects (in a real system, this would come from object detection)
        self.known_objects = {
            'red_cup': {'position': [1.0, 0.5, 0.8], 'confidence': 0.9},
            'blue_box': {'position': [0.8, -0.3, 0.6], 'confidence': 0.85}
        }
        
        return True
    
    async def _execute_language_step(self, step: TaskStep, task_parameters: Dict[str, Any]) -> bool:
        """Execute a language processing step"""
        question = step.parameters.get('question', '')
        
        self.get_logger().info(f'Analyzing question: {question}')
        
        # Simulate language processing
        import asyncio
        await asyncio.sleep(0.5)
        
        return True
    
    async def _execute_reasoning_step(self, step: TaskStep, task_parameters: Dict[str, Any]) -> bool:
        """Execute a reasoning step"""
        query = step.parameters.get('query', '')
        
        self.get_logger().info(f'Searching knowledge base for: {query}')
        
        # Simulate knowledge base search
        import asyncio
        await asyncio.sleep(0.3)
        
        return True


def main(args=None):
    rclpy.init(args=args)
    
    task_planner = TaskPlanner()
    
    # Use multi-threaded executor to handle action callbacks properly
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(task_planner)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        task_planner.get_logger().info('Shutting down task planner')
    finally:
        task_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Integration and Testing

### System Integration Framework

```python
"""
Integration and testing framework for the autonomous humanoid system.
This includes simulation testing, hardware-in-the-loop testing,
and performance validation.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image, JointState
from humanoid_interfaces.msg import HumanoidStatus, ObjectDetection
from builtin_interfaces.msg import Time

import time
import threading
from typing import Dict, List, Callable, Any
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json

@dataclass
class TestResult:
    """Structure for storing test results"""
    test_name: str
    passed: bool
    duration: float
    details: Dict[str, Any]
    metrics: Dict[str, float]  # Performance metrics

class IntegrationTestFramework(Node):
    def __init__(self):
        super().__init__('integration_test_framework')
        
        # Publishers for test scenarios
        self.command_pub = self.create_publisher(
            String, '/test_commands', QoSProfile(depth=10))
        self.goal_pub = self.create_publisher(
            Pose, '/test_goals', QoSProfile(depth=10))
        
        # Subscribers for monitoring
        self.status_sub = self.create_subscription(
            HumanoidStatus, '/humanoid/status', self.status_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/head_camera/rgb/image_raw', self.image_callback, 10)
        
        # Internal state
        self.status_history = []
        self.joint_history = []
        self.test_results = []
        self.current_test = None
        self.test_start_time = None
        self.test_active = False
        
        # Test scenarios
        self.test_scenarios = {
            'basic_mobility': self.test_basic_mobility,
            'object_interaction': self.test_object_interaction,
            'conversation': self.test_conversation,
            'navigation': self.test_navigation,
            'multi_task': self.test_multi_task_scenario
        }
        
        self.get_logger().info('Integration test framework initialized')
    
    def status_callback(self, msg: HumanoidStatus):
        """Store status for analysis"""
        self.status_history.append((self.get_clock().now(), msg))
        
        if self.test_active:
            # Log for current test
            self.get_logger().debug(f'Robot position: ({msg.position.x:.2f}, {msg.position.y:.2f})')
    
    def joint_callback(self, msg: JointState):
        """Store joint state for analysis"""
        self.joint_history.append((self.get_clock().now(), msg))
    
    def image_callback(self, msg: Image):
        """Process camera images during tests"""
        # In a real test, this might analyze image data
        pass
    
    def run_test_suite(self) -> List[TestResult]:
        """Run all integration tests"""
        results = []
        
        self.get_logger().info('Starting integration test suite')
        
        for test_name, test_func in self.test_scenarios.items():
            self.get_logger().info(f'Running test: {test_name}')
            
            # Run the test and collect result
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                result.duration = duration
                results.append(result)
                status = "PASSED" if result.passed else "FAILED"
                self.get_logger().info(f'Test {test_name}: {status} ({duration:.2f}s)')
            else:
                # Test didn't return a proper result
                failed_result = TestResult(
                    test_name=test_name,
                    passed=False,
                    duration=duration,
                    details={'error': 'Test function returned None'},
                    metrics={}
                )
                results.append(failed_result)
                self.get_logger().error(f'Test {test_name} failed to return proper result')
        
        # Generate test report
        self.generate_test_report(results)
        
        return results
    
    def test_basic_mobility(self) -> TestResult:
        """Test basic mobility functions"""
        test_name = 'basic_mobility'
        self.get_logger().info(f'Running {test_name}')
        
        start_time = time.time()
        try:
            # Clear history
            self.status_history = []
            self.joint_history = []
            
            # Send mobility commands
            self._send_command('move_forward', {'distance': 1.0, 'speed': 0.5})
            time.sleep(3)  # Wait for movement
            
            self._send_command('turn', {'angle': 90.0, 'speed': 0.3})
            time.sleep(2)  # Wait for turn
            
            self._send_command('move_backward', {'distance': 0.5, 'speed': 0.3})
            time.sleep(2)  # Wait for movement
            
            # Analyze results
            success = self._analyze_mobility_results()
            
            # Calculate metrics
            metrics = self._calculate_mobility_metrics()
            
            return TestResult(
                test_name=test_name,
                passed=success,
                duration=time.time() - start_time,
                details={'status_history_length': len(self.status_history)},
                metrics=metrics
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                duration=time.time() - start_time,
                details={'error': str(e)},
                metrics={}
            )
    
    def test_object_interaction(self) -> TestResult:
        """Test object detection and manipulation"""
        test_name = 'object_interaction'
        self.get_logger().info(f'Running {test_name}')
        
        start_time = time.time()
        try:
            # Clear history
            self.status_history = []
            
            # Simulate object detection scenario
            self._send_command('look_for', {'object': 'red_cup'})
            time.sleep(2)
            
            # Simulate approach and grasp
            self._send_command('approach_object', {'object': 'red_cup'})
            time.sleep(3)
            
            self._send_command('grasp_object', {'object': 'red_cup'})
            time.sleep(2)
            
            # Analyze results
            success = self._analyze_object_interaction_results()
            
            # Calculate metrics
            metrics = self._calculate_object_metrics()
            
            return TestResult(
                test_name=test_name,
                passed=success,
                duration=time.time() - start_time,
                details={'actions_performed': 3},
                metrics=metrics
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                duration=time.time() - start_time,
                details={'error': str(e)},
                metrics={}
            )
    
    def test_conversation(self) -> TestResult:
        """Test conversational capabilities"""
        test_name = 'conversation'
        self.get_logger().info(f'Running {test_name}')
        
        start_time = time.time()
        try:
            # This would interface with the conversational system
            # For simulation, we'll track speech events
            
            # Simulate receiving and responding to commands
            conversation_steps = [
                'hello',
                'what can you do',
                'please pick up the red cup',
                'thank you',
                'goodbye'
            ]
            
            response_times = []
            
            for step in conversation_steps:
                step_start = time.time()
                
                # In a real system, this would process the step through
                # the conversational pipeline
                time.sleep(np.random.uniform(0.5, 1.5))  # Simulate processing time
                
                response_times.append(time.time() - step_start)
            
            # Analyze results
            avg_response_time = np.mean(response_times)
            success = avg_response_time < 2.0  # Response should be under 2 seconds
            
            metrics = {
                'avg_response_time': avg_response_time,
                'conversation_steps': len(conversation_steps)
            }
            
            return TestResult(
                test_name=test_name,
                passed=success,
                duration=time.time() - start_time,
                details={'response_times': response_times},
                metrics=metrics
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                duration=time.time() - start_time,
                details={'error': str(e)},
                metrics={}
            )
    
    def test_navigation(self) -> TestResult:
        """Test navigation system"""
        test_name = 'navigation'
        self.get_logger().info(f'Running {test_name}')
        
        start_time = time.time()
        try:
            # Clear history
            self.status_history = []
            
            # Define navigation waypoints
            waypoints = [
                {'x': 1.0, 'y': 0.0},
                {'x': 1.0, 'y': 1.0},
                {'x': 0.0, 'y': 1.0},
                {'x': 0.0, 'y': 0.0}  # Return to start
            ]
            
            navigation_success = True
            path_times = []
            
            for i, waypoint in enumerate(waypoints):
                wp_start = time.time()
                
                # Send navigation command
                pose_msg = Pose()
                pose_msg.position.x = waypoint['x']
                pose_msg.position.y = waypoint['y']
                pose_msg.position.z = 0.0
                pose_msg.orientation.w = 1.0  # No rotation
                
                self.goal_pub.publish(pose_msg)
                time.sleep(5)  # Wait for navigation
                
                path_times.append(time.time() - wp_start)
            
            # Analyze navigation results
            avg_time_per_waypoint = np.mean(path_times)
            success = avg_time_per_waypoint < 6.0  # Should navigate each leg in under 6 seconds
            
            metrics = {
                'avg_waypoint_time': avg_time_per_waypoint,
                'total_waypoints': len(waypoints),
                'path_efficiency': 0.85  # Placeholder value
            }
            
            return TestResult(
                test_name=test_name,
                passed=success,
                duration=time.time() - start_time,
                details={'waypoints_completed': len(waypoints)},
                metrics=metrics
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                duration=time.time() - start_time,
                details={'error': str(e)},
                metrics={}
            )
    
    def test_multi_task_scenario(self) -> TestResult:
        """Test coordinated multi-task scenario"""
        test_name = 'multi_task_scenario'
        self.get_logger().info(f'Running {test_name}')
        
        start_time = time.time()
        try:
            # Scenario: Human asks robot to navigate to kitchen, find a cup, and bring it back
            self.status_history = []
            
            # Step 1: Navigate to kitchen
            self._send_command('navigate', {'location': 'kitchen'})
            time.sleep(4)
            
            # Step 2: Look for cup
            self._send_command('find_object', {'object': 'cup'})
            time.sleep(3)
            
            # Step 3: Approach and grasp cup
            self._send_command('grasp_object', {'object': 'cup'})
            time.sleep(3)
            
            # Step 4: Return to user
            self._send_command('navigate', {'location': 'user'})
            time.sleep(4)
            
            # Step 5: Deliver object
            self._send_command('release_object', {})
            time.sleep(1)
            
            # Analyze multi-step results
            success = self._analyze_multi_task_results()
            
            metrics = {
                'total_completion_time': time.time() - start_time,
                'task_steps_completed': 5
            }
            
            return TestResult(
                test_name=test_name,
                passed=success,
                duration=time.time() - start_time,
                details={'task_sequence': ['navigate', 'find', 'grasp', 'return', 'deliver']},
                metrics=metrics
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                duration=time.time() - start_time,
                details={'error': str(e)},
                metrics={}
            )
    
    def _send_command(self, command: str, params: Dict[str, Any]):
        """Send command to the robot for testing"""
        cmd_msg = String()
        cmd_msg.data = json.dumps({'command': command, 'params': params})
        self.command_pub.publish(cmd_msg)
    
    def _analyze_mobility_results(self) -> bool:
        """Analyze mobility test results"""
        # Check if robot moved as expected
        if len(self.status_history) < 10:
            return False  # Not enough data
        
        # Get starting and ending positions
        start_pos = self.status_history[0][1].position
        end_pos = self.status_history[-1][1].position
        
        # Calculate distance traveled
        distance = np.sqrt(
            (end_pos.x - start_pos.x)**2 + 
            (end_pos.y - start_pos.y)**2
        )
        
        # Check if robot moved at least 1 meter (for the 1m forward test)
        return distance >= 0.8  # Allow some tolerance
    
    def _analyze_object_interaction_results(self) -> bool:
        """Analyze object interaction test results"""
        # For simulation purposes, assume success
        return True
    
    def _analyze_multi_task_results(self) -> bool:
        """Analyze multi-task scenario results"""
        # For simulation purposes, assume success
        return True
    
    def _calculate_mobility_metrics(self) -> Dict[str, float]:
        """Calculate mobility-specific metrics"""
        return {
            'average_velocity': 0.3,  # Placeholder
            'path_accuracy': 0.95,    # Placeholder
            'energy_efficiency': 0.8  # Placeholder
        }
    
    def _calculate_object_metrics(self) -> Dict[str, float]:
        """Calculate object interaction metrics"""
        return {
            'detection_rate': 0.9,     # Placeholder
            'grasp_success_rate': 0.85, # Placeholder
            'manipulation_accuracy': 0.92 # Placeholder
        }
    
    def generate_test_report(self, results: List[TestResult]):
        """Generate comprehensive test report"""
        self.get_logger().info('\n' + '='*50)
        self.get_logger().info('INTEGRATION TEST RESULTS')
        self.get_logger().info('='*50)
        
        passed_count = sum(1 for result in results if result.passed)
        total_count = len(results)
        
        self.get_logger().info(f'Tests passed: {passed_count}/{total_count}')
        self.get_logger().info(f'Success rate: {passed_count/total_count*100:.1f}%')
        
        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            self.get_logger().info(f'{status} {result.test_name} ({result.duration:.2f}s)')
            
            if result.passed and result.metrics:
                for metric, value in result.metrics.items():
                    self.get_logger().info(f'  {metric}: {value:.3f}')
            
            if not result.passed and 'error' in result.details:
                self.get_logger().info(f'  Error: {result.details["error"]}')
        
        self.get_logger().info('='*50)
        
        # Save results to file
        self._save_test_results(results)
    
    def _save_test_results(self, results: List[TestResult]):
        """Save test results to file"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integration_test_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'test_name': result.test_name,
                'passed': result.passed,
                'duration': result.duration,
                'details': result.details,
                'metrics': result.metrics
            })
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_tests': len(results),
                'passed_tests': sum(1 for r in results if r.passed),
                'results': serializable_results
            }, f, indent=2)
        
        self.get_logger().info(f'Test results saved to {filename}')


def main(args=None):
    rclpy.init(args=args)
    
    test_framework = IntegrationTestFramework()
    
    try:
        results = test_framework.run_test_suite()
        
        # Wait a moment for any pending messages
        time.sleep(1)
        
    except KeyboardInterrupt:
        test_framework.get_logger().info('Test suite interrupted by user')
    finally:
        test_framework.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Autonomous Capstone Scenario: Voice Command to Object Manipulation

Now we'll implement the complete capstone scenario as specified:

```python
"""
Capstone autonomous humanoid scenario: Voice command to object manipulation.
This demonstrates the complete integration of all system components.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image
from humanoid_interfaces.msg import HumanoidStatus, ObjectDetection
from humanoid_interfaces.action import PerformTask, NavigateToPose, ManipulateObject

import time
import threading
from typing import Dict, List, Optional
import numpy as np


class CapstoneScenario(Node):
    """
    Implements the capstone scenario: A humanoid robot that receives 
    a voice command, plans actions, navigates obstacles, identifies 
    an object, and manipulates it in simulation.
    """
    def __init__(self):
        super().__init__('capstone_scenario')
        
        # Action clients for complex behaviors
        self.nav_client = ActionClient(
            self, NavigateToPose, '/humanoid/navigate_to_pose')
        self.manip_client = ActionClient(
            self, ManipulateObject, '/humanoid/manipulate_object')
        self.task_client = ActionClient(
            self, PerformTask, '/task_performer/perform_task')
        
        # Publishers for commands
        self.voice_cmd_pub = self.create_publisher(
            String, '/speech_recognition/text', QoSProfile(depth=10))
        self.status_pub = self.create_publisher(
            HumanoidStatus, '/humanoid/status', QoSProfile(depth=10))
        
        # Subscribers for monitoring
        self.status_sub = self.create_subscription(
            HumanoidStatus, '/humanoid/status', self.status_callback, 10)
        self.object_sub = self.create_subscription(
            ObjectDetection, '/object_detection/objects', self.object_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/head_camera/rgb/image_raw', self.image_callback, 10)
        
        # Internal state
        self.robot_position = Point()
        self.detected_objects = []
        self.current_state = 'idle'  # idle, listening, processing, executing, completed
        self.scenario_active = False
        
        # Scenario parameters
        self.voice_command = "Please pick up the red cup on the table and bring it to me"
        self.target_object = "red cup"
        self.user_location = Point(x=0.0, y=0.0, z=0.0)
        
        self.get_logger().info('Capstone scenario initialized')
    
    def status_callback(self, msg: HumanoidStatus):
        """Update robot position from status messages"""
        self.robot_position.x = msg.position.x
        self.robot_position.y = msg.position.y
        self.robot_position.z = msg.position.z
    
    def object_callback(self, msg: ObjectDetection):
        """Handle detected objects"""
        self.detected_objects.append({
            'name': msg.object_name,
            'position': msg.position,
            'confidence': msg.confidence
        })
    
    def image_callback(self, msg: Image):
        """Process camera images during scenario"""
        # In a real implementation, this would feed into object detection
        pass
    
    def execute_capstone_scenario(self):
        """Execute the complete capstone scenario"""
        self.get_logger().info(f'Starting capstone scenario: {self.voice_command}')
        self.scenario_active = True
        self.current_state = 'listening'
        
        try:
            # Step 1: Simulate hearing the voice command
            self.get_logger().info('Step 1: Receiving voice command')
            self._simulate_voice_input(self.voice_command)
            
            # Wait briefly for command processing
            time.sleep(1.0)
            
            # Step 2: Parse command and plan actions
            self.get_logger().info('Step 2: Processing command and planning actions')
            self.current_state = 'processing'
            actions = self._parse_and_plan_actions(self.voice_command)
            
            # Step 3: Execute navigation to object
            self.get_logger().info('Step 3: Navigating to object location')
            self.current_state = 'executing'
            
            # Find the target object
            object_position = self._locate_target_object()
            if object_position:
                success = self._navigate_to_position(object_position)
                if not success:
                    self.get_logger().error('Navigation to object failed')
                    return False
            else:
                self.get_logger().error('Could not locate target object')
                return False
            
            # Step 4: Manipulate the object
            self.get_logger().info('Step 4: Manipulating object')
            success = self._manipulate_object()
            if not success:
                self.get_logger().error('Object manipulation failed')
                return False
            
            # Step 5: Return to user
            self.get_logger().info('Step 5: Returning to user')
            success = self._navigate_to_position(self.user_location)
            if not success:
                self.get_logger().error('Navigation to user failed')
                return False
            
            # Step 6: Deliver object
            self.get_logger().info('Step 6: Delivering object')
            success = self._deliver_object()
            if not success:
                self.get_logger().error('Object delivery failed')
                return False
            
            self.current_state = 'completed'
            self.get_logger().info('Capstone scenario completed successfully!')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Capstone scenario failed: {e}')
            self.current_state = 'error'
            return False
        finally:
            self.scenario_active = False
    
    def _simulate_voice_input(self, command: str):
        """Simulate voice command input"""
        self.get_logger().info(f'Simulated voice input: {command}')
        
        # In a real system, this would interface with speech recognition
        # For simulation, we'll just log the command
        cmd_msg = String()
        cmd_msg.data = command
        self.voice_cmd_pub.publish(cmd_msg)
    
    def _parse_and_plan_actions(self, command: str) -> List[Dict]:
        """Parse the command and create action plan"""
        self.get_logger().info(f'Parsing command: {command}')
        
        # Simple command parsing for demonstration
        # A real system would use NLP and VLA models
        actions = []
        
        if 'pick up' in command.lower() or 'grasp' in command.lower():
            object_name = self._extract_object_name(command)
            actions.append({
                'type': 'navigate_to_object',
                'object': object_name
            })
            actions.append({
                'type': 'grasp_object',
                'object': object_name
            })
        
        if 'bring it to me' in command.lower() or 'deliver' in command.lower():
            actions.append({
                'type': 'navigate_to_user',
                'target': 'user'
            })
            actions.append({
                'type': 'deliver_object',
                'action': 'release'
            })
        
        self.get_logger().info(f'Generated action plan: {actions}')
        return actions
    
    def _extract_object_name(self, command: str) -> str:
        """Extract object name from command"""
        # Simple extraction for demonstration
        command_lower = command.lower()
        
        # Look for color + object patterns
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
        objects = ['cup', 'box', 'ball', 'bottle', 'book']
        
        for color in colors:
            if color in command_lower:
                for obj in objects:
                    if obj in command_lower:
                        return f"{color} {obj}"
        
        # Default to red cup if pattern not found
        return 'red cup'
    
    def _locate_target_object(self) -> Optional[Point]:
        """Locate the target object in the environment"""
        self.get_logger().info(f'Locating target object: {self.target_object}')
        
        # In a real system, this would interface with object detection
        # For simulation, we'll return a fixed position
        # This could also use visual SLAM to locate known objects
        
        # Simulate object detection by creating a "detected" object
        detected_obj = {
            'name': self.target_object,
            'position': Point(x=1.5, y=0.8, z=0.75),  # Table position
            'confidence': 0.9
        }
        
        self.detected_objects.append(detected_obj)
        self.get_logger().info(f'Located {self.target_object} at ({detected_obj["position"].x}, {detected_obj["position"].y}, {detected_obj["position"].z})')
        
        return detected_obj['position']
    
    def _navigate_to_position(self, target: Point) -> bool:
        """Navigate the robot to a target position"""
        self.get_logger().info(f'Navigating to position: ({target.x}, {target.y}, {target.z})')
        
        # Wait for navigation server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation server not available')
            return False
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = target.x
        goal_msg.pose.pose.position.y = target.y
        goal_msg.pose.pose.position.z = target.z
        goal_msg.pose.pose.orientation.w = 1.0  # No rotation
        
        # Send navigation goal
        future = self.nav_client.send_goal_async(goal_msg)
        
        # Wait for result (with timeout)
        import asyncio
        try:
            # Simulate navigation execution
            time.sleep(3.0)  # Simulated navigation time
            
            # For simulation purposes, assume navigation succeeds
            self.get_logger().info(f'Navigation completed to ({target.x}, {target.y})')
            return True
        except Exception as e:
            self.get_logger().error(f'Navigation failed: {e}')
            return False
    
    def _manipulate_object(self) -> bool:
        """Manipulate the target object"""
        self.get_logger().info(f'Manipulating object: {self.target_object}')
        
        # Wait for manipulation server
        if not self.manip_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Manipulation server not available')
            return False
        
        # Create manipulation goal
        goal_msg = ManipulateObject.Goal()
        goal_msg.object_name = self.target_object
        goal_msg.action = 'grasp'
        goal_msg.grasp_type = 'top_grasp'
        
        # Send manipulation goal
        future = self.manip_client.send_goal_async(goal_msg)
        
        # Simulate manipulation
        time.sleep(2.0)  # Simulated manipulation time
        
        self.get_logger().info(f'Successfully grasped {self.target_object}')
        return True
    
    def _deliver_object(self) -> bool:
        """Deliver the object to the user"""
        self.get_logger().info(f'Delivering object to user')
        
        # Wait for manipulation server
        if not self.manip_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Manipulation server not available')
            return False
        
        # Create manipulation goal for release
        goal_msg = ManipulateObject.Goal()
        goal_msg.object_name = self.target_object
        goal_msg.action = 'release'
        goal_msg.grasp_type = 'place'
        
        # Send manipulation goal
        future = self.manip_client.send_goal_async(goal_msg)
        
        # Simulate release
        time.sleep(1.0)  # Simulated release time
        
        self.get_logger().info(f'Successfully delivered {self.target_object} to user')
        return True


def main(args=None):
    rclpy.init(args=args)
    
    capstone = CapstoneScenario()
    
    try:
        # Execute the capstone scenario
        success = capstone.execute_capstone_scenario()
        
        if success:
            capstone.get_logger().info('Capstone scenario completed successfully!')
        else:
            capstone.get_logger().error('Capstone scenario failed!')
        
        # Keep node alive briefly to publish final status
        time.sleep(1)
        
    except KeyboardInterrupt:
        capstone.get_logger().info('Capstone scenario interrupted by user')
    finally:
        capstone.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Performance Optimization and Deployment

### System Optimization Guidelines

```yaml
# Performance optimization guidelines for autonomous humanoid system
optimization_strategies:

  # Computing optimizations
  computing:
    - "Use NVIDIA TensorRT for optimized neural network inference"
    - "Implement model quantization (INT8) for faster execution"
    - "Use multi-threading for parallel processing of independent tasks"
    - "Implement GPU memory management to prevent memory leaks"
    - "Use efficient data structures and minimize memory allocations"
  
  # Communication optimizations  
  communication:
    - "Use appropriate QoS profiles for different message types"
    - "Implement message compression for high-bandwidth sensors"
    - "Use shared memory for inter-process communication when possible"
    - "Optimize network configuration for low-latency communication"
  
  # Algorithm optimizations
  algorithms:
    - "Implement efficient path planning algorithms (ARA*, D* Lite)"
    - "Use approximate methods where exact solutions aren't critical"
    - "Implement caching for frequently accessed data"
    - "Use efficient search structures (KD-trees, octrees) for spatial queries"
  
  # Hardware-specific optimizations
  hardware:
    - "Optimize for target computing platform (Jetson, workstation)"
    - "Use hardware acceleration for image processing (CUDA, Tensor cores)"
    - "Implement power management for battery operation"
    - "Use real-time scheduling for critical control loops"

deployment_considerations:

  # Safety measures
  safety:
    - "Implement emergency stop functionality"
    - "Use safety-rated hardware where required"
    - "Implement multiple safety layers (software and hardware)"
    - "Validate all safety-critical functions"
  
  # Reliability
  reliability:
    - "Implement comprehensive error handling and recovery"
    - "Use watchdog timers for critical processes"
    - "Implement graceful degradation when components fail"
    - "Include self-diagnostic capabilities"
  
  # Maintainability
  maintainability:
    - "Use modular architecture for easy updates"
    - "Implement comprehensive logging and monitoring"
    - "Provide remote diagnostics capabilities"
    - "Include over-the-air update functionality"
  
  # Scalability
  scalability:
    - "Design for future hardware upgrades"
    - "Use standardized interfaces and protocols"
    - "Implement cloud connectivity for remote operation"
    - "Plan for multi-robot coordination"
```

## Conclusion and Next Steps

The autonomous humanoid system represents the integration of all the technologies covered in this book, from ROS 2 fundamentals to Vision-Language-Action systems. This capstone project demonstrates how physical AI systems can operate autonomously in real-world environments, processing natural language commands and executing complex physical tasks.

### Key Integration Points:

1. **Sensory Integration**: Combining multiple sensor modalities (vision, audio, IMU) for comprehensive environment understanding
2. **Cognitive Integration**: Coordinating language processing, planning, and decision-making
3. **Action Integration**: Unifying navigation, manipulation, and locomotion
4. **Communication Integration**: Enabling fluid human-robot interaction

### Performance Considerations:

- Real-time response to voice commands within 2 seconds
- Navigation path planning in less than 0.5 seconds
- Object detection and manipulation within 5 seconds
- System reliability of 99.5% uptime

### Future Enhancements:

- Advanced machine learning for improved adaptation
- Multi-robot coordination capabilities
- Enhanced conversational abilities
- Improved energy efficiency

The autonomous humanoid system created in this capstone project serves as a foundation for future development in physical AI and embodied intelligence, demonstrating the potential for truly autonomous humanoid robots operating in human environments.

![Autonomous Humanoid](/img/autonomous-humanoid-capstone.jpg)
*Image Placeholder: Diagram showing the complete autonomous humanoid system with all subsystems integrated*

---

## Key Takeaways

- The autonomous humanoid system integrates all book concepts into a functional platform
- Modularity and real-time performance are critical for autonomous operation
- Safety and reliability must be built into all system components
- Human-robot interaction is enhanced through multimodal integration
- Comprehensive testing ensures system reliability and safety
- Performance optimization is essential for real-time operation

## Final Thoughts

This capstone project demonstrates that building autonomous humanoid robots requires integration across multiple domains: mechanical engineering, electrical systems, computer vision, natural language processing, control theory, and robotics. Success requires not only technical expertise in each area but also the ability to integrate these components into a cohesive, reliable system that can operate autonomically in real-world environments.

The journey from individual components to a complete autonomous system showcases the complexity and interdisciplinary nature of humanoid robotics, while also highlighting the tremendous potential of physical AI systems to enhance human life.