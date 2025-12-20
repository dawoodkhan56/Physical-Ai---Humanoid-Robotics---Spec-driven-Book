import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI Foundations',
    description: (
      <>
        Learn about the fundamental principles of Physical AI and embodied intelligence,
        where artificial intelligence systems interact directly with the physical world.
      </>
    ),
  },
  {
    title: 'ROS 2 & Robotics',
    description: (
      <>
        Master ROS 2 fundamentals and understand how to build robust robotic systems
        with proper communication and control patterns.
      </>
    ),
  },
  {
    title: 'Simulation & Control',
    description: (
      <>
        Explore advanced simulation environments and control strategies for humanoid robots
        using Gazebo, Unity, and NVIDIA Isaac platforms.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <div 
          style={{
            backgroundColor: '#4D96FF',
            borderRadius: '10px',
            width: '100px',
            height: '100px',
            margin: '0 auto 1rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '3rem',
            color: 'white'
          }}
        >
          🤖
        </div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}