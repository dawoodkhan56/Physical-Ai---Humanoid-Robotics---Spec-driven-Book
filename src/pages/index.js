/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/introduction">
            Read the Book - 15min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="A comprehensive guide to Physical AI and Humanoid Robotics by Dawood Khan">
      <HomepageHeader />
      <main>
        <section className={styles.aboutSection}>
          <div className="container">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <h2>Physical AI & Humanoid Robotics</h2>
                <p style={{fontSize: '1.2em', textAlign: 'center'}}>Designed by Dawood Khan</p>
                <p>
                  Welcome to the comprehensive guide on Physical AI & Humanoid Robotics: Embodied Intelligence in the Physical World. 
                  This book explores the cutting-edge intersection of artificial intelligence and robotics, focusing on systems that 
                  interact directly with the physical environment through sensors and actuators.
                </p>
                <p>
                  Through this book, you will learn about:
                </p>
                <ul>
                  <li>The fundamental principles of Physical AI and embodied intelligence</li>
                  <li>ROS 2 fundamentals and the robotic nervous system</li>
                  <li>Digital twin environments using Gazebo and Unity</li>
                  <li>NVIDIA Isaac platforms for AI-robot brains</li>
                  <li>Vision-Language Action (VLA) systems</li>
                  <li>Conversational robotics and human-robot interaction</li>
                  <li>Complete autonomous humanoid implementation</li>
                </ul>
                <div className={styles.buttons} style={{marginTop: '2rem'}}>
                  <Link className="button button--primary button--lg" to="/docs/introduction">
                    Get Started
                  </Link>
                  <Link className="button button--secondary button--lg" to="/docs/introduction">
                    View Documentation
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}