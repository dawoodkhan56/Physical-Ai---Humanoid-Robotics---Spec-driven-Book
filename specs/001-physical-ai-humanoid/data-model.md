# Data Model: Physical AI & Humanoid Robotics Educational Content

## Core Entities

### Module
- **Description**: A major section of the curriculum focusing on a specific aspect of humanoid robotics
- **Fields**: 
  - id: string (unique identifier, e.g., "module-1-robotic-nervous-system")
  - title: string (e.g., "The Robotic Nervous System")
  - duration: number (estimated weeks to complete)
  - objectives: array of strings (learning objectives)
  - prerequisites: array of strings (required knowledge)
  - contentPaths: array of strings (paths to markdown files)
  - exercises: array of exercise objects
- **Relationships**: Contains multiple Lessons; connects to Resources
- **Validation**: All fields required; duration must be positive number

### Lesson
- **Description**: A subsection within a Module covering a specific topic
- **Fields**:
  - id: string (unique identifier)
  - moduleId: string (reference to parent Module)
  - title: string (e.g., "ROS 2 Architecture")
  - sequence: number (order within the module)
  - duration: number (estimated hours to complete)
  - contentPath: string (path to markdown file)
  - learningObjectives: array of strings
  - keywords: array of strings
- **Relationships**: Belongs to one Module; contains Resources; connects to related Lessons
- **Validation**: All fields required; sequence must be positive integer

### Exercise
- **Description**: Practical activity for learners to apply concepts learned
- **Fields**:
  - id: string (unique identifier)
  - moduleId: string (reference to parent Module)
  - lessonId: string (optional reference to parent Lesson)
  - title: string
  - type: string (e.g., "simulation", "implementation", "analysis")
  - difficulty: string (e.g., "beginner", "intermediate", "advanced")
  - duration: number (estimated hours to complete)
  - instructionsPath: string (path to markdown file with instructions)
  - solutionPath: string (path to markdown file with solution)
  - requiredResources: array of strings
- **Relationships**: Belongs to Module; optionally belongs to Lesson
- **Validation**: All fields required; type must be one of allowed values

### Resource
- **Description**: Supplementary material like images, diagrams, or code samples
- **Fields**:
  - id: string (unique identifier)
  - title: string
  - type: string (e.g., "image", "diagram", "code", "video", "model")
  - path: string (relative path to file)
  - tags: array of strings
  - associatedWith: array of strings (module/lesson IDs)
  - description: string
- **Relationships**: Connected to Modules, Lessons, and Exercises
- **Validation**: All fields required; type must be one of allowed values

### LearningObjective
- **Description**: Specific skill or knowledge point that learners should acquire
- **Fields**:
  - id: string (unique identifier)
  - text: string (the objective statement)
  - moduleIds: array of strings (modules where this objective is taught)
  - lessonIds: array of strings (lessons where this objective is taught)
  - difficulty: string (e.g., "foundational", "intermediate", "advanced")
- **Relationships**: Connected to Modules and Lessons
- **Validation**: All fields required; difficulty must be one of allowed values

### KnowledgeCheck
- **Description**: Assessment to verify learner understanding
- **Fields**:
  - id: string (unique identifier)
  - moduleId: string (reference to parent Module)
  - lessonId: string (optional reference to parent Lesson)
  - title: string
  - questions: array of question objects
  - passingScore: number (percentage required to pass)
  - timeLimit: number (minutes; 0 = no limit)
  - description: string
- **Relationships**: Belongs to Module; optionally belongs to Lesson
- **Validation**: All fields required; passingScore must be 0-100; timeLimit must be non-negative

## State Transitions

### ModuleState
- **Draft** → **Reviewed** → **Published** → **Archived**
- **Conditions**: 
  - Draft → Reviewed: All lessons completed and reviewed by subject matter expert
  - Reviewed → Published: Quality assurance passed and all exercises tested
  - Published → Archived: Module has been superseded by new content

### LessonState
- **Outline** → **ContentDraft** → **Reviewed** → **Published**
- **Conditions**:
  - Outline → ContentDraft: Initial content written
  - ContentDraft → Reviewed: Content reviewed and feedback incorporated
  - Reviewed → Published: Content finalized and linked to module

### LearnerProgress
- **NotStarted** → **InProgress** → **Completed** → **Archived**
- **Conditions**:
  - NotStarted → InProgress: Learner begins first lesson
  - InProgress → Completed: Learner completes all lessons and exercises in module
  - Completed → Archived: Learner moves to next module or course completion

## Relationships

```
[Module] 1 -- * [Lesson] 1 -- * [Exercise]
   |                      |
   |                      * -- * [Resource]
   |
   * -- * [Resource]
   |
   * -- * [KnowledgeCheck]
   |
   * -- * [LearningObjective]

[Lesson] -- * [Resource]
   |
   * -- * [LearningObjective]
   |
   * -- * [KnowledgeCheck]

[Exercise] -- * [Resource]
```

## Validation Rules

1. **Module Duration**: The sum of lesson durations within a module should not exceed the module's total duration by more than 10%
2. **Prerequisites**: A module's prerequisites must be satisfied by successfully completed previous modules
3. **Resource Uniqueness**: Each resource path must be unique across all modules
4. **Knowledge Check Alignment**: Each knowledge check must have at least one question related to each learning objective of the associated lesson/module
5. **Exercise Difficulty**: Exercise difficulty should not exceed the maximum difficulty level of the associated module/lesson