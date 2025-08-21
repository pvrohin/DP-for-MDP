# Dynamic Programming for Markov Decision Processes (MDPs)

This repository contains implementations of three fundamental algorithms for solving Markov Decision Processes (MDPs) using dynamic programming approaches. The project demonstrates value iteration, policy iteration, and generalized policy iteration algorithms applied to a grid-world navigation problem.

## Overview

The project implements three classic dynamic programming algorithms for MDPs:

1. **Value Iteration** (`value_iteration.py`) - Iteratively computes optimal value functions
2. **Policy Iteration** (`policy_iteration.py`) - Alternates between policy evaluation and policy improvement
3. **Generalized Policy Iteration** (`generalized_policy_iteration.py`) - Combines value and policy updates in a single loop

## Problem Domain

The algorithms are applied to a **14×51 grid-world navigation problem** where:

- **States**: Each grid cell represents a state (714 total states)
- **Actions**: 8 possible movements (up, down, left, right, and 4 diagonal directions)
- **Rewards**: 
  - Free space: -1 per step
  - Obstacles: -50 (inaccessible)
  - Goal: +100 (located at position [7, 10])
- **Environment**: Stochastic with 60% probability of intended action, 20% for each adjacent direction
- **Discount Factor**: γ = 0.95

## Algorithm Implementations

### 1. Value Iteration (`value_iteration.py`)

**Algorithm**: Pure value iteration approach
- **Convergence Criterion**: δ = 0.01
- **Process**: Iteratively updates value function using Bellman optimality equation
- **Policy**: Derived from value function after convergence

**Key Features**:
- Direct value function optimization
- Automatic policy extraction
- Guaranteed convergence to optimal policy

### 2. Policy Iteration (`policy_iteration.py`)

**Algorithm**: Classic policy iteration with separate evaluation and improvement phases
- **Convergence Criterion**: δ = 0.1
- **Process**: 
  1. Policy evaluation (value iteration under fixed policy)
  2. Policy improvement (greedy action selection)
  3. Repeat until policy stabilizes

**Key Features**:
- Two-phase approach: evaluation and improvement
- Often faster convergence than pure value iteration
- Clear separation of concerns

### 3. Generalized Policy Iteration (`generalized_policy_iteration.py`)

**Algorithm**: Hybrid approach combining value and policy updates
- **Convergence Criterion**: δ = 0.1
- **Process**: Interleaves value updates and policy improvements in single loop
- **Advantage**: Can converge faster by not waiting for full value convergence

**Key Features**:
- Single-loop implementation
- Early policy updates
- Balanced approach between value and policy iteration

## Environment Details

### Grid World Structure
```
World dimensions: 14 × 51
- Border walls (value 1): Inaccessible obstacles
- Interior space (value 0): Navigable areas
- Goal location: [7, 10] with reward +100
- Obstacle penalty: -50
- Movement cost: -1 per step
```

### Action Space
8 possible movements from each state:
- **Cardinal directions**: Up, Down, Left, Right
- **Diagonal directions**: Up-Left, Up-Right, Down-Left, Down-Right

### Transition Model
- **Intended action**: 60% probability
- **Adjacent actions**: 20% probability each (stochastic environment)
- **Boundary handling**: Special cases for edge states

## Usage

### Prerequisites
```bash
pip install numpy matplotlib
```

### Running the Algorithms

1. **Value Iteration**:
   ```bash
   python value_iteration.py
   ```

2. **Policy Iteration**:
   ```bash
   python policy_iteration.py
   ```

3. **Generalized Policy Iteration**:
   ```bash
   python generalized_policy_iteration.py
   ```

### Output

Each algorithm produces:
1. **Value Function Visualization**: Grayscale heatmap showing state values
2. **Policy Visualization**: Arrows indicating optimal actions from each state
3. **World Map**: Grid showing accessible vs. inaccessible areas
4. **Performance Metrics**: Execution time and convergence information

## Implementation Details

### Core Functions

- `actions_possible(state)`: Returns valid actions from given state
- `model_distribution(from_state, to_state)`: Computes transition probabilities
- `policy_update(policy, state, optimal_state, close_states)`: Updates policy for given state
- `optimal_action(pi, state)`: Extracts optimal action from policy

### Data Structures

- **States**: 2D array of [row, column] coordinates
- **Value Function**: 1D array indexed by state position
- **Policy**: Dictionary mapping state-action pairs to probabilities
- **Rewards**: 2D array matching world dimensions

### Convergence Criteria

- **Value Iteration**: Maximum difference between consecutive value updates < δ
- **Policy Iteration**: Policy remains unchanged between iterations
- **Generalized**: Both value convergence and policy stability

## Performance Characteristics

### Convergence Speed
- **Value Iteration**: Typically requires more iterations but simpler implementation
- **Policy Iteration**: Often faster convergence, especially for structured problems
- **Generalized**: Variable performance depending on problem characteristics

### Memory Usage
- All algorithms use O(|S|) memory for value function storage
- Policy storage: O(|S| × |A|) where |A| is action space size
- Total memory: O(|S| × |A|) for full implementation

## Theoretical Background

### Markov Decision Process
An MDP is defined by the tuple (S, A, P, R, γ) where:
- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probability function P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor

### Bellman Equations
- **Value Function**: V(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]
- **Policy Function**: π(s) = argmax_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]

### Convergence Guarantees
- All three algorithms converge to optimal policy under standard MDP assumptions
- Value iteration: Guaranteed convergence for finite MDPs
- Policy iteration: Finite convergence due to finite policy space
- Generalized: Convergence under appropriate update schedules

## Extensions and Modifications

### Customizing the Environment
- Modify `World` array to change obstacle layout
- Adjust `rewards` array for different reward structures
- Change `gamma` for different discount factors

### Algorithm Parameters
- Modify convergence thresholds (`delta`)
- Adjust maximum iteration limits
- Change transition probability distributions

### Visualization Options
- Customize colormaps for value function display
- Modify arrow properties for policy visualization
- Add additional plotting features

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Puterman, M. L. (2014). Markov Decision Processes: Discrete Stochastic Dynamic Programming
- Bellman, R. (1957). Dynamic Programming

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the implementations or add new algorithms.

---

**Note**: This implementation is designed for educational purposes and demonstrates the core concepts of dynamic programming for MDPs. For production use, consider using established libraries like OpenAI Gym or similar reinforcement learning frameworks.
