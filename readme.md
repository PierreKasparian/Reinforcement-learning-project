# Wildfire Management with Deep Reinforcement Learning

This repository contains the code and environment implementation for the project comparing value-based reinforcement learning algorithms (**Pure DQN**, **Pure DDQN**, and **Rainbow DQN**) on a custom stochastic wildfire suppression environment.

---

## **How to Run the Code**


How to Run the Code:

1. **Installation:** Ensure you have Python installed along with the required dependencies from ```requirements.txt```

2. **Training and Evaluation Pipeline:** To train an agent and evaluate performance across paired test seeds:

```bash
python train_and_evaluate.py --algo rainbow --episodes 20000
```

Command-line Options:

```bash
--algo: Choose from dqn, ddqn, rainbow, or all (runs each agent sequentially).--episodes: Number of training episodes per algorithm (default: 20000).--eval-only: Skip training and evaluate existing saved agent checkpoints.
```

 3. **Standalone Training:** To train a single agent directly and save diagnostic plots and models:
 
 ```bash
 python train.py --algo rainbow --episodes 20000
 ```

4. **Visual Playback Simulation:** To render and visualize a saved agent model performing fire suppression using Pygame:

```bash
python agent_sim.py --algo rainbow --checkpoint ./checkpoints/rainbow_wildfire_final.pth
```


5. **Environment Simulation:** To launch an interactive manual session of the wildfire environment (press R during execution to trigger live weather events like rain):

```bash
python env_sim.py
```

## Script Descriptions:
```actions.py```: Defines discrete action types, orientations, action costs, and factorized action space mappings for the grid.
```buffer.py```: Implements uniform replay buffers and prioritized experience replay (PER) with $n$-step transition buffers.
```DQN.py```: Implements the standard Deep Q-Network agent with $\epsilon$-greedy exploration and target network updating.
```DDQN.py```: Implements the Double Deep Q-Network agent to decouple action selection from evaluation and mitigate maximization bias.
```rainbow.py```: Implements the full Rainbow DQN agent incorporating Categorical (C51) Distributional RL, Multi-Step Returns, Prioritized Experience Replay, and Noisy Networks.
```model.py```: Constructs the shared convolutional feature extractor and factorized spatial Q-network/distributional heads.
```forest_fire_env.py```: Implements the wildfire environment, simulating cellular automata fire dynamics, wind vector fields, and rain events.
```render.py```: Manages Pygame visualization routines for rendering grid states, active suppression zones, and weather overlays.
```train.py```: Manages single-agent training loops, checkpoint saves, and metric logging.
```train_and_evaluate.py```: Controls the complete multi-agent training pipeline, evaluation on fixed seeds, and benchmark summary generation.
```agent_sim.py```: Executes visual playback simulations using saved neural network checkpoints.
```env_sim.py```: Runs an interactive manual simulation of the environment.

### Authors

**Axel Bechtel**
Github: **Vencilo**

**Jesper Wentzell**
Github: **Kiopp**

**Pierre Kasparian**
Github: **PierreKasparian**

**Ahmet Alper Basturk**
Github: **AhmetAlperBasturk**