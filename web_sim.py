import traci
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import DQN
import uuid
import threading
import time
import os

sim_data = {
    "step": 0,
    "reward": 0.0,
    "wait_time": 0.0,
    "current_phase": "-"
}

class TrafficLightEnvHeadless(Env):
    def __init__(self, config_file, max_steps=1000):
        super().__init__()
        self.config_file = config_file
        self.max_steps = max_steps
        self.step_count = 0
        self.connection_label = str(uuid.uuid4())
        self.is_connected = False
        self.tl_ids = []
        self.controlled_lanes = {}
        
        traci.start(["sumo", "-c", self.config_file, "--no-step-log", "true"], label=self.connection_label)
        self.is_connected = True
        traci.simulationStep()
        
        self.tl_ids = traci.trafficlight.getIDList()
        for tl_id in self.tl_ids:
            self.controlled_lanes[tl_id] = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id)))
        
        self.action_space = Discrete(3)
        obs_size = sum(len(lanes) for lanes in self.controlled_lanes.values()) + 2 * len(self.tl_ids)
        self.observation_space = Box(low=0, high=1000, shape=(obs_size,), dtype=np.float32)
        
        traci.close()
        self.is_connected = False

    def reset(self, seed=None, options=None):
        if self.is_connected:
            try:
                traci.close()
            except: pass
            self.is_connected = False
        traci.start(["sumo", "-c", self.config_file, "--no-step-log", "true"], label=self.connection_label)
        self.is_connected = True
        traci.simulationStep()
        self.step_count = 0
        return self._get_state(), {}

    def step(self, action):
        for tl_id in self.tl_ids:
            current_phase = traci.trafficlight.getPhase(tl_id)
            current_duration = traci.trafficlight.getPhaseDuration(tl_id)
            green_phases = [i for i, state in enumerate(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].phases) if 'G' in state.state]
            if action == 0 and current_phase in green_phases:
                traci.trafficlight.setPhaseDuration(tl_id, current_duration + 5)
            elif action == 1:
                traci.trafficlight.setPhase(tl_id, (current_phase + 1) % len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases))
        
        traci.simulationStep()
        self.step_count += 1
        
        state = self._get_state()
        reward = sum(-traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())
        done = self.step_count >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0
        return state, reward, done, False, {}

    def _get_state(self):
        try:
            state = []
            for tl_id in self.tl_ids:
                waiting = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.controlled_lanes[tl_id]]
                phase = traci.trafficlight.getPhase(tl_id)
                current_time = traci.simulation.getTime()
                time_remaining = np.clip(traci.trafficlight.getPhaseDuration(tl_id) - (traci.trafficlight.getNextSwitch(tl_id) - current_time), 0, 1000)
                state.extend(waiting + [phase, time_remaining])
            return np.array(state, dtype=np.float32)
        except:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

def run_simulation_loop():
    global sim_data
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intersection.sumocfg")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/kingcircle")
    
    env = TrafficLightEnvHeadless(config_file)
    model = DQN.load(model_path)
    
    while True:
        obs, _ = env.reset()
        total_reward = 0
        try:
            for step in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                
                sim_data["step"] = step
                sim_data["reward"] = float(total_reward)
                sim_data["wait_time"] = float(-reward)
                sim_data["current_phase"] = int(obs[-2]) if len(obs) >= 2 else 0
                
                time.sleep(0.5) 
                if done: break
        except Exception as e:
            print("SIM ERROR:", e)
            time.sleep(5)
            
def start_simulation_thread():
    t = threading.Thread(target=run_simulation_loop, daemon=True)
    t.start()
    return t
