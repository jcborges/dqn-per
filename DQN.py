
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from Memory import *


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.999    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.dqn_learning_rate = 0.001
        self.model = self._build_model()
        self.memory = Memory(1000000)  # PER Memory
        self.batch_size = 32

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.dqn_learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        # Calculate TD-Error for Prioritized Experience Replay
        td_error = reward + self.gamma * np.argmax(self.model.predict(next_state)[0]) - np.argmax(
            self.model.predict(state)[0])
        # Save TD-Error into Memory
        self.memory.add(td_error, (state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Exploration
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action (Exploitation)

    def replay(self):
        batch, idxs, is_weight = self.memory.sample(self.batch_size)
        for i in range(self.batch_size):
            state, action, reward, next_state, done = batch[i]
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            else:
                target = reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Gradient Update. Pay attention at the sample weight as proposed by the PER Paper
            self.model.fit(state, target_f, epochs=1, verbose=0, sample_weight=np.array([is_weight[i]]))
        if self.epsilon > self.epsilon_min: # Epsilon Update
            self.epsilon *= self.epsilon_decay