import copy
import random
import numpy as np
import matplotlib.pylab as pylab
import environment as env
from keras import layers
import keras

EPISODES = 1000


class DeepSARSAgent:
    def __init__(self, state_size: int):
        self.load_model = False
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)
        self.state_size = state_size
        self.discount_factor = 0.99  # gamma
        self.learning_rate = 0.001

        self.epsilon = 1.
        self.epsilon_decay = .9997
        self.epsilon_min = 0.1
        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights("./model/deep_sarsa.weights.h5")

    def build_model(self):
        # 인공신경망 생성
        model = keras.models.Sequential()
        model.add(layers.Dense(12, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(12, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        # 0과 1 사이의 수가 epsilon보다 작으면 = epsilon의 확률로
        if np.random.rand() <= self.epsilon:
            # 무작위로 행동 선택
            return random.randrange(self.action_size)
        else:
            # Q 함수의 결과값 중 가장 큰 값을 가지는 행동 출력
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action):

        # epsilon의 값을 감소시키는 코드
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 데이터 전처리(몰라도 됨)
        state = np.float32(state)
        next_state = np.float32(next_state)

        # target = Q(S(t), A(t))
        target = self.model.predict(state)[0]

        # new Q(S(t), A(t)) = R + r * Q(S(t+1), A(t+1))
        target[action] = (reward + self.discount_factor * self.model.predict(next_state)[0][next_action])

        # 데이터 전처리
        target = np.reshape(target, [1, 5])

        # LR을 고려한 인공신경망 수정
        self.model.fit(state, target, epochs=1, verbose=0)


if __name__ == "__main__":
    state_size = len(env.get_state())

    agent = DeepSARSAgent(state_size)

    global_step = 0
    scores, episodes = [], []

    check = [0, 0]

    for e in range(EPISODES):

        # 게임 초기화
        done = False
        score = 0
        env.reset()
        state = env.get_state()
        state = np.reshape(state, [1, state_size])

        step = 0

        while not done:
            # fresh env
            global_step += 1

            # Agent로부터 취할 행동을 받아오기 (A(t))
            action = agent.get_action(state)

            # 게임 진행 후, 새로운 상태와 보상 받기 (S(t+1), R(t+1))
            next_state, reward, done = env.step(action)

            # 데이터 전처리
            next_state = np.reshape(next_state, [1, state_size])

            # 다음 행동 선택 (A(t+1))
            next_action = agent.get_action(next_state)

            # 모델 학습(S(t+1))
            agent.train_model(state, action, reward, next_state, next_action)

            # 살이 있는 경우 점수 1점 추가
            score += 1

            # S(t+1)을 S(t)로 수정
            state = copy.deepcopy(next_state)

            if done:
                scores.append(score)
                episodes.append(e)
                pylab.title('DeepSarsa')
                pylab.plot(episodes, scores, 'b')
                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)
                pylab.savefig("./save_graph/deep_sarsa.png")

        if e % 100 == 0:
            agent.model.save_weights("./save_model/deep_sarsa.weights.h5")

        if e % 15 == 0:
            pylab.show()

        if step >= 250:
            done = True
        else:
            step += 1

