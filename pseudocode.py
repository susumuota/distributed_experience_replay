
@ray.remote
class ReplayMemory:
    def __init__(self):
    def save_episode(self, transitions):
    def get_batch(self):

@ray.remote
class SharedStorage:
    def __init__(self):
    def get_weights(self):
    def set_weights(self, weights):

@ray.remote
class Actor:
    def __init__(self, env, epsilon, gamma):
    def play_loop(self, storage, replay_memory):
        while True:
            weights = ray.get(storage.get_weights.remote())
            self.model.set_weights(copy.deepcopy(weights))
            transitions = self.play_episode()
            replay_memory.save_episode.remote(transitions)

@ray.remote
class Learner:
    def __init__(self):
    def learn_loop(self, storage, replay_memory):
        while True:
            batch = ray.get(replay_memory.get_batch.remote())
            history = self.update_weights(batch)
            weights = self.model.get_weights()
            storage.set_weights.remote(weights)

def main():
    ray.init(address='auto')

    replay_memory = ReplayMemory.remote()
    storage = SharedStorage.remote()

    learner_worker = Learner.remote()
    learner_worker.learn_loop.remote(storage, replay_memory)

    actor_workers = [Actor.remote(gym.make('CartPole-v0'),
                                  epsilon_i(n, i), gamma)
                     for i in range(n)]
    [actor_worker.play_loop.remote(storage, replay_memory)
     for actor_worker in actor_workers]

    try:
        while True:
            # print log
            time.sleep(1.0)
    except KeyboardInterrupt as err:
        pass

    ray.shutdown()

if __name__ == '__main__':
    main()

