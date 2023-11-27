import tensorflow as tf 
from tensorflow import keras 
from keras.optimizers import Adam 
import tensorflow_probability as tfp 
from networks import ActorCriticNetwork

class Agent:
    def __init__(self, alpha=0.003, gamma = 0.99, n_actions = 2 ):
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.action = None 
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_action=n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        self.action = action 

        return action.numpy()[0]

    def save_model(self):
        print(".... saving model....")
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print(".... Loading Models .....")
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        # In order to Calculate the gradients 
        with tf.GradientTape(persistent= True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value) # using squeeze in order to get rid of that batch dimention 
            state_value_ = tf.squeeze(state_value_) 
            # loss works best for the single value 
            # So it needs to be scaler containing a single value rather than a vector 

            action_probs  = tfp.distributions.Categorical(probs = probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma*state_value_(1 - int(done)) - state_value

            actor_loss = -log_prob*delta
            critic_loss = (tf.square(delta))/2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip( gradient, self.actor_critic.trainable_variables))



