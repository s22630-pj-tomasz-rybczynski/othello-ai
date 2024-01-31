"""
Authors: Tomasz Rybczyński, Filip Marcoń

In order to be able to run script with this game you will need:
Python at least 3.8
gymnasium, numpy, ALE

To run script you need to run command "python3 main.py"

The provided code is an implementation of a simple policy gradient method for training an agent to play the Pong
game using the A.L.E (Arcade Learning Environment) framework."""

import gymnasium as gym
import numpy as np


def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    processed_observation = input_observation[35:195]

    processed_observation = np.array(processed_observation)

    if processed_observation.size == 0:
        return np.zeros(input_dimensions), None

    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1
    processed_observation = processed_observation.astype(float).ravel()

    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image


def downsample(image):
    downsampled_image = image[::2, ::2, :]
    return downsampled_image


def remove_color(image):
    return image[:, :, 0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(vector):
    vector[vector < 0] = 0
    return vector


def neural_net(observation_matrix, weights):
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values


def Move_up_or_down(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        return 2
    else:
        return 3


def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }


def weights_update(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g ** 2
        weights[layer_name] += (learning_rate * g) / (np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name])  # reset batch gradient buffer


def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def discount_plus_rewards(gradient_log_p, episode_rewards, gamma):
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards


def save_model(weights, filename='model_weights.npy'):
    np.save(filename, weights)
    print(f'Model weights saved to {filename}')


def load_model(filename='model_weights.npy'):
    try:
        loaded_weights = np.load(filename, allow_pickle=True).item()
        print(f'Model weights loaded from {filename}')
        return loaded_weights
    except FileNotFoundError:
        print(f'No saved model found at {filename}. Starting with random weights.')
        return None


def main():
    env = gym.make("ALE/Pong-v5", render_mode="human")

    observation = env.reset()  # This gets us the image

    episode_number = 0
    batch_size = 10
    gamma = 0.99
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-4
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    weights = load_model()  # Load existing weights or start with None for random initialization

    if weights is None:
        weights = {
            '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
            '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
        }

    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                      prev_processed_observations,
                                                                                      input_dimensions)
        hidden_layer_values, up_probability = neural_net(processed_observations, weights)

        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = Move_up_or_down(up_probability)

        step_result = env.step(action)
        observation, reward, done, info = step_result[:4]

        reward_sum += reward
        episode_rewards.append(reward)

        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)

        if done:
            episode_number += 1

            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            episode_gradient_log_ps_discounted = discount_plus_rewards(episode_gradient_log_ps, episode_rewards, gamma)

            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                weights
            )

            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                weights_update(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
            observation = env.reset()  # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            reward_sum = 0
            prev_processed_observations = None

            if episode_number % 10 == 0:
                save_model(weights)


main()
