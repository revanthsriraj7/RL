import gymnasium as gym
import pickle
import numpy as np
import time

# Create environment
env = gym.make("Blackjack-v1", render_mode="human")

total_reward = 0

# Load Q-table
with open("q_table.pkl", "rb") as f:
    q_values = pickle.load(f)

# Play games
for episode in range(5):

    obs, info = env.reset()

    done = False

    print("\n==============================")
    print(f"GAME {episode + 1}")
    print("==============================")

    while not done:

        # Internal card details
        player_cards = env.unwrapped.player
        dealer_cards = env.unwrapped.dealer

        print(f"\nPlayer cards: {player_cards} | Total: {sum(player_cards)}")
        print(f"Dealer showing: {dealer_cards[0]}")

        print(f"State: {obs}")

        # Choose best learned action
        action = np.argmax(q_values[obs])

        action_name = "HIT" if action == 1 else "STAND"

        print(f"Agent Action: {action_name}")

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        obs = next_obs

        total_reward += reward

        time.sleep(1)

    # Final state
    print("\n----- FINAL RESULT -----")

    print(f"Final Player Cards: {env.unwrapped.player}")
    print(f"Final Dealer Cards: {env.unwrapped.dealer}")

    player_total = sum(env.unwrapped.player)
    dealer_total = sum(env.unwrapped.dealer)

    print(f"Player Total: {player_total}")
    print(f"Dealer Total: {dealer_total}")

    if reward == 1:
        print("RESULT: AGENT WON")
    elif reward == -1:
        print("RESULT: AGENT LOST")
    else:
        print("RESULT: DRAW")

    print(f"total reward; {total_reward}")

env.close()