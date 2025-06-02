import gym # type: ignore

# 建立環境
env = gym.make("CartPole-v1", render_mode="human")  # 加上 render_mode 來顯示畫面
observation, _ = env.reset()

total_reward = 0
done = False

while not done:
    # 拆解觀測值
    cart_pos, cart_vel, pole_angle, pole_vel = observation

    # 固定策略：根據 pole angle 做判斷
    action = 0 if pole_angle < 0 else 1

    # 執行動作
    observation, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"總得分：{total_reward}")
env.close()
