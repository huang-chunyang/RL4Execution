import argparse
import json
import numpy as np
import torch

def main():
    # 示例代码：假设有一个自定义环境可以根据配置文件和策略进行初始化
    # 此处省略自定义环境初始化代码

    # 示例环境交互，假设我们有一个自定义环境类 CustomEnv
    # env = CustomEnv(config_path, policy_path)
    
    # 输出一些非JSON格式的内容
    print("Starting interaction with custom environment")
    print("Action taken:")

    # 这里直接返回假定的state, reward, done用于演示
    state = np.array([0.0, 0.1, 0.2, 0.3])  # 假设的新状态
    # reward = torch.Tensor([0.0, 0.1, 0.2, 0.3])   
    reward = [0, 2, 3]# 假设的奖励
    done = [0, 0, 1]           # 假设的完成标志

    # 将结果格式化为JSON并输出，添加前缀以便区分
    result = ({
        'state': state.tolist(),
        'reward': reward,
        'done': done
    }, 1)
    print("JSON_OUTPUT_START")
    print(json.dumps(result))
    print("JSON_OUTPUT_END")

if __name__ == "__main__":
    main()
