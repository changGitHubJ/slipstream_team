import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("./training_log3.txt", "r") as f:
        lines = f.readlines()
        reward = []
        for line in lines:
            vals = line.split(" ")
            reward.append(float(vals[4]))
        
        plt.plot(reward)
        plt.show()