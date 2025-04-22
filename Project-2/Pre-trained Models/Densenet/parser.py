import re
from matplotlib import pyplot as plt


def plotLines(epochNum, train_losses, test_losses):
    # plt.figure(figsize=(20, 10))
    # plot the loss
    # plt.plot(1, 2, 2).set_title("Train v/s Test loss")
    title = "Losses for Fold - "+str(epochNum)
    plt.title(title)
    plt.plot(train_losses, label='Training loss', linewidth=3, color="y")
    plt.plot(test_losses, label='Test loss', linewidth=3, color="r")
    plt.legend()
    # save the result
    plt.savefig("Epoch_" + str(epochNum) + '.png', dpi=900)
    plt.clf()


with open("/Users/jarvis/dump.txt", "r") as f:
    text = f.read()

allFolds = text.split("Fold")
allFolds.pop(0)

hashmap = {}

for idx in range(len(allFolds)):
    fold = allFolds[idx]
    epochs = fold.split("INFO Epoch")
    epochs.pop(0)

    hashmap[idx] = {"avg_loss": [], "avg_val_loss": []}

    for epoch in epochs:
        # epoch = epochs[idx]
        # epoch.pop(0)
        match = re.search("avg_loss:\s+(?P<avg_loss>\d+\.\d+)", epoch)
        hashmap[idx]["avg_loss"].append(float(match.group("avg_loss")))
        match = re.search("avg_val_loss:\s+(?P<avg_val_loss>\d+\.\d+)", epoch)
        hashmap[idx]["avg_val_loss"].append(float(match.group("avg_val_loss")))

for item in hashmap:
    plotLines(item, hashmap[item]['avg_loss'], hashmap[item]['avg_val_loss'])
    # print(hashmap[item]['avg_val_loss'])
