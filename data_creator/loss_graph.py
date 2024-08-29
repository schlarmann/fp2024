import matplotlib.pyplot as plt

AVERAGING_WINDOW = 10
EPOCHS = 2500

genloss = []
discloss = []
with open("model/log_1724575437.8804302.csv") as f:
    f.readline()
    for line in f.readlines():
        line = line.split(",")
        genloss.append(float(line[1]))
        discloss.append(float(line[2]))

genloss_avg = []
discloss_avg = []
for i in range(0, len(genloss), AVERAGING_WINDOW):
    genloss_avg.append(sum(genloss[i:i+AVERAGING_WINDOW])/AVERAGING_WINDOW)
    discloss_avg.append(sum(discloss[i:i+AVERAGING_WINDOW])/AVERAGING_WINDOW)

factor = EPOCHS / len(genloss_avg) # 2500 is the number of epochs
skip = int(300/factor)+1

#plt.plot(genloss_avg, label="Generator Loss")
#plt.plot(discloss_avg, label="Discriminator Loss")
plt.plot(genloss, label="Generator Loss")
plt.plot(discloss, label="Discriminator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# Scale the x-axis to match the number of epochs
#plt.xticks([i for i in range(0, len(genloss_avg), skip)], [str(int(i*factor)) for i in range(0, len(genloss_avg), skip)])
plt.legend()
plt.show()