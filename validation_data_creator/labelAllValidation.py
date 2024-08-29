import os
import necFile
import necLabeler
import time

IN_DIR = "validation_data"
OUT_DIR = "labeled_validation_data"


maxWires = 60
failedFiles = []
print("Failed files: " + str(len(failedFiles)))
beginFiles = os.listdir(OUT_DIR) + failedFiles
print("Begin files: " + str(len(beginFiles)))
failFile = open("failedFiles3.txt", "a")
total = len(os.listdir(IN_DIR))# - len(beginFiles)
totalTimeStart = time.time()
lastTimeStart = time.time()
counter = 0

nl = necLabeler.necLabeler()

for file in os.listdir(IN_DIR):
    if file.endswith(".nec"):
        if file in beginFiles:
            print("+ Skipping " + file + "...")
            #counter += 1
            continue
        timeStart = time.time()
        if counter != 0:
            processed = counter
            remaining = (timeStart - totalTimeStart) * (total/processed)
            timeEstimate = time.strftime("%H:%M:%S", time.gmtime(remaining))
        else:
            timeEstimate = "Calculating..."
        print("# Labeling " + file + f" ({counter+1}/{total}) - Estimate: {timeEstimate}...")
        nf = necFile.necFile()
        try:
            nf.initFile(IN_DIR + "/" + file)
            nf.extendWiresTo(maxWires)
        except:
            print("- Failed to read file " + file + "...")
            failFile.write(file + "\n")
            failFile.flush()
            counter += 1
            continue
        if len(nf.wires) != maxWires:
            print("- File with wrong number of wires: " + file)
            failFile.write(file + "\n")
            failFile.flush()
            counter += 1
            continue
        wasSuccessful = False
        for i in range(5): #Retry 3 times
            try:
                nf2 = nl.labelFile(nf)
            except:
                for i in range(len(nf.wires)):
                    nf.wires[i][nf.WIRE_RADIUS] /= 2 # Decrease the radius of the wires
                nl.reset()
            else:
                #nl.showGain()
                nl.reset()
                nf2.writeFile(OUT_DIR + "/" + file)
                wasSuccessful = True
                counter += 1
                break
        if not wasSuccessful:
            print("- Failed to label " + file + "...")
            failFile.write(file + "\n")
            failFile.flush()
            counter += 1
failFile.close()