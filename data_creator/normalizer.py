import os
import necFile
import time

LAST_START = 10745

IN_DIR = "labeledData"

EXPECTED_WIRES = 60
EXPECTED_LENGTH = EXPECTED_WIRES*8 + 7

total = len(os.listdir(IN_DIR))
totalTimeStart = time.time()
lastTimeStart = time.time()
counter = LAST_START

csvFile = open("normalized_output.csv", "w")
csvFile.write("maxGain, frontBackRatio, absImpedance, freqScore, antennaLengthScore, greatestExtentScore, excitationSegment,")
for i in range(EXPECTED_WIRES):
    csvFile.write(f"wire{i}StartX,wire{i}StartY,wire{i}StartZ,")
    csvFile.write(f"wire{i}EndX,wire{i}EndY,wire{i}EndZ,")
    csvFile.write(f"wire{i}Radius,wire{i}Segments,")
csvFile.write("\n")

for file in os.listdir(IN_DIR)[LAST_START:]:
    if file.endswith(".nec"):
        timeStart = time.time()
        if counter != 0:
            processed = counter
            remaining = (timeStart - totalTimeStart) * (total/processed)
            timeEstimate = time.strftime("%H:%M:%S", time.gmtime(remaining))
        else:
            timeEstimate = "Calculating..."
        print("# Normalizing " + file + f" ({counter+1}/{total}) - Estimate: {timeEstimate}...")
        nf = necFile.necFile()
        try:
            nf.initFile(IN_DIR + "/" + file)
        except:
            print("- Failed to read file " + file + "...")
            counter += 1
            continue
        if len(nf.wires) != EXPECTED_WIRES:
            print("- File with wrong number of wires: " + file)
            counter += 1
            continue
        normNec = nf.normalize()
        if len(normNec) != EXPECTED_LENGTH:
            print("- File with wrong length: " + file+" "+str(len(normNec)))
            counter += 1
            continue
        wasSuccessful = -1
        for i in range(len(normNec)):
            if normNec[i] > 1 or normNec[i] < -1:
                wasSuccessful = i
                break
        if not wasSuccessful == -1:
            print("- Failed to normalize " + file + f": normNec[{wasSuccessful}] = {normNec[wasSuccessful]}")
            counter += 1
            continue
        for i in range(len(normNec)):
            csvFile.write(str(normNec[i]) + ",")
        csvFile.write("\n")
        csvFile.flush()
        counter += 1
        
csvFile.close()