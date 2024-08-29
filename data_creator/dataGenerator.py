import os
import time
import random
import necFile
import numpy as np

IN_DIR = "indata"
OUT_DIR = "generatedData"
NUM_ELEMENTS = 60
CHOSEN_FROM_FILES = 40
STEP_FILES = 2
STEPS = STEP_FILES+1
FUZZ = 2

def fuzzNecFile(nf:necFile.necFile):
    for i in range(FUZZ):
        nf2 = nf.deepcopy()
        nf2.fuzz(atol=nf.SIGMA)
        name = nf2.filename.split(".nec")[0] + "_fuzz" + str(i+1) + ".nec"
        nf2.writeFile(name)

def stepFiles(nf1:necFile.necFile, nf2:necFile.necFile):
    for j in range(1,STEPS):
        nf1_wc = nf1.deepcopy()
        nf2_wc = nf2.deepcopy()
        nf3 = nf1_wc.deepcopy()

        for i in range(NUM_ELEMENTS):
            supportingVectorS = nf2_wc.wires[i][nf3.WIRE_START] - nf1_wc.wires[i][nf3.WIRE_START]
            supportingVectorE = nf2_wc.wires[i][nf3.WIRE_END] - nf1_wc.wires[i][nf3.WIRE_END]
            nf3.wires[i][nf3.WIRE_START] += supportingVectorS * j/(STEPS+1)
            nf3.wires[i][nf3.WIRE_END] += supportingVectorE * j/(STEPS+1)
        nf3.calculateUniqueNodes()

        name = nf1_wc.filename.split(".nec")[0].split("/")[1] + "_" + nf2_wc.filename.split(".nec")[0].split("/")[1] + "_step" + str(j) + ".nec"
        
        try:
            nf3.writeFile(OUT_DIR + "/" + name)
            fuzzNecFile(nf3)
        except:
            # Ignore calculation errors here...
            print("\t\tCalculation error at step " + str(j) + " for " + name + "...")
            continue

beginFiles = os.listdir(OUT_DIR)
total = len(os.listdir(IN_DIR))
totalTimeStart = time.time()
lastTimeStart = time.time()
counter = 0
for file in os.listdir(IN_DIR):
    if file.endswith(".nec"):
        if file in beginFiles:
            print("Skipping " + file + "...")
            counter += 1
            continue
        timeStart = time.time()
        if counter != 0:
            processed = counter
            remaining = (timeStart - totalTimeStart) * (total/processed)
            timeEstimate = time.strftime("%H:%M:%S", time.gmtime(remaining))
        else:
            timeEstimate = "Calculating..."
        print("Processing " + file + f" ({counter+1}/{total}) - Estimate: {timeEstimate}...")
        # Copy the file to the output directory
        nf = necFile.necFile()
        nf.initFile(IN_DIR + "/" + file)
        nf.extendWiresTo(NUM_ELEMENTS)
        nf.writeFile(OUT_DIR + "/" + file)
        fuzzNecFile(nf)

        files2 = [file2 for file2 in os.listdir(IN_DIR) if file2.endswith(".nec") and file[0:4] != file2[0:4]]
        files2 = random.sample(files2, CHOSEN_FROM_FILES)
        for file2 in files2:

            print("\tStepping with " + file2 + "...")
            nf2 = necFile.necFile()
            nf2.initFile(IN_DIR + "/" + file2)
            nf2.extendWiresTo(NUM_ELEMENTS)
            
            stepFiles(nf, nf2)

        counter += 1