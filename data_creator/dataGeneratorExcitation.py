import os
import time
import random
import necFile
import numpy as np

IN_DIR = "generatedData"
OUT1_DIR = "labeledData"
OUT_DIR = "changedExcitation"
NUM_ELEMENTS = 60
CHOSEN_FROM_FILES = 1000
FILES_FACTOR = 3

beginFiles = [file2 for file2 in  os.listdir(OUT1_DIR) if "_excitation" not in file2]
print("Files in " + OUT1_DIR + ": " + str(len(beginFiles)))
usedFiles = random.sample(beginFiles, CHOSEN_FROM_FILES)
total = len(usedFiles)
totalTimeStart = time.time()
lastTimeStart = time.time()
counter = 0
for file in usedFiles:
    if file.endswith(".nec"):
        timeStart = time.time()
        processed = counter
        if processed > 0:
            remaining = (timeStart - totalTimeStart) * (total/processed)
            timeEstimate = time.strftime("%H:%M:%S", time.gmtime(remaining))
        else:
            timeEstimate = "Calculating..."
        print("Processing " + file + f" ({counter+1}/{total}) - Estimate: {timeEstimate}...")
        # Copy the file to the output directory
        nf = necFile.necFile()
        nf.initFile(IN_DIR + "/" + file)
        for i in range(FILES_FACTOR):
            nf2 = nf.deepcopy()
            excitationWire = random.randint(0, NUM_ELEMENTS-1)
            nf2.excitation_segment = excitationWire
            nf2.exitation_position = 1 + int(nf2.wires[excitationWire][nf2.WIRE_SEGMENTS]/2)
            name = OUT_DIR + "/" + file.split(".nec")[0] + "_excitation" + str(i+1) + ".nec"
            nf2.writeFile(name)
        counter += 1