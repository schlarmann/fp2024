import subprocess
import time

RETRIES = 50
DO_INITIAL = True

if(DO_INITIAL):
    p1 = subprocess.Popen(["python3.11", "/home/paul/Dokumente/fp/src/model/train_cgan3d.py"])
    p1.wait() 
    print("Initital Training Done!")
    time.sleep(5*60) #wait 5 minutes until system "restabilizes"

totalstart = time.time()
for i in range(RETRIES):
    #p1 = subprocess.Popen(["pwd"])
    starttime = time.time()
    p1 = subprocess.Popen(["python3.11", "/home/paul/Dokumente/fp/src/model/train_fileload.py"])
    p1.wait()
    endtime = time.time()
    delta = endtime - starttime
    eta = ((endtime-totalstart)/(i+1))*(RETRIES-i+1) / 3600
    print(f"{i+1}/{RETRIES} Done! Took {delta} seconds. Verbleibend: {eta} h")
    time.sleep(5*60) #wait 5 minutes until system "restabilizes"
