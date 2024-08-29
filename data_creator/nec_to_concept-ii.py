import necFile
INFILE = "indata/yagi_15_no_main.wire_converted.nec"

outfile = necFile.necFile()
outfile.initFile(INFILE)

with open(INFILE+"_converted.wire", "w") as f:
    f.write(f"{len(outfile.wires)}\n")
    for wire in outfile.wires:
        start = outfile.filePointToMMPoint(wire[0])
        end = outfile.filePointToMMPoint(wire[1])
        radius = outfile.convertRelativeToM(wire[2])*1000
        f.write(f"{start[0]:g}\t{start[1]:g}\t{start[2]:g}\t{end[0]:g}\t{end[1]:g}\t{end[2]:g}\t{int(radius*1000)}\t{int(wire[3])}\n")

exitation_location = outfile.exitation_position
if exitation_location == 0:
    exitation_location = "Start"
elif exitation_location == outfile.wires[outfile.excitation_segment][3]:
    exitation_location = "End"
else:
    exitation_location = "Center"

print(f"Set Frequency to {outfile.freq} MHz; Excitation segment is wire {outfile.excitation_segment}; Location \"{exitation_location}\"")