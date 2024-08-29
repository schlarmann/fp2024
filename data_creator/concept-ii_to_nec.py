import necFile
INFILE = "indata/yagi_main.wire"
IN_FREQ = 2450 # MHz
EXCITATION_SEGMENT = 0


C = 299792458
WAVELENGTH = C / (IN_FREQ*1000)

with open(INFILE, "r") as f:
    lines = f.readlines()

# Get the number of wires
num_wires = int(lines[0])
wire_count = 0
exitation_position = 0
outfile = necFile.necFile()
outfile.initManual(IN_FREQ, EXCITATION_SEGMENT, 0, num_wires)
for wire in lines[1:]:
    wire = wire.strip().replace(" ", "\t")
    w = wire.strip().split("\t")
    startpoint = [float(w[0]), float(w[1]), float(w[2])]
    endpoint = [float(w[3]), float(w[4]), float(w[5])]
    radius = float(w[6])/1000 # um to mm
    number_basis_functions = int(w[7])

    if wire_count == EXCITATION_SEGMENT:
        outfile.exitation_position = int( number_basis_functions/2 -0.01) # assume center point
        print(outfile.exitation_position)

    outfile.addWireMM(startpoint, endpoint, radius, number_basis_functions)

    wire_count += 1

outfile.addComment("CONCEPT-II Importer")
outfile.addComment("Converted from "+INFILE)
outfile.addComment(f"Frequency manually set to {IN_FREQ} MHz, Excitation segment: {EXCITATION_SEGMENT}")

outfile.writeFile(INFILE+"_converted.nec")