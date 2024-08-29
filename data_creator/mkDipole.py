import necFile

# Empirical Value for ideal impedance at 2.45 GHz
lambda_length = 0.47512
in_length_factors = [
    0.8,
    0.9,
    0.95,
    1.0,
    1.05,
    1.1,
    1.2,
]

def mkDipole(length, horizontal = False):
    nf = necFile.necFile()
    nf.initManual(2450, 0, 5, 1)
    
    # Driven element
    distPoint = (lambda_length*length)/2
    if horizontal:
        nf.addWire([0,0,distPoint], [0,0,-distPoint], 0.015, 9)
    else:
        nf.addWire([0,distPoint,0], [0,-distPoint,0], 0.015, 9)

    nf.addComment("Generated dipole antenna")
    if horizontal:
        nf.writeFile("indata/dipole_"+str(length)+"_Horizontal.nec")
    else:
        nf.writeFile("indata/dipole_"+str(length)+"_Vertical.nec")


def mkDipole45(length, horizontal = False):
    nf = necFile.necFile()
    nf.initManual(2450, 0, 5, 1)
    
    # Driven element
    distPoint = ( (lambda_length*length)/2)/2**0.5
    if horizontal:
        nf.addWire([0,distPoint,distPoint], [0,-distPoint,-distPoint], 0.015, 9)
    else:
        nf.addWire([0,distPoint,-distPoint], [0,-distPoint,distPoint], 0.015, 9)

    nf.addComment("Generated dipole antenna")
    if horizontal:
        nf.writeFile("indata/dipole_"+str(length)+"_+45Deg.nec")
    else:
        nf.writeFile("indata/dipole_"+str(length)+"_-45Deg.nec")

for length in in_length_factors:
    mkDipole(length, True)
    mkDipole(length, False)
    mkDipole45(length, True)
    mkDipole45(length, False)