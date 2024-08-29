import necFile

# Empirical Value for ideal impedance at 2.45 GHz
lambda_length = 0.47512
a = 0.375
b = 0.0575
c = 0.0675
base_spacing = (b+c)/10
spacings = [
    0.5*base_spacing,
    1000*base_spacing
]

def mkMoxon(spacing, horizontal = False):
    nf = necFile.necFile()
    nf.initManual(2450, 0, 5, 6)
    
    # Driven element
    distPoint = (a)/2
    d = b+c+spacing
    if horizontal:
        nf.addWire([0,0, distPoint], [0,0,-distPoint], 0.005, 9)
        nf.addWire([0,0, distPoint], [0,b, distPoint], 0.005, 3)
        nf.addWire([0,0,-distPoint], [0,b,-distPoint], 0.005, 3)
        
        nf.addWire([0,d, distPoint], [0,d  ,-distPoint], 0.005, 9)
        nf.addWire([0,d, distPoint], [0,d-c, distPoint], 0.005, 3)
        nf.addWire([0,d,-distPoint], [0,d-c,-distPoint], 0.005, 3)
        
    else:
        nf.addWire([0, distPoint,0], [0,-distPoint,0], 0.005, 9)
        nf.addWire([0, distPoint,0], [0, distPoint,b], 0.005, 3)
        nf.addWire([0,-distPoint,0], [0,-distPoint,b], 0.005, 3)
        
        nf.addWire([0, distPoint,d], [0,-distPoint,d], 0.005, 9)
        nf.addWire([0, distPoint,d], [0, distPoint,d-c], 0.005, 3)
        nf.addWire([0,-distPoint,d], [0,-distPoint,d-c], 0.005, 3)

    nf.addComment("Generated Moxon antenna")
    if horizontal:
        nf.writeFile("validation_data/moxon_"+str(spacing)+"_Horizontal.nec")
    else:
        nf.writeFile("validation_data/moxon_"+str(spacing)+"_Vertical.nec")


def mkMoxon45(length, horizontal = False):
    nf = necFile.necFile()
    nf.initManual(2450, 0, 5, 6)
    
    # Driven element
    distPoint = ( a/2)/2**0.5
    d = b+c+spacing
    if horizontal:
        nf.addWire([ distPoint,0, distPoint], [-distPoint,0,-distPoint], 0.005, 9)
        nf.addWire([ distPoint,0, distPoint], [ distPoint,b, distPoint], 0.005, 3)
        nf.addWire([-distPoint,0,-distPoint], [-distPoint,b,-distPoint], 0.005, 3)
        
        nf.addWire([ distPoint,d, distPoint], [-distPoint,d  ,-distPoint], 0.005, 9)
        nf.addWire([ distPoint,d, distPoint], [ distPoint,d-c, distPoint], 0.005, 3)
        nf.addWire([-distPoint,d,-distPoint], [-distPoint,d-c,-distPoint], 0.005, 3)
        
    else:
        nf.addWire([-distPoint, distPoint,0], [ distPoint,-distPoint,0], 0.005, 9)
        nf.addWire([-distPoint, distPoint,0], [-distPoint, distPoint,b], 0.005, 3)
        nf.addWire([ distPoint,-distPoint,0], [ distPoint,-distPoint,b], 0.005, 3)
        
        nf.addWire([-distPoint, distPoint,d], [ distPoint,-distPoint,d], 0.005, 9)
        nf.addWire([-distPoint, distPoint,d], [-distPoint, distPoint,d-c], 0.005, 3)
        nf.addWire([ distPoint,-distPoint,d], [ distPoint,-distPoint,d-c], 0.005, 3)

    nf.addComment("Generated Moxon antenna")
    if horizontal:
        nf.writeFile("validation_data/moxon_"+str(length)+"_+45Deg.nec")
    else:
        nf.writeFile("validation_data/moxon_"+str(length)+"_-45Deg.nec")


for spacing in spacings:
    mkMoxon(spacing, True)
    mkMoxon(spacing, False)
    mkMoxon45(spacing, True)
    mkMoxon45(spacing, False)