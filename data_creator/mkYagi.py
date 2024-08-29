import necFile

# Values from NBS Technical Note 688
in_lengths = [
    [0.482,0.442],
    [0.482,0.428,0.424,0.428],
    [0.482,0.428,0.420,0.420,0.428],
    [0.482,0.432,0.415,0.407,0.398,0.390,0.390,0.390,0.390,0.398,0.407],
    [0.482,0.428,0.420,0.407,0.398,0.394,0.390,0.386,0.386,0.386,0.386,0.386,0.386,0.386,0.386,0.386],
    [0.487,0.451,0.438,0.428,0.422,0.416,0.413,0.410,0.408,0.405,0.402,0.400,0.398,0.396]
]
in_spacings = [
    [0.2],
    [0.2],
    [0.25],
    [0.2],
    [0.2],
    [0.149,0.060,0.112,0.155,0.193,0.224,0.253,0.276,0.296,0.314,0.328,0.341,0.350,0.361]
]
len_dipole = 0.47512

def mkYagi(lengths, spacings, horizontal = False):
    if len(spacings) == 1:
        spacings = spacings * len(lengths)
    nf = necFile.necFile()
    nf.initManual(2450, 1, 5, len(lengths)+1)
    # Reflector
    distPoint = lengths[0]/2
    if horizontal:
        nf.addWire([-spacings[0],0,distPoint], [-spacings[0],0,-distPoint], 0.015, 5)
    else:
        nf.addWire([-spacings[0],distPoint,0], [-spacings[0],-distPoint,0], 0.015, 5)
    
    # Driven element
    distPoint = len_dipole/2
    if horizontal:
        nf.addWire([0,0,distPoint], [0,0,-distPoint], 0.015, 9)
    else:
        nf.addWire([0,distPoint,0], [0,-distPoint,0], 0.015, 9)

    point = 0
    for i in range(1,len(lengths)):
        distPoint = lengths[i]/2
        point += spacings[i]
        if horizontal:
            nf.addWire([point,0,distPoint], [point,0,-distPoint], 0.015, 5)
        else:
            nf.addWire([point,distPoint,0], [point,-distPoint,0], 0.015, 5)

    nf.addComment("Generated Yagi-Uda antenna")
    if horizontal:
        nf.writeFile("indata/yagi_"+str(len(lengths)+1)+"_Horizontal.nec")
    else:
        nf.writeFile("indata/yagi_"+str(len(lengths)+1)+"_Vertical.nec")


def mkYagi45(lengths, spacings, horizontal = False):
    if len(spacings) == 1:
        spacings = spacings * len(lengths)
    nf = necFile.necFile()
    nf.initManual(2450, 1, 5, len(lengths)+1)
    # Reflector
    distPoint = (lengths[0]/2)/2**0.5
    if horizontal:
        nf.addWire([-spacings[0],distPoint,distPoint], [-spacings[0],-distPoint,-distPoint], 0.015, 5)
    else:
        nf.addWire([-spacings[0],distPoint,-distPoint], [-spacings[0],-distPoint,distPoint], 0.015, 5)
    
    # Driven element
    distPoint = (len_dipole/2)/2**0.5
    if horizontal:
        nf.addWire([0,distPoint,distPoint], [0,-distPoint,-distPoint], 0.015, 9)
    else:
        nf.addWire([0,distPoint,-distPoint], [0,-distPoint,distPoint], 0.015, 9)

    point = 0
    for i in range(1,len(lengths)):
        distPoint = (lengths[i]/2)/2**0.5
        point += spacings[i]
        if horizontal:
            nf.addWire([point,distPoint,distPoint], [point,-distPoint,-distPoint], 0.015, 5)
        else:
            nf.addWire([point,distPoint,-distPoint], [point,-distPoint,distPoint], 0.015, 5)

    nf.addComment("Generated Yagi-Uda antenna")
    if horizontal:
        nf.writeFile("indata/yagi_"+str(len(lengths)+1)+"_+45Deg.nec")
    else:
        nf.writeFile("indata/yagi_"+str(len(lengths)+1)+"_-45Deg.nec")

for i in range(len(in_lengths)):
    mkYagi(in_lengths[i], in_spacings[i], True)
    mkYagi(in_lengths[i], in_spacings[i], False)
    mkYagi45(in_lengths[i], in_spacings[i], True)
    mkYagi45(in_lengths[i], in_spacings[i], False)