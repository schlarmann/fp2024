import numpy as np
import io

class necFile:
    COMMENT = "#"

    # Metadata:
    # Version, Antenna frequency, exitation element, number of wire segments
    META = "$ "
    META_SEP = " "
    META_VALUE_SEP = "="
    META_VERSION = "V"
    META_FREQ = "FREQ"
    META_EXCITATION_SEGMENT = "EXS"
    META_EXCITATION_POSITION = "EXP"
    META_WIRE_SEGMENTS = "WS"

    WIRE = "> "
    WIRE_SEP = "\t"
    WIRE_START = 0
    WIRE_END = 1
    WIRE_RADIUS = 2
    WIRE_SEGMENTS = 3

    LABEL = "%"
    LABEL_SEP = META_SEP
    LABEL_VALUE_SEP = META_VALUE_SEP
    # maxGain, frontBackRatio, Zscore, freqScore, antennaLengthScore, greatestExtentScore
    LABEL_MAX_GAIN = "MG"
    LABEL_FRONT_BACK_RATIO = "FBR"
    LABEL_ZSCORE = "ZS"
    LABEL_FREQ_SCORE = "FS"
    LABEL_ANTENNA_LENGTH_SCORE = "ALS"
    LABEL_GREATEST_EXTENT_SCORE = "GES"

    FLOAT_EPSILON = 0.001/1000 # 1 um
    SIGMA  = 1/20 # Wavelengths

    MAX_VECTOR_POINT = 20
    MAX_IMPEDANCE = 10000 # Ohms
    MAX_WIRE_RADIUS = 0.1
    MAX_ANTENNA_LENGTH = 20
    MAX_GAIN = 100
    MAX_FBR = 40
    MAX_SEGMENTS = 20
    MAX_WIRES = 60

    C = 299792458

    VERSION = "0.2"

    X = 0
    Y = 1
    Z = 2

    def __init__(self, string=None, filename=None):
        self.hasLabels = False
        self.comment = []
        self.wires = []
        self.filename = filename
        self.fileversion = self.VERSION
        if not string is None:
            self.initString(string)

    def __repr__(self):
        return f'necFile("""{self.toString()}""")'
    def __str__(self):
        return self.toString()
    def deepcopy(self):
        return necFile(self.toString(), self.filename)

    def initManual(self, freq, excitation_segment, exitation_position, wire_segments):
        self.fileversion = self.VERSION
        self.freq = freq
        self.wavelength = self.C / (self.freq*1000000) # MHz to Hz
        self.FLOAT_EPSILON = self.convertMToRelative(self.FLOAT_EPSILON)
        self.excitation_segment = excitation_segment
        self.exitation_position = exitation_position
        self.wire_segments = wire_segments


    def initFile(self, file):
        self.readFile(file)
        self.wavelength = self.C / (self.freq*1000000) # MHz to Hz
        self.FLOAT_EPSILON = self.convertMToRelative(self.FLOAT_EPSILON)

    def initString(self, stringVal):
        self.fromString(stringVal)
        self.wavelength = self.C / (self.freq*1000000) # MHz to Hz
        self.FLOAT_EPSILON = self.convertMToRelative(self.FLOAT_EPSILON)

    def convertRelativeToM(self, value):
        return value*self.wavelength
    def convertMToRelative(self, value):
        return value/self.wavelength

    def mmPointToFilePoint(self, point_mm):
        return self.mPointToFilePoint(point_mm/1000)
    def filePointToMMPoint(self, point):
        return self.filePointToMPoint(point)*1000

    def mPointToFilePoint(self, point_mm):
        return self.convertMToRelative(point_mm)
    def filePointToMPoint(self, point):
        return self.convertRelativeToM(point)

    def addWireMM(self, in_startpoint_mm, in_endpoint_mm, radius_mm, number_basis_functions):
        startpoint_mm = np.array(in_startpoint_mm)
        endpoint_mm = np.array(in_endpoint_mm)
        startpoint = self.mmPointToFilePoint(startpoint_mm)
        endpoint = self.mmPointToFilePoint(endpoint_mm)
        radius = self.convertMToRelative(radius_mm/1000)
        self.addWire(startpoint, endpoint, radius, number_basis_functions)

    def pointEqual(self, p1, p2):
        return np.allclose(p1,p2,rtol=0, atol=self.FLOAT_EPSILON)

    def totalWireLength(self):
        length = 0
        for wire in self.wires:
            length += self.wireLength(wire)
        return length

    def maxDistance(self):
        max = 0
        for node in self.nodes:
            length = np.linalg.norm(node)
            if length > max:
                max = length
        return max

    def wireLength(self, wire):
        return np.linalg.norm(wire[self.WIRE_END]-wire[self.WIRE_START])

    def addWire(self, in_startpoint, in_endpoint, radius, number_basis_functions):
        startpoint = np.array(in_startpoint)
        endpoint = np.array(in_endpoint)

        # Check if startpoint and endpoint are the same
        if self.pointEqual(startpoint, endpoint):
            raise Exception("Startpoint and endpoint are the same")
        # Check if radius is 0
        if radius == 0:
            raise Exception("Radius is 0")
        if len(self.wires) == self.wire_segments:
            raise Exception("Too many wires added to file")
            

        # Check if wire is already in file
        for w in self.wires:
            if self.pointEqual(w[self.WIRE_START], startpoint) and self.pointEqual(w[self.WIRE_END], endpoint):
                raise Exception("Wire already in file")

        # Equalize points
        for w in self.wires:
            if self.pointEqual(w[self.WIRE_START], startpoint):
                startpoint = w[self.WIRE_START]
            if self.pointEqual(w[self.WIRE_END], endpoint):
                endpoint = w[self.WIRE_END]
            if self.pointEqual(w[self.WIRE_END], startpoint):
                startpoint = w[self.WIRE_END]
            if self.pointEqual(w[self.WIRE_START], endpoint):
                endpoint = w[self.WIRE_START]
        
        if len(self.wires) == 0:
            self.nodes = [startpoint, endpoint]
        else:
            foundS = False
            foundE = False
            for point in self.nodes:
                if self.pointEqual(point, startpoint):
                    foundS = True
                if self.pointEqual(point, endpoint):
                    foundE = True
                if foundS and foundE:
                    break
            if not foundS:
                self.nodes.append(startpoint)
            if not foundE:
                self.nodes.append(endpoint)

        self.wires.append([startpoint, endpoint, radius, number_basis_functions])

    def get_nec_excitation_voltage(self, volt=complex(1.0)):
        """Use with necpp.nec_excitation_voltage():
        necpp.nec_excitation_voltage(nec, *necFile.get_nec_excitation_voltage(a,b))
        """
        return[self.excitation_segment, self.exitation_position, volt.real, volt.imag]

    def get_nec_fr_card(self, frequencyCount=1, frequencyDelta = 0, doLinearRange=True):
        """Use with necpp.nec_fr_card():
        necpp.nec_fr_card(nec, *necFile.get_nec_fr_card())
        """
        in_ifrq = 1
        if doLinearRange:
            in_ifrq = 0
        if frequencyCount > 1 and frequencyDelta == 0:
            raise Exception("Frequency delta is 0")
        in_nfrq	= frequencyCount
        if frequencyDelta > 0:
            in_freq_mhz	= self.freq - frequencyDelta/2 # Go both ways
            in_del_freq	= frequencyDelta
        else:
            in_freq_mhz	= self.freq
            in_del_freq	= 0
        return [in_ifrq, in_nfrq, in_freq_mhz, in_del_freq]

    def get_nec_wire(self, idx):
        """Use with necpp.nec_wire():
        for w in range(len(necFile.wires))):
            necpp.nec_wire(nec, *necFile.get_nec_wire(a))
        """
        if idx >= len(self.wires):
            raise Exception("Wire index out of range")
        w = self.wires[idx]
        startm = self.filePointToMPoint(w[self.WIRE_START])
        endm = self.filePointToMPoint(w[self.WIRE_END])
        xs = startm[0]
        ys = startm[1]
        zs = startm[2]
        xe = endm[0]
        ye = endm[1]
        ze = endm[2]
        radius = self.filePointToMPoint(w[self.WIRE_RADIUS])
        # tag_id	The tag ID.
        # segment_count	The number of segments.
        # xw1	The x coordinate of the wire starting point.
        # yw1	The y coordinate of the wire starting point.
        # zw1	The z coordinate of the wire starting point.
        # xw2	The x coordinate of the wire ending point.
        # yw2	The y coordinate of the wire ending point.
        # zw2	The z coordinate of the wire ending point.
        # rad	The wire radius (meters)
        # rdel	For tapered wires, the. Otherwise set to 1.0
        # rrad	For tapered wires, the. Otherwise set to 1.0 
        return [idx, w[self.WIRE_SEGMENTS], xs, ys, zs, xe, ye, ze, radius, 1.0,1.0]
        
    def replaceNode(self, idx, new_point_in):
        if len(self.nodes) == 0:
            self.calculateUniqueNodes()
        
        if idx >= len(self.nodes):
            raise Exception("Node index out of range")

        new_point = np.array(new_point_in)

        for w in self.wires:
            if self.pointEqual(w[self.WIRE_START], self.nodes[idx]):
                w[self.WIRE_START] = new_point
            if self.pointEqual(w[self.WIRE_END], self.nodes[idx]):
                w[self.WIRE_END] = new_point

        self.calculateUniqueNodes()
    
    def show(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for w in self.wires:
            pointVec = []
            for i in range(3):
                pointVec.append([w[self.WIRE_START][i], w[self.WIRE_END][i]])
                
            ax.plot(pointVec[0], pointVec[1], pointVec[2], color='darkblue')
        plt.show()

    def slowFuzz(self, rtol = 0.0, atol = None):
        if atol == None:
            atol = self.SIGMA
        if len(self.nodes) == 0:
            self.calculateUniqueNodes()
        for idx in range(len(self.nodes)):
            self.replaceNode(idx, np.random.normal(rtol,atol,(3)) + self.nodes[idx])
    def fuzz(self, rtol = 0.0, atol = None):
        if atol == None:
            atol = self.SIGMA
        startlist = []
        endlist = []
        for widx, w in enumerate(self.wires):
            if widx not in startlist:
                startlist.append(widx)
                inStart = w[self.WIRE_START]
                newStart = np.random.normal(rtol,atol,(3)) + inStart
                w[self.WIRE_START] = newStart
                for widx2 in range(widx+1, len(self.wires)):
                    w2 = self.wires[widx2]
                    #if widx2 in startlist:
                    #    continue
                    if self.pointEqual(inStart, w2[self.WIRE_START]):
                        w2[self.WIRE_START] = newStart
                        startlist.append(widx2)
                    elif self.pointEqual(inStart, w2[self.WIRE_END]):
                        w2[self.WIRE_END] = newStart
                        endlist.append(widx2)
            if widx not in endlist:
                endlist.append(widx)
                inEnd = w[self.WIRE_END]
                newEnd = np.random.normal(rtol,atol,(3)) + inEnd
                w[self.WIRE_END] = newEnd
                for widx2 in range(widx+1, len(self.wires)):
                    w2 = self.wires[widx2]
                    #if widx2 in endlist:
                    #    continue
                    if self.pointEqual(inEnd, w2[self.WIRE_START]):
                        w2[self.WIRE_START] = newEnd
                        startlist.append(widx2)
                    elif self.pointEqual(inEnd, w2[self.WIRE_END]):
                        w2[self.WIRE_END] = newEnd
                        endlist.append(widx2)
        self.calculateUniqueNodes()

    def calculateUniqueNodes(self):
        if len(self.wires) == 0:
            raise Exception("No wires added to file")
        self.nodes = [self.wires[0][self.WIRE_START]] # Start somewhere

        for w in self.wires:
            foundS = False
            foundE = False
            for points in self.nodes:
                if self.pointEqual(points, w[self.WIRE_START]):
                    foundS = True
                if self.pointEqual(points, w[self.WIRE_END]):
                    foundE = True
                if foundS and foundE:
                    break
            if not foundS:
                self.nodes.append(w[self.WIRE_START])
            if not foundE:
                self.nodes.append(w[self.WIRE_END])

    def multiplyWire(self, wire, new_elements):
        new_segments = max(int(wire[self.WIRE_SEGMENTS]/new_elements),1)
        supportingVector = wire[self.WIRE_END] - wire[self.WIRE_START]
        supportingVector /= new_elements
        retVal = []
        currentVector = wire[self.WIRE_START]
        radiusFactor = max(0.25, 1/new_elements)
        for i in range(new_elements):
            nextVector = currentVector + supportingVector
            retVal.append([currentVector, nextVector, wire[self.WIRE_RADIUS]*radiusFactor, new_segments])
            currentVector = nextVector
        return retVal

    def extendWiresTo(self, number_of_wires):
        """Splits existing wires until we have number_of_wires in the file
        """
        import random
        if len(self.wires) == number_of_wires:
            return
        extraArray = []
        if number_of_wires % len(self.wires) != 0:
            remainder = number_of_wires % len(self.wires)
            for i in range(remainder):
                randVal = random.randint(0, len(self.wires)-1)
                while randVal in extraArray:
                    randVal = (randVal+1) % len(self.wires)
                extraArray.append(randVal)
        #print(extraArray)
        newWires = []
        newExitationSegment = 0
        newExitationPosition = 0
        for idx, wire in enumerate(self.wires):
            per_wire_factor = int(number_of_wires / len(self.wires))
            if idx in extraArray:
                per_wire_factor += 1
            addedWires = self.multiplyWire(wire, per_wire_factor)
            if idx == self.excitation_segment:
                pos = int( ( self.exitation_position / wire[self.WIRE_SEGMENTS] ) *len(addedWires) )
                newExitationSegment = len(newWires) + pos
                newExitationPosition = 1 + int(addedWires[pos][self.WIRE_SEGMENTS]/2)
            newWires += addedWires

        self.wires = newWires
        self.wire_segments = len(self.wires)
        self.exitation_position = newExitationPosition
        self.excitation_segment = newExitationSegment

    def addComment(self, comment):
        self.comment.append(comment)

    def getLabels2(self):
        # Example comment:
        # Labels: Max Gain, Front-to-Back Ratio, FBR dB, Impedance, Actual Center Frequency, Length, Greatest extent
        # Values: 2.988, 4.723, 6.742, (33.50568744555414+10.483648936682295j), 1944.908, 14.680, 2.417
        #                              (-255.42050794497177-3271.866970443891j)
        impedance = 0
        for c in self.comment:
            if "j)" in c:
                carvedImpedance = "("+ c.split("(")[1].split("j)")[0] +"j)"
                impedance = complex(carvedImpedance)
        absImpedance = abs(impedance)
        return [self.maxGain, self.frontBackRatio, absImpedance, self.freqScore, self.antennaLengthScore, self.greatestExtentScore]


    def normalize_wire(self, wire):
        retVal = wire
        retVal[self.WIRE_RADIUS] /= self.MAX_WIRE_RADIUS
        retVal[self.WIRE_SEGMENTS] /= self.MAX_SEGMENTS
        for i in range(3):
            retVal[self.WIRE_START][i] /= self.MAX_VECTOR_POINT
            retVal[self.WIRE_END][i] /= self.MAX_VECTOR_POINT
        return retVal

    def denormalize_wire(self, wire):
        retVal = wire
        retVal[self.WIRE_RADIUS] *= self.MAX_WIRE_RADIUS
        retVal[self.WIRE_SEGMENTS] = int(retVal[self.WIRE_SEGMENTS]*self.MAX_SEGMENTS)
        for i in range(3):
            retVal[self.WIRE_START][i] *= self.MAX_VECTOR_POINT
            retVal[self.WIRE_END][i] *= self.MAX_VECTOR_POINT
        return retVal

    def normalize_labels(self):
        maxGain, frontBackRatio, absImpedance, freqScore, antennaLengthScore, greatestExtentScore = self.getLabels2()
        return [maxGain/self.MAX_GAIN, frontBackRatio/self.MAX_FBR, absImpedance/self.MAX_IMPEDANCE, freqScore, antennaLengthScore, greatestExtentScore]

    def normalize(self):
        retVec = self.normalize_labels()
        retVec.append(self.excitation_segment / self.wire_segments)
        for idx, wire in enumerate(self.wires):
            # Append "flattened" wire
            normWire = self.normalize_wire(wire)
            for i in range(3):
                retVec.append(normWire[self.WIRE_START][i])
            for i in range(3):
                retVec.append(normWire[self.WIRE_END][i])
            retVec.append(normWire[self.WIRE_RADIUS])
            retVec.append(normWire[self.WIRE_SEGMENTS])
        return retVec

    def denormalize(self, vec, freq, segments):
        self.maxGain, self.frontBackRatio, absImpedance, self.freqScore, self.antennaLengthScore, self.greatestExtentScore = vec[:6]
        self.maxGain *= self.MAX_GAIN
        self.frontBackRatio *= self.MAX_FBR
        absImpedance *= self.MAX_IMPEDANCE
        fakeImpedance = ((absImpedance**2) /2 )**0.5
        self.addComment("(%.2f+%.2fj)" % (fakeImpedance, fakeImpedance))
        self.freq = freq
        self.wavelength = self.C / (self.freq*1000000) # MHz to Hz
        self.FLOAT_EPSILON = self.convertMToRelative(self.FLOAT_EPSILON)
        self.wire_segments = segments
        self.excitation_segment = int(vec[6] * self.wire_segments)
        for i in range(7, len(vec), 8):
            wire = []
            wire.append(np.array([vec[i], vec[i+1], vec[i+2]]))
            wire.append(np.array([vec[i+3], vec[i+4], vec[i+5]]))
            wire.append(vec[i+6])
            wire.append(vec[i+7])
            self.wires.append(self.denormalize_wire(wire))
        # always set excitation position to the middle of the wire
        self.exitation_position = 1 + int(self.wires[self.excitation_segment][self.WIRE_SEGMENTS]/2)
        self.wire_segments = len(self.wires)
        self.Zscore = 0 # ignored
        self.hasLabels = True

    def addLabels(self, maxGain, frontBackRatio, Zscore, freqScore, antennaLengthScore, greatestExtentScore):
        self.hasLabels = True
        self.maxGain = maxGain
        self.frontBackRatio = frontBackRatio
        self.Zscore = Zscore
        self.freqScore = freqScore
        self.antennaLengthScore = antennaLengthScore
        self.greatestExtentScore = greatestExtentScore

    def getLabels(self):
        return [self.maxGain, self.frontBackRatio, self.Zscore, self.freqScore, self.antennaLengthScore, self.greatestExtentScore]

    def toString(self):
        meta = self.metaElement(self.META_VERSION, self.fileversion)
        meta += self.metaElement(self.META_FREQ, self.freq)
        meta += self.metaElement(self.META_EXCITATION_SEGMENT, self.excitation_segment)
        meta += self.metaElement(self.META_EXCITATION_POSITION, self.exitation_position)
        meta += self.metaElement(self.META_WIRE_SEGMENTS, self.wire_segments)

        if self.hasLabels:
            labels = self.labelElement(self.LABEL_MAX_GAIN, self.maxGain)
            labels += self.labelElement(self.LABEL_FRONT_BACK_RATIO, self.frontBackRatio)
            labels += self.labelElement(self.LABEL_ZSCORE, self.Zscore)
            labels += self.labelElement(self.LABEL_FREQ_SCORE, self.freqScore)
            labels += self.labelElement(self.LABEL_ANTENNA_LENGTH_SCORE, self.antennaLengthScore)
            labels += self.labelElement(self.LABEL_GREATEST_EXTENT_SCORE, self.greatestExtentScore)

        f = io.StringIO()
        f.write(self.createIntComment(f"NECFile created by necFile.py Version {self.VERSION}")+"\n")
        for c in self.comment:
            f.write(self.createComment(c)+"\n")
        f.write(self.createIntComment("Meta information:")+"\n")
        f.write(self.createMeta(meta)+"\n")
        if self.hasLabels:
            f.write(self.createIntComment("Labels:")+"\n")
            f.write(self.createLabels(labels)+"\n")
        f.write(self.createIntComment(f" Wavelength ~ {self.wavelength:0.3f} m")+"\n")
        f.write(self.createIntComment(f"START XYZ \tEND XYZ \tRadius \tSegments")+"\n")
        for w in self.wires:
            f.write(self.createWire(w)+"\n")
        return f.getvalue()

    def fromString(self, string):
        lines = string.split("\n")
        for l in lines:
            if l.startswith(self.COMMENT) and l[1] != self.COMMENT:
                self.comment.append(l[len(self.COMMENT)+1:-1])
            if l.startswith(self.META):
                for metaElement in l[len(self.META):].strip().split(self.META_SEP):
                    [element, value] = metaElement.split(self.META_VALUE_SEP)
                    if element == self.META_VERSION:
                        self.fileversion = value
                    if element == self.META_FREQ:
                        self.freq = int(value)
                    if element == self.META_EXCITATION_SEGMENT:
                        self.excitation_segment = int(value)
                    if element == self.META_EXCITATION_POSITION:
                        self.exitation_position = int(value)
                    if element == self.META_WIRE_SEGMENTS:
                        self.wire_segments = int(value)
            if l.startswith(self.LABEL):
                self.hasLabels = True
                for labelElement in l[len(self.LABEL):].strip().split(self.LABEL_SEP):
                    [element, value] = labelElement.split(self.LABEL_VALUE_SEP)
                    if element == self.LABEL_MAX_GAIN:
                        self.maxGain = float(value)
                    if element == self.LABEL_FRONT_BACK_RATIO:
                        self.frontBackRatio = float(value)
                    if element == self.LABEL_ZSCORE:
                        self.Zscore = float(value)
                    if element == self.LABEL_FREQ_SCORE:
                        self.freqScore = float(value)
                    if element == self.LABEL_ANTENNA_LENGTH_SCORE:
                        self.antennaLengthScore = float(value)
                    if element == self.LABEL_GREATEST_EXTENT_SCORE:
                        self.greatestExtentScore = float(value)
            if l.startswith(self.WIRE):
                [startx, starty, startz, endx, endy, endz, radius, segments] = l[len(self.WIRE):].split(self.WIRE_SEP)
                startpoint = [float(startx), float(starty), float(startz)]
                endpoint = [float(endx), float(endy), float(endz)]
                self.addWire(startpoint, endpoint, float(radius), int(segments))

    def writeFile(self, filename = None):
        if filename == None:
            filename = self.filename
        else:
            self.filename = filename
        with open(filename, "w") as f:
            f.write(self.toString())

    def readFile(self, filename = None):
        if filename == None:
            filename = self.filename
        else:
            self.filename = filename
        with open(filename, "r") as f:
            lines = f.read()
        self.fromString(lines)

    def createComment(self, comment):
        return self.COMMENT + " " + comment
    def createIntComment(self, comment):
        return self.COMMENT + self.COMMENT + " " + comment
    def createMeta(self, meta):
        return self.META + meta
    def createLabels(self, label):
        return self.LABEL + label
    def createWire(self, wire):
        [startpoint, endpoint, radius, number_basis_functions] = wire
        return self.WIRE+ self.WIRE_SEP.join([
            f"{startpoint[self.X]}", f"{startpoint[self.Y]}", f"{startpoint[self.Z]}", 
            f"{endpoint[self.X]}", f"{endpoint[self.Y]}", f"{endpoint[self.Z]}", 
            f"{radius}", 
            f"{number_basis_functions}"])
    def metaElement(self, element, value):
        return str(element)+self.META_VALUE_SEP+str(value)+self.META_SEP
    def labelElement(self, element, value):
        return str(element)+self.LABEL_VALUE_SEP+str(value)+self.LABEL_SEP

    