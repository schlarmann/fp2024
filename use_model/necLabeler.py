import necpp
import necFile
import numpy as np

class necLabeler:

    CENTER_FREQ = 2450
    FREQ_RANGE = CENTER_FREQ
    FREQ_COUNT = 10

    THETA_START_ANGLE = 90
    THETA_END_ANGLE = 90
    THETA_POINTS = 1
    PHI_START_ANGLE = 0
    PHI_END_ANGLE = 360
    PHI_POINTS = 720

    MAX_ANTENNA_LENGTH = 20 # Wavelengths
    MAX_ANTENNA_EXTENT = 10 # Wavelengths

    def __init__(self):
        self.nec = necpp.nec_create()
        self.gains = []
        self.angles = []

    def reset(self):
        necpp.nec_delete(self.nec)
        self.nec = necpp.nec_create()

    def clear(self):
        self.gains = []
        self.angles = []
        necpp.nec_delete(self.nec)

    def __del__(self):
        necpp.nec_delete(self.nec)


    def handle_nec(self, result):
        if (result != 0):
            print(necpp.nec_error_message())
            raise Exception(necpp.nec_error_message())

    def phiIdxToAngle(self, idx):
        return self.PHI_START_ANGLE + (self.PHI_END_ANGLE - self.PHI_START_ANGLE) * idx / self.PHI_POINTS
    def phiAngleToIdx(self, angle):
        return int((angle - self.PHI_START_ANGLE) / (self.PHI_END_ANGLE - self.PHI_START_ANGLE) * self.PHI_POINTS)
    def thetaIdxToAngle(self, idx):
        return self.THETA_START_ANGLE + (self.THETA_END_ANGLE - self.THETA_START_ANGLE) * idx / self.THETA_POINTS
    def thetaAngleToIdx(self, angle):
        return int((angle - self.THETA_START_ANGLE) / (self.THETA_END_ANGLE - self.THETA_START_ANGLE) * self.THETA_POINTS)

    def showGain(self):
        import matplotlib.pyplot as plt

        theta = 2 * np.pi * np.array(self.angles)/360
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta, self.gains)
        ax.set_rmax(self.maxGain+0.5)
        ax.set_rmin(self.minGain-0.5)
        #ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)

        ax.set_title("Gain over Phi (0°=+X-Axis, 90°=+Y-Axis)", va='bottom')
        plt.show()

    def getMaxGain(self, freq=0):
        maxGain = -999
        maxTheta = 0
        maxPhi = 0
        self.gains = []
        for theta in range(self.THETA_POINTS):
            for phi in range(self.PHI_POINTS):
                gain = necpp.nec_gain(self.nec, freq, theta, phi)
                self.gains.append(gain)
                self.angles.append(self.phiIdxToAngle(phi))
                if gain > maxGain:
                    maxGain = gain
                    maxTheta = theta
                    maxPhi = phi
        return maxGain, maxTheta, maxPhi

    def getOppoGain(self, maxTheta, maxPhi, freq=0):
        oppoTheta = int(maxTheta + self.THETA_POINTS / 2) % self.THETA_POINTS
        oppoPhi = int(maxPhi + self.PHI_POINTS / 2) % self.PHI_POINTS
        print(f"Opposite Position: Theta {self.thetaIdxToAngle(oppoTheta)}, Phi {self.phiIdxToAngle(oppoPhi)},")
        return necpp.nec_gain(self.nec, freq, oppoTheta, oppoPhi)

    def freqIdxToFreq(self, idx):
        return self.CENTER_FREQ - self.FREQ_RANGE / 2 + self.FREQ_RANGE * idx / self.FREQ_COUNT

    def gainToPower(self, gain):
        return 10 ** (gain / 10)

    def labelFile(self, in_nf: necFile.necFile):
        nf = in_nf.deepcopy()
        nf.freq = self.CENTER_FREQ

        # Setup geometry
        for idx in range(len(nf.wires)):
            self.handle_nec( necpp.nec_wire(self.nec, *nf.get_nec_wire(idx)))
        self.handle_nec(necpp.nec_geometry_complete(self.nec, 0)) # No groundplane

        # Setup simulation parameters
        freqIdx = int(self.FREQ_COUNT / 2)
        self.handle_nec(necpp.nec_fr_card(self.nec, 0,self.FREQ_COUNT, self.freqIdxToFreq(0), self.freqIdxToFreq(1)-self.freqIdxToFreq(0)))
        self.handle_nec(necpp.nec_excitation_voltage(self.nec, *nf.get_nec_excitation_voltage(1.0))) 

        # Simulate
        # Theta: From Z to PHI
        # Phi: Angle From X to Y
        # --> Going around the Z axis, starting in X direction: Theta always 90, Phi 0-360
        calc_mode = 0 # 0 - normal mode. Space-wave fields are computed. An infinite ground plane is included if it has been specified previously on a GN card; otherwise, the antenna is in free space. 
        output_format = 0 # 0 major axis, minor axis and total gain printed. 
        normalization = 5 # 5 total gain normalized.
        D = 0 # 0 power gain. / 1 directive gain.
        A = 0 # 0 no averaging. / 1 average gain computed.

        n_theta = self.THETA_POINTS # The number of theta angles. 
        n_phi = self.PHI_POINTS # The number of phi angles.
        theta0 = self.THETA_START_ANGLE # Initial theta angle in degrees
        phi0 = self.PHI_START_ANGLE # Initial phi angle in degrees
        # Increment for theta in degrees
        if self.THETA_START_ANGLE == self.THETA_END_ANGLE or self.THETA_POINTS == 1:
            delta_theta = 0
        else:
            delta_theta = (self.THETA_END_ANGLE-self.THETA_START_ANGLE) /self.THETA_POINTS
        # Increment for phi in degrees
        if self.PHI_START_ANGLE == self.PHI_END_ANGLE or self.PHI_POINTS == 1:
            delta_phi = 0
        else: 
            delta_phi = (self.PHI_END_ANGLE-self.PHI_START_ANGLE) / self.PHI_POINTS

        radial_distance = 0 # Radial distance (R) of field point from the origin in meters. radial_distance is optional. If it is zero, the radiated electric field will have the factor exp(-jkR)/R omitted
        gain_norm = 0 # Gain normalization factor if normalization has been requested
        
        self.handle_nec(necpp.nec_rp_card(self.nec, calc_mode, n_theta, n_phi, output_format,normalization, D,A, theta0,phi0, delta_theta, delta_phi, radial_distance, gain_norm))

        self.maxGain, maxTheta, maxPhi = self.getMaxGain(freq=freqIdx)
        #print(f"Max Position: Theta {self.thetaIdxToAngle(maxTheta)}, Phi {self.phiIdxToAngle(maxPhi)}")
        oppoGain = self.getOppoGain(maxTheta, maxPhi, freq=freqIdx)
        #print(f"Opposite Gain: {oppoGain}")
        frontBackRatio = self.maxGain - oppoGain
        frontBackRatioLin = self.gainToPower(frontBackRatio)
        self.minGain = necpp.nec_gain_min(self.nec,freqIdx)
        impedance = complex(necpp.nec_impedance_real(self.nec,freqIdx), necpp.nec_impedance_imag(self.nec,freqIdx))

        # Calculate a score for the impedance between 0 and 1, where 1 is a perfect match (50 Ohms, phase does not matter)
        # and 0 is a short circuit Z = 0 or open circuit Z = Inf
        Z = abs(impedance)
        Zideal = 50
        if Z < 50:
            Zscore = 1 - abs(Z-Zideal)/Zideal
        else:
            Zscore = abs(Zideal/Z)
        #print(f"Z: {Z} Ohm; Zscore: {Zscore}")

        # Calculate the actual center frequency
        center = self.maxGain
        center_freq = int(self.FREQ_COUNT/2)
        for i in range(self.FREQ_COUNT):
            freqGain = necpp.nec_gain_max(self.nec, i)
            if(freqGain > center):
                center = freqGain
                center_freq = i
        #print(f"Center Frequency: {self.freqIdxToFreq(center_freq)} MHz; Gain: {freqGain}")
        
        # Make a linear interpolation between the center frequency and the two closest frequencies
        try:
            center = necpp.nec_gain_max(self.nec, center_freq)
        except:
            print(f"Error at center frequency: {center_freq}")
            raise
        higher = -999
        lower = -999
        if center_freq != self.FREQ_COUNT:        
            higher = necpp.nec_gain_max(self.nec, center_freq+1)
        if center_freq != 0:
            lower = necpp.nec_gain_max(self.nec, center_freq-1)
        fractionHigher = (self.gainToPower(center)+self.gainToPower(higher)) / (4*self.gainToPower(center))
        fractionLower = (self.gainToPower(center)+self.gainToPower(lower)) / (4*self.gainToPower(center))
        fraction = (fractionHigher - fractionLower)
        center_freq += fraction
        dist = abs(center_freq - freqIdx)
        # Calculate a frequency score between 0 and 1, where 1 is the best possible score (1: Center = 2450 MHz)
        freqScore = 1 - 2*dist / self.FREQ_COUNT
        #print(f"Adjusted Center Frequency: {self.freqIdxToFreq(center_freq)} MHz")

        antennaLength = nf.totalWireLength()
        #print(f"Total Wire Length: {antennaLength} lambda")
        if antennaLength >= self.MAX_ANTENNA_LENGTH:
            antennaLengthScore = 0
        else:
            antennaLengthScore = (self.MAX_ANTENNA_LENGTH - antennaLength) / self.MAX_ANTENNA_LENGTH

        greatestExtent = nf.maxDistance()
        #print(f"Greatest distance from Origin: {greatestExtent} lambda")
        if greatestExtent >= self.MAX_ANTENNA_EXTENT:
            greatestExtentScore = 0
        else:
            greatestExtentScore = (self.MAX_ANTENNA_EXTENT - greatestExtent) / self.MAX_ANTENNA_EXTENT

        if self.maxGain > 25:
            # Got a good antenna
            print(f"Good Antenna!: Max Gain, Front-to-Back Ratio, FBR dB, Impedance, Actual Center Frequency, Length, Greatest extent")
            print(f"Values: {self.maxGain:0.3f}, {frontBackRatioLin:0.3f}, {frontBackRatio:0.3f}, {impedance}, {self.freqIdxToFreq(center_freq):0.3f}, {antennaLength:0.3f}, {greatestExtent:0.3f}")
        
        #print(f"Max Gain: {self.maxGain}; Front-to-Back Ratio: {frontBackRatio:0.3f} dB / {frontBackRatioLin:0.3f}; Min Gain: {self.minGain}; Impedance: {impedance}; Frequency Score: {freqScore}")
        out_nf = in_nf.deepcopy()
        out_nf.addComment(f"Labels: Max Gain, Front-to-Back Ratio, FBR dB, Impedance, Actual Center Frequency, Length, Greatest extent")
        out_nf.addComment(f"Values: {self.maxGain:0.3f}, {frontBackRatioLin:0.3f}, {frontBackRatio:0.3f}, {impedance}, {self.freqIdxToFreq(center_freq):0.3f}, {antennaLength:0.3f}, {greatestExtent:0.3f}")
        
        out_nf.addLabels(self.maxGain, frontBackRatio, Zscore, freqScore, antennaLengthScore, greatestExtentScore)
        return out_nf
