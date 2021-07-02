# Kinematics Calculations (units: meters)
# sources: http://doc.aldebaran.com/2-8/family/nao_technical/links_naov6.html
# Note: NAO v5 and v6 measurements are identical

"""X Axis"""
UpperArmLength = 105. / 1000
LowerArmLength = 55.95 / 1000
HandOffsetX = 57.75 / 1000
WristYaw_Finger11 = 69.07 / 1000
Finger11_Finger12 = 14.36 / 1000
Finger12_Finger13 = 14.36 / 1000
Finger13_FingerTip = 14.36 / 1000

shoulder_to_hand_x = UpperArmLength + LowerArmLength + HandOffsetX
# shoulder_to_hand_x = 0.2187 meters (hand kinematics result)
shoulder_to_inner_finger_x = UpperArmLength + LowerArmLength + WristYaw_Finger11 + Finger11_Finger12 + Finger12_Finger13 + Finger13_FingerTip
# shoulder_to_inner_finger_x = 0.2731 meters (finger kinematics result)

"""Y Axis"""
ShoulderOffsetY = 98. / 1000
ElbowOffsetY = 15. / 1000
Finger2OffsetY = -11.57 / 1000

torso_to_hand_y = ShoulderOffsetY + ElbowOffsetY
# torso_to_hand_y = 0.113 meters (hand kinematics result)
torso_to_finger2_y = torso_to_hand_y + Finger2OffsetY
# torso_to_finger2_y = 0.10143 meters (finger kinematics result)

"""Z Axis"""
HipOffsetZ = 85. / 1000
ThighLength = 100. / 1000
TibiaLength = 102.90 / 1000
FootHeight = 45.19 / 1000
ShoulderOffsetZ = 100. / 1000
HandOffsetZ = 12.31 / 1000
Finger2OffsetZ = -3.04 / 1000

foot_to_torso_z = FootHeight + TibiaLength + ThighLength + HipOffsetZ
# foot_to_torso_z = 0.33309 meters (zero point Z)
foot_to_shoulder_z = foot_to_torso_z + ShoulderOffsetZ
# foot_to_shoulder_z = 0.43309 meters
foot_to_hand_z = foot_to_shoulder_z - HandOffsetZ
# foot_to_hand_z = 0.42078 meters
torso_to_shoulder_z = ShoulderOffsetZ
# torso_to_shoulder_z = 0.100 meters
torso_to_hand_z = torso_to_shoulder_z - HandOffsetZ
# torso_to_hand_z = 0.08769 meters (hand kinematics result)
torso_to_inner_finger_z = torso_to_shoulder_z + Finger2OffsetZ
# torso_to_inner_finger_z = 0.09696 meters (finger kinematics result)

""" 
NAO coordinate system 
positive x-axis: front of nao
positive y-axis: left of nao
positive z-axis: up from ground
"""


# NAO Zero Point
# [x, y, z] = [0.0, 0.0, 0.0]

# NAO RHand Position
# [x, y, z] = [0.2187, -0.113, 0.08769]
# NAO LHand Position
# [x, y, z] = [0.2187, 0.113, 0.08769]

# NAO RInnerFinger Position
# [x, y, z] = [0.2731, -0.10143, 0.09696]
# NAO LInnerFinger Position
# [x, y, z] = [0.2731, 0.10143, 0.09696]


""" 
webots coordinate system 
positive x-axis: front of nao
positive y-axis: up from ground
positive z-axis: left of nao
"""
# NAO Zero Point
# [x, y, z] = [0.0, 0.33309, 0.0]

# NAO RHand Position
# [x, y, z] = [0.2187, 0.4208, 0.1130]
# NAO LHand Position
# [x, y, z] = [0.2187, 0.4208, -0.1130]

# NAO RInnerFinger Position
# [x, y, z] = [0.2731, 0.4300, 0.1014]
# NAO LInnerFinger Position
# [x, y, z] = [0.2731, 0.4300, -0.1014]


"""Notes"""
# The results of calling f_left_hand_h25([0. 0. 0. 0. 0.]) is [0.2187  0.1130    0.08769   -1.504e-49    -0.    -0.  ].
# The results of calling f_right_hand_h25([0. 0. 0. 0. 0.]) is [0.2187  -0.1130    0.08769   -1.504e-49    -0.    -0.  ].