
import numpy as np
import cv2
#from utils.controller import Controller # Original controller implementation by Rafael Sonderegger, opted using simplePid library instead
from simple_pid import PID # Simple PID library by Brett Beauregard
import yaml

def findEdges(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    edgesGreen = cv2.Canny(mask_green, 100, 255)
    edgesGreen[64:78, 44:52] = 0
    edgesGreen[83:-1, :] = 0
    kernel = np.ones((3, 3), np.uint8)
    edgesGreen = cv2.dilate(edgesGreen, kernel, iterations=2)
    edgesGreen = cv2.erode(edgesGreen, kernel, iterations=2)
    return edgesGreen

def findClosestEdgePos(edges, carPos = np.array([70, 48])):
    edgesPos = np.nonzero(edges)
    #Find single closest edge point
    distanceCarToEdges = np.linalg.norm(np.array(carPos)[:, None] - np.array(edgesPos), axis=0)
    closestEdgeIdx = np.argmin(distanceCarToEdges)
    closestEdgePos = np.array([edgesPos[0][closestEdgeIdx], edgesPos[1][closestEdgeIdx]])
    return closestEdgePos

def findTrackVector(edges, closestEdgePos):
    #Find vector describing track direction (tangent to track) using a square around the closest edge point
    squareSize = 3
    squareMiddlePoint = closestEdgePos
    square = edges.copy()[squareMiddlePoint[0] - squareSize: squareMiddlePoint[0] + squareSize + 1,
                                squareMiddlePoint[1] - squareSize: squareMiddlePoint[1] + squareSize + 1]
    edgesIdxSquare = np.nonzero(square)
    pnt1 = np.array([edgesIdxSquare[0][0], edgesIdxSquare[1][0]])
    pnt2 = np.array([edgesIdxSquare[0][-1], edgesIdxSquare[1][-1]])
    vector_track = pnt2 - pnt1
    return vector_track

def calculateTargetPoint(image, widthOfTrack, freq, scale_dist, Amplitude, t):
    # Find edges of track
    edges = findEdges(image) # returns a binary image with edges of track
    closestEdgePos = findClosestEdgePos(edges) # returns the position of the closest edge point to the car
    vector_track = findTrackVector(edges, closestEdgePos) # returns a vector describing the direction of the track
    
    #Make sure the track vector is pointing towards the car direction
    if np.dot(vector_track, np.array([-1, 0])) < 0:
        vector_track = -vector_track

    #Normalized track heading vector and perpendicular vector
    vector_track_normalized = vector_track / np.linalg.norm(vector_track)
    vector_track_perp_normalized = np.array([-vector_track_normalized[1], vector_track_normalized[0]])

    #Make sure that both vectors have reasonable values
    if np.isnan(vector_track_normalized).any() or np.isnan(vector_track_perp_normalized).any():
        return None, None, None, None

    #Check if the vector is pointing towards the inside of the track
    controlPixelPos = closestEdgePos + (vector_track_perp_normalized*3).astype(int)
    controlPixel = image[controlPixelPos[0], controlPixelPos[1]]
    if controlPixel[1] > 200: # Green pixel meaning outside of track
        vector_track_perp_normalized = -vector_track_perp_normalized
    
    #Find the estimated middle point of the track relative to the closest edge point
    estimatedMiddlePoint = (closestEdgePos + vector_track_perp_normalized * widthOfTrack / 2).astype(int)

    # Calculate the next num_points points on the trajectory (sinusoidal curve)
    sin_coeff = Amplitude * np.sin((t+1) * freq * 2 * np.pi)
    #cross product btw track vector and perpendicular vector positive
    sin_vector = (sin_coeff * vector_track_perp_normalized).astype(int)
    if np.cross(vector_track_normalized, vector_track_perp_normalized) < 0:
        sin_vector = -sin_vector
    sin_vector = sin_vector.astype(int)
    dir_vector = vector_track_normalized * scale_dist
    sinusPoints_pos = (estimatedMiddlePoint + dir_vector + sin_vector)
    targetPoint = sinusPoints_pos
    targetPoint = int(targetPoint[0]), int(targetPoint[1]) 

    return targetPoint, estimatedMiddlePoint, vector_track_normalized ,vector_track_perp_normalized


def find_edge_1dStrip(array, direction):
    # Find edge point of a 1D array. 
    # If none is found return -1
    starting_point = int(len(array) // 2)
    idx = -1
    if direction == 'left':
        for i in range(starting_point, -1, -1):
            if array[i] != 0:
                idx = i
                break
    elif direction == 'right':
        for i in range(starting_point, len(array)):
            if array[i] != 0:
                idx = i
                break
    return idx

def find_middle_point(strip_1d):
    # Check if there is edge point for both left and right side of track.
    # If none is found set border of strip as edge point
    idx1 = find_edge_1dStrip(strip_1d, 'left')
    idx2 = find_edge_1dStrip(strip_1d, 'right')

    if idx1 == -1:
        idx1 = 0
    if idx2 == -1:
        idx2 = len(strip_1d) - 1

    idx_middle = int((idx1 + idx2) / 2)
    return idx_middle

def calculateDistAngle(idx_middle_upper, idx_middle_lower, strip_width, strip_height):
    # Calculate distance and angle from middle of the track
    # idx_middle_upper: index of middle point on upper edge of strip
    # idx_middle_lower: index of middle point on lower edge of strip
    # strip_width: width of the strip
    # strip_height: height of the strip
    # return: distance and angle
    
    # Compute distance to middleline
    distance_to_middleline = strip_width // 2 - idx_middle_lower
    # Compute angular error
    upper_lenght_to_target = strip_width // 2 - idx_middle_upper
    angle_to_target = np.arctan(upper_lenght_to_target / strip_height)
    return distance_to_middleline, angle_to_target


def processImage(image):
    # Cropping image down to a strip
    strip_height = 20
    strip_width = 96
    middle_height = 65
    top = int(middle_height - strip_height / 2)
    bottom = int(middle_height + strip_height / 2)
    # Crop the strip from the image
    strip = image[top:bottom, :]

    ## Mask where only edge is retained
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
    imask_green = mask_green>0
    gray_mask = imask_green.astype(np.uint8)
    gray_mask = gray_mask*255
    # Only use two edges of the strip: Upper and lower and find edge points coordinates
    upper_edge = gray_mask[0, :]
    lower_edge = gray_mask[strip_height - 1, :]
    # Get index of middle point on the upper and lower edge
    idx_middle_upper = find_middle_point(upper_edge)
    idx_middle_lower = find_middle_point(lower_edge)

    distance, angle = calculateDistAngle(idx_middle_upper, idx_middle_lower, strip_width, strip_height)
    return distance, angle


def calculateAction(observation , target_velocity):

    # Initialize controllers
    pid_angle = PID(0.5, -0.01, 0.05, setpoint=0)
    pid_distance = PID(0.5, -0.005, 0.05, setpoint=0)
    pid_velocity = PID(0.05, -0.1, 0.2, setpoint=target_velocity)
    
    # Distinguish observation type
    image = observation['image']
    velocity = observation['velocity']

    # Get distance from processed image
    error_dist, error_ang = processImage(image)

    # Get control outputs from PD controllers
    control_ang = pid_angle(error_ang)
    control_dist = pid_distance(error_dist)
    control_vel = pid_velocity(velocity)

    #print("Control outputs: ", control_ang, control_dist, control_vel)
    acc = control_vel
    breaking = 0
    if acc < 0:
        acc = 0
        breaking = -control_vel
    
    # Calculate and return final action
    action = [control_ang , acc, breaking]
    return action

def action_sinusoidalTrajectory(t, freq, observation, Amplitude, target_velocity):
    # Observations are the following:
    image = observation['image']
    velocity = observation['velocity']

    # Environment constants
    carPos = np.array([70, 48]) # Position of the car in the image (pixel coordinates)
    widthOfTrack = 20 # Approx width of the track in pixels

    # Initialize controllers
    pid_angle = PID(0.5, -0.2, 0.0, setpoint=0)
    pid_velocity = PID(0.05, 0.1, 0.1, setpoint=target_velocity)

    # Find the next target point of sinusoidal trajectory
    scale_dist = 10 # This scales the vertical distance of the next target point from tip of car
    targetPoint, estimatedMiddlePoint, vector_track_normalized, vector_track_perp_normalized = calculateTargetPoint(image, widthOfTrack, freq, scale_dist , Amplitude, t)
    
    if targetPoint is None:
        action = [0,0,0] # If unreasonable values where found for the target point, keep the previous action. This avoids an edge case error
        return action

    # Calculate the angle to the target point
    error = targetPoint - carPos
    carVector = np.array([-1, 0])
    

    angle = np.arccos(np.dot(error, carVector) / (np.linalg.norm(error) * np.linalg.norm(carVector)))
    #Check if the angle is positive or negative -> negative full left turn, positive full right turn        
    if error[1] > 0:
        angle = -angle        
    steeringAngle = pid_angle(angle)
    # Calculate the acceleration or if negative, the breaking
    acc = pid_velocity(velocity)
    breaking = 0
    if acc < 0:
        breaking = -acc
        acc = 0
    action = [steeringAngle, acc, breaking]

    #print("Actions: ", action)
    return action
    
def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams