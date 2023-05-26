#!/usr/bin/python3.10

import setup_path 
import airsim
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R 
import pprint
import asyncio
import os
from struct import pack


# connect to the AirSim simulator
client = airsim.VehicleClient()
client.confirmConnection()

# set camera name and image type to request images and detections
camera_name = "0"
image_type = airsim.ImageType.Scene

# set detection radius in [cm]
client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
# add desired object name to detect in wild card/regex format
# client.simAddDetectionFilterMeshName(camera_name, image_type, "Cylinder*") 
client.simAddDetectionFilterMeshName(camera_name, image_type, "SM_TrafficBarrel*")

obstacle_index = 1

obstacles = {}

class EchoClientProtocol(asyncio.Protocol):
    outside_transport = None
    def __init__(self, on_con_lost):
        self.initial_connection = bytearray()
        self.initial_connection.append(0x79)
        self.initial_connection.append(0x95)
        self.admission_connection = bytearray()
        self.admission_connection.append(0x79)
        self.admission_connection.append(0x96)
        self.connection_OK = bytearray()
        self.connection_OK.append(0x79)
        self.connection_OK.append(0x97)
        
        self.on_con_lost = on_con_lost
        self.is_connected = False

    def connection_made(self, transport):
        self.transport = transport
        EchoClientProtocol.outside_transport = transport
        transport.write(self.initial_connection)
        print('the request sent to establish the connection')

    def data_received(self, data):
        if not self.is_connected and data[0] == self.admission_connection[0] and data[1] == self.admission_connection[1]:
            self.is_connected = True
            self.transport.write(self.connection_OK)
            print('the connection has established!')
            
        #print('Data received: {!r}'.format(data.decode()))

    def connection_lost(self, exc):
        print('The server closed the connection')
        self.on_con_lost.set_result(True)


async def get_data(image_recieved, client, camera_name, image_type):
    
    image_recieved.set_result(client.simGetImage(camera_name, image_type))


def create_indentity_mat():
    M = np.zeros((4, 4))
    M[0, 0] = 1
    M[1, 1] = 1
    M[2, 2] = 1
    M[3, 3] = 1
    return M


def convertMatrixToRightHand(mat4x4, isScaleCorrection = True):
    
    Mi = create_indentity_mat()
    Mi[0,0] = -1.0
    Mi[1,1] = -1.0
    
    if isScaleCorrection:
        mat4x4[0, 0] *= 0.5
        mat4x4[1, 1] *= 0.5
        mat4x4[2, 2] *= 0.5 
        
    return Mi @ mat4x4 @ Mi;

    

async def test_task():
    global obstacle_index
    global obstacles
    loop = asyncio.get_event_loop()
    width = 100.0
    height = 100.0
    step = 1.0
    while True:
        png = np.zeros((500, 500, 3), dtype=np.uint8)
        await asyncio.sleep(0)
        
        image_recieved = loop.create_future()
        try:
            await get_data(image_recieved, client, camera_name, image_type)
            await image_recieved
            rawImage = image_recieved.result()
            
        except Exception as e:
            print(str(e))
        if not rawImage:
            continue
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        cylinders = client.simGetDetections(camera_name, image_type)
        disappeared_obstacles = []
        for obstacle in obstacles.keys():
            if not obstacle in [cylinder.name  for cylinder in cylinders]:
                disappeared_obstacles.append(obstacle)
        
        for obstacle_name in disappeared_obstacles:
            del obstacles[obstacle_name]
        sending_obstacles = {}
        sent_msg = bytearray()
        M = np.zeros((4, 4))
        M[0, 0] = 1
        M[1, 1] = 1
        M[2, 2] = 1
        M[3, 3] = 1
        M = convertMatrixToRightHand(M, False)
        M_listed = M.reshape((16,)).tolist()
        
        try:
            sent_msg.append(0x44)
            sent_msg.append(0x47)
            sent_msg += pack("f", width)
            sent_msg += pack("f", height)
            sent_msg += pack("f", step)
            sent_msg += pack("f", width / step / 2 + (height / step - 1) * (width / step))
            sent_msg += pack("fff", 50.0, 0, 80)
            
            sent_msg += pack("f", 0)
            sent_msg += pack("Q", 0)
            sent_msg += pack("Q", 0)   
            sent_msg += pack("16f", *M_listed)   
        except Exception as e:
            print(str(e))
            
            
        if cylinders:
            for cylinder in cylinders:
                s = pprint.pformat(cylinder)
                M = np.zeros((4, 4))
                # print("Object: %s" % s)
                
                #print("size = ", cylinder.box3D.max.x_val - cylinder.box3D.min.x_val,
                #                 cylinder.box3D.max.y_val - cylinder.box3D.min.y_val,
                #                 cylinder.box3D.max.z_val - cylinder.box3D.min.z_val)
                
                position = np.array([ cylinder.relative_pose.position.y_val,
                                    cylinder.relative_pose.position.z_val,
                                    cylinder.relative_pose.position.x_val])
                relativa_pose = np.array([cylinder.relative_pose.position.x_val, cylinder.relative_pose.position.y_val, cylinder.relative_pose.position.z_val])
                max_box = np.array([cylinder.box3D.max.x_val, cylinder.box3D.max.y_val, cylinder.box3D.max.z_val])
                min_box = np.array([cylinder.box3D.min.x_val, cylinder.box3D.min.y_val, cylinder.box3D.min.z_val])
                r = R.from_quat([cylinder.relative_pose.orientation.x_val,
                                cylinder.relative_pose.orientation.y_val, 
                                cylinder.relative_pose.orientation.z_val,
                                cylinder.relative_pose.orientation.w_val])
            
                #relativa_pose = r.apply(relativa_pose)
                
                max_box -= relativa_pose
                min_box -= relativa_pose
                #print(r.as_matrix())
                vectors = r.apply([max_box, min_box], inverse=False)
                scale = vectors[0] - vectors[1]
                scale = abs(scale)
                #print(scale)
                
            
                print(position)
                
                M[:3, :3] = r.as_matrix()
                M[3, :] = [0, 0, 0, 1]
                M[:3, 3] = position
                M[0, 0] *= scale[0]
                M[1, 1] *= scale[1]
                M[2, 2] *= scale[2]
                #print(M)
                M = convertMatrixToRightHand(M)
                M_listed = M.reshape((16,)).tolist()
                #print(M_listed)
                try:
                    if not cylinder.name in obstacles.keys():
                        obstacles[cylinder.name] = obstacle_index
                        obstacle_index += 1 
                except Exception as e:
                    print('an exception happened' + str(e))
                sending_obstacles[obstacles[cylinder.name]] = M_listed
                cv2.rectangle(png,(int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val)),(int(cylinder.box2D.max.x_val),int(cylinder.box2D.max.y_val)),(255,0,0),2)
                cv2.putText(png, cylinder.name, (int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))
                #print(obstacles)
                sent_msg += pack('Q', obstacles[cylinder.name])
                sent_msg += pack('16f', *M_listed)
        
        sent_msg[34:42] = pack("Q", len(sending_obstacles) + 1)
        #sent_msg[34:42] = pack("Q", 1)
        if EchoClientProtocol.outside_transport is not None:
            EchoClientProtocol.outside_transport.write(sent_msg)
        cv2.imshow("AirSim", png)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            client.simClearDetectionMeshNames(camera_name, image_type)
        elif cv2.waitKey(1) & 0xFF == ord('a'):
            client.simAddDetectionFilterMeshName(camera_name, image_type, "Cylinder*")
    cv2.destroyAllWindows() 
        
    
    
async def network():
    # Get a reference to the event loop as we plan to use
    # low-level APIs.
    while True:
        loop = asyncio.get_running_loop()

        on_con_lost = loop.create_future()

        try:
            transport, protocol = await loop.create_connection(
                    lambda: EchoClientProtocol(on_con_lost),
                    'localhost', 15556)
        except:
            print('it has just tried to create a new connection')
            on_con_lost.set_result(False)
            #loop.run_forever()
            # Wait until the protocol signals that the connection
            # is lost and close the transport.
        print("after the connection established")
            
        try:
            await on_con_lost
        finally:
            print("the connection closed!")
            if on_con_lost.result():
                transport.close()
            await asyncio.sleep(2)
        
async def main():
    # create tasks for the two loops
    task1 = asyncio.create_task(network())
    task2 = asyncio.create_task(test_task())

    # wait for both tasks to complete
    await asyncio.gather(task1, task2)

asyncio.run(main())

