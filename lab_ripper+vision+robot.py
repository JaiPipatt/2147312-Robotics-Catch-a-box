import socket , time
import binascii

def main():
    g_ip="10.10.0.60" #replace by the IP address of the UR robot
    g_port=63352      #PORT used by robotiq gripper -> use this port only to connect with gripper

    robot_ip = '10.10.0.60' #replace by the IP address of the UR robot
    port = 30003

    vision_ip = '10.10.1.60'
    vision_port = 2025

    g = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    g.connect((g_ip, g_port))
    r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    r.connect((robot_ip, port))
    v = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    v.connect((vision_ip, vision_port))

    # init gripper
    g.send(b'GET ACT\n')
    g_recv = str(g.recv(10), 'UTF-8')
    if '1' in g_recv :
        print ('Gripper Activated')
    print ('get ACT  == ' + g_recv)
    g.send(b'GET POS\n')
    g_recv = str(g.recv(10), 'UTF-8')
    if g_recv :
        g.send(b'SET ACT 1\n')
        g_recv = str(g.recv(255), 'UTF-8')
        print (g_recv)
        time.sleep(3)
        g.send(b'SET GTO 1\n')
        g.send(b'SET SPE 255\n')
        g.send(b'SET FOR 255\n')
    i=0

    # get object coordinates from vision system
    v.send(b'cap!')
    coor = v.recv(255)
    coor = coor.decode("utf-8")

    # move robot based on coordinates
    print(type(coor))
    x = str(float(coor.split(',')[0])/1000)
    y = str(float(coor.split(',')[1][0:-2])/1000)
    print(x,y)
    command = 'movel(pose_add(get_actual_tcp_pose(), p['+x+ ','+'-'+y+', 0, 0, 0, 0]),1,1,2,0)\n'
    r.send(command.encode("utf-8"))
    time.sleep(2)
    r.send(b'movel(pose_add(get_actual_tcp_pose(), p[0 , -0.09, 0, 0, 0, 0]),1,1,2,0)\n')
    time.sleep(2)
    r.send(b'movel(pose_add(get_actual_tcp_pose(), p[-0.02 , 0, 0, 0, 0, 0]),1,1,2,0)\n')
    time.sleep(2)
    g.send(b'SET POS 0\n')
    r.send(b'movel(pose_add(get_actual_tcp_pose(), p[0 , 0, -0.25, 0, 0, 0]),1,1,2,0)\n')
    time.sleep(2)
    g.send(b'SET POS 255\n')
    time.sleep(2)
    r.send(b'movel(pose_add(get_actual_tcp_pose(), p[0 , 0, 0.25, 0, 0, 0]),1,1,2,0)\n')
    time.sleep(2)
        
if __name__ == '__main__':
    main()