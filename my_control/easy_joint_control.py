from lerobot.robots.so_follower import SO100Follower,SO101FollowerConfig
import time
import numpy as np

def move_linear_smooth_arry(robot,obs,start_pos,target_pos,duration=2.0,dt=0.02):
    steps = int(duration / dt)
    home = {
        name: value
        for name,value in obs.items()
        if name.endswith(".pos")
    }
    joint_names = list(home.keys())

    for i in range(steps + 1):
        alpha = i / steps
        action = {}
        q_cmd = start_pos + alpha * (target_pos - start_pos)
        action = dict(zip(joint_names,q_cmd))
        robot.send_action(action)
        time.sleep(dt)

def main():
    config = SO101FollowerConfig(
        port = "/dev/ttyACM0",
        id = "my_awesome_follower_arm",
        use_degrees = True,
    )

    robot = SO100Follower(config)

    try:
        robot.connect()
        print("=== Connect success === \n")
        obs = robot.get_observation()

        home = {
            name: value
            for name,value in obs.items()
            if name.endswith(".pos")
        }

        # print the home pos
        for name,value in home.items():  # transform to key: value style
            # if name.endswith(".pos"): # if the end name ==  .pos ,choose the key: value
                print(f"{name:5s}: {value:8.2f}") 
                time.sleep(0.1)


        #  2. 创建目标位置
        q_home = np.array(list(home.values())) #now the q = [0 - 4] and a gripper
        q_target = q_home.copy()
        q_target = [-46.99,-53.01,38.68,74.81,10.51,45.000]

        # 3. Home -> Target
        print("\nMoving to target...")
        move_linear_smooth_arry(
            robot,
            obs,
            q_home,
            q_target,
        )
        # 4. 停顿 3 秒
        # -----------------------------
        print("Target reached. Wait 3 seconds...")

        time.sleep(3.0)

        print("Return home ")

        move_linear_smooth_arry(
            robot,
            obs,
            q_target,
            q_home,
        )

    except KeyboardInterrupt:
        print("Key Stop")

    except Exception as e:
        print("ERROR: ",e)

    finally:
        if robot.is_connected:
            robot.disconnect()    

if __name__ == "__main__":
    main()