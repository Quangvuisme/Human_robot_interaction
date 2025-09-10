# === Falcon Tube Pick-and-Place with Interbotix RX150 Robot ===
# Author: [Quang Vu]
# Date: 2025-05
# Description: Full HRI pipeline to detect Falcon tubes, analyze pointing gestures,
#              compute 3D positions & roll, and command robot to pick and place.

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from ultralytics import YOLO
import math, sys, time, threading
from queue import Queue
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import speech_recognition as sr


# Verbal control mode: "pick", "place", "handover"
verbal_mode = None
def verbal_control_loop():
    global verbal_mode
    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            print("🎤 [VERBAL] Đang lắng nghe lệnh...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source,  timeout=8, phrase_time_limit=5)
                command = recognizer.recognize_google(audio, language='en-US')
                print("✅ [VERBAL] Nghe được:", command)

                command = command.lower()
                if "pick it" in command:
                    verbal_mode = "pick"
                if "pick here" in command:
                    verbal_mode = "pick"
                elif "put there" in command:
                    verbal_mode = "place"
                elif "give me" in command:
                    verbal_mode = "handover"
                else:
                    print("❓ [VERBAL] Không hiểu lệnh.")

            except Exception as e:
                print("❌ [VERBAL]", str(e))

        time.sleep(1)  # tránh CPU overload

# === EMA filter state ===
filtered_base3d = None
filtered_tip3d = None

# === EMA Filter function ===
def ema_filter(new_value, prev_filtered, alpha=0.2):
    return alpha * new_value + (1 - alpha) * prev_filtered

# === Calibration & Constants ===
theta_deg = 40
theta_rad = math.radians(theta_deg)

M0 = np.array([
    [1, 0, 0, 0.162 + 0.259 * math.cos(theta_rad)],
    [0, 1, 0, 0.197],
    [0, 0, 1, 0.702 - 0.259 * math.sin(theta_rad)],
    [0, 0, 0, 1]
])

M1 = np.array([
    [math.cos(theta_rad), 0, math.sin(theta_rad), 0],
    [0, 1, 0, 0],
    [-math.sin(theta_rad), 0, math.cos(theta_rad), 0],
    [0, 0, 0, 1]
])

M2 = np.array([
    [0, -1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

M02 = M0 @ M1 @ M2
M02_inv = np.linalg.inv(M02)

rack_place_position_robot = np.array([0.29, -0.08, 0.14])
rack_pick_position_robot = np.array([0.29, -0.08, 0.14])


initial_rack_place_position = rack_place_position_robot.copy()
initial_rack_pick_position = rack_pick_position_robot.copy()


place_count = 0
pick_count = 0
max_slots = 2

# Định nghĩa workspace dạng hộp axis-aligned
workspace_limits = {
    "x": (-0.15, 0.42),
    "y": (-0.15, 0.42),
    "z": (0.02, 0.30)
}

# === Robot Init ===
bot = InterbotixManipulatorXS(robot_model='rx150', group_name='arm', gripper_name='gripper')
robot_startup()
if bot.arm.group_info.num_joints < 5:
    print("Robot needs 5 joints.")
    robot_shutdown()
    sys.exit()

robot_queue = Queue()

def robot_worker():
    holding = False
    while True:
        task = robot_queue.get()
        if task is None: break
        try:
            action, data = task
            if action == 'pick':
                x, y, z, roll = data
                bot.arm.go_to_home_pose()
                bot.gripper.release()
                bot.arm.set_ee_pose_components(x=x, y=y, z=0.08, roll=roll+math.atan2(y,x), pitch=np.pi/2, yaw=0)
                bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.01, roll=roll+math.atan2(y,x), pitch=np.pi/2, yaw=0)
                bot.gripper.grasp()
                bot.arm.set_ee_pose_components(x=x, y=y, z=0.08, roll=roll+math.atan2(y,x), pitch=np.pi/2, yaw=0)
                bot.arm.set_ee_pose_components(x=x, y=y, z=0.20, roll=0, pitch=0, yaw=0)
                holding = True
            elif action == 'place':
                x, y, z = data
                bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.1, roll=0, pitch=0, yaw=0)
                bot.arm.set_ee_pose_components(x=x, y=y, z=z, roll=0, pitch=0, yaw=0)
                bot.gripper.release()
                bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.1, roll=0, pitch=0, yaw=0)
                bot.arm.go_to_home_pose()
                holding = False
            elif action == 'handover':
                x, y, z = data
                bot.arm.set_ee_pose_components(x=x, y=y, z=z)
                print("Holding for handover...")
                time.sleep(2)  # Thời gian cho người nhận
                bot.gripper.release()
                bot.arm.set_ee_pose_components(x=x, y=y, z=z)
                bot.arm.go_to_home_pose()
                holding = False
            elif action == 'pick_rack':
                bot.gripper.release()
                bot.arm.set_ee_pose_components(x=x, y=y, z=0.25)
                bot.arm.set_ee_pose_components(x=x, y=y, z=0.20)               
                bot.arm.set_ee_pose_components(x=x, y=y, z=z)
                bot.gripper.grasp()
                bot.arm.set_ee_pose_components(x=x, y=y, z=0.20)               
                bot.arm.set_ee_pose_components(x=x, y=y, z=0.25)
        except Exception as e:
            print("Robot error:", e)
        robot_queue.task_done()

threading.Thread(target=robot_worker, daemon=True).start()

# === RealSense Init ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# === MediaPipe & YOLOv8 Init ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
model = YOLO('/home/quang/Colab_tube/runs/segment/train/weights/best.pt')

# === Utility Functions ===
def cam_to_robot(p): return (M02_inv @ np.append(p, 1))[:3]
def deproject(u, v, d): return np.array(rs.rs2_deproject_pixel_to_point(intr, [u, v], d))
def get_avg_depth(u, v, depth, k=2):
    h, w = depth.shape
    u1, v1 = max(u-k,0), max(v-k,0)
    u2, v2 = min(u+k, w-1), min(v+k, h-1)
    roi = depth[v1:v2+1, u1:u2+1]
    roi = roi[roi > 0]
    return np.median(roi) * depth_scale if roi.size > 0 else 0

def is_pointing_to(base, tip, target, thresh=0.06):
    vec = tip - base
    if np.linalg.norm(vec) < 1e-6: return False
    vec = vec / np.linalg.norm(vec)
    to_target = target - base
    dist = np.linalg.norm(np.cross(to_target, vec)) / np.linalg.norm(vec)
    return dist < thresh

def get_pointing_vector(depth_img, lm, h, w):
    u1, v1 = int(lm.landmark[5].x * w), int(lm.landmark[5].y * h)
    u2, v2 = int(lm.landmark[7].x * w), int(lm.landmark[7].y * h)
    d1, d2 = get_avg_depth(u1, v1, depth_img), get_avg_depth(u2, v2, depth_img)
    if d1 > 0 and d2 > 0:
        return deproject(u1, v1, d1), deproject(u2, v2, d2)
    return None, None

def is_ok_sign(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2 +
        (thumb_tip.z - index_tip.z) ** 2
    )

    return distance < 0.03  # Threshold kiểm tra điểm chạm

def is_in_workspace(x, y, z, limits):
    return (limits["x"][0] <= x <= limits["x"][1] and
            limits["y"][0] <= y <= limits["y"][1] and
            limits["z"][0] <= z <= limits["z"][1])

def closest_point_in_workspace(x, y, z, limits):
    cx = max(limits["x"][0], min(x, limits["x"][1]))
    cy = max(limits["y"][0], min(y, limits["y"][1]))
    cz = max(limits["z"][0], min(z, limits["z"][1]))
    return cx, cy, cz

# === Main Loop ===
try:

    verbal_thread = threading.Thread(target=verbal_control_loop, daemon=True)
    verbal_thread.start()
    gesture_rack = None
    gesture_detected = False
    gesture_start = None
    pointing_start = None
    selected_id = None
    picked_obj = None
    pointing_obj = None  # Đối tượng đang được chỉ

    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color = np.asanyarray(aligned.get_color_frame().get_data())
        depth = np.asanyarray(aligned.get_depth_frame().get_data())

        results = model(source=color, task='segment', imgsz=640, conf=0.5, verbose=False)[0]
        annotated = results.plot()
        objects = []

        if results.masks:
            for i, mask_tensor in enumerate(results.masks.data):
                cls_name = results.names[int(results.boxes.cls[i].item())].lower()
                if "tube" not in cls_name: continue

                mask = cv2.resize(mask_tensor.cpu().numpy().astype(np.uint8), (color.shape[1], color.shape[0]))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue

                cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M['m00'] == 0: continue
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                d_center = depth[cy, cx] * depth_scale
                center3d = deproject(cx, cy, d_center)

                tube_region = cv2.bitwise_and(color, color, mask=mask)
                hsv = cv2.cvtColor(tube_region, cv2.COLOR_BGR2HSV)
                mask_orange = cv2.inRange(hsv, np.array([5,100,100]), np.array([20,255,255]))
                contours_o, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours_o: continue

                M_o = cv2.moments(max(contours_o, key=cv2.contourArea))
                if M_o['m00'] == 0: continue
                cx_o = int(M_o['m10']/M_o['m00'])
                cy_o = int(M_o['m01']/M_o['m00'])
                d_orange = depth[cy_o, cx_o] * depth_scale

                orange3d = deproject(cx_o, cy_o, d_orange)

                center3d_robot = (M02_inv @ np.append(center3d, 1))[:3]
                orange3d_robot = (M02_inv @ np.append(orange3d, 1))[:3]

                roll =-math.atan2(orange3d_robot[1]-center3d_robot[1], orange3d_robot[0]-center3d_robot[0])

                objects.append({'bbox': cv2.boundingRect(cnt), 'center3d': center3d, 'roll': roll, 'id': i, 'highlight': False})

        image_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image_rgb)

        if results_hand.multi_hand_landmarks:
            h, w, _ = color.shape
            hand_lm = results_hand.multi_hand_landmarks[0]

            mp.solutions.drawing_utils.draw_landmarks(
                annotated, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
            base3d, tip3d = get_pointing_vector(depth, hand_lm, h, w)

            # === EMA filtering ===
            if base3d is not None and tip3d is not None:
                if filtered_base3d is None:
                    filtered_base3d = base3d
                    filtered_tip3d = tip3d
                else:
                    filtered_base3d = ema_filter(base3d, filtered_base3d)
                    filtered_tip3d = ema_filter(tip3d, filtered_tip3d)  

            if filtered_base3d is not None and filtered_tip3d is not None:
                # Vẽ vector chỉ tay
                base2d = int(hand_lm.landmark[5].x * w), int(hand_lm.landmark[5].y * h)
                tip2d = int(hand_lm.landmark[7].x * w), int(hand_lm.landmark[7].y * h)
                cv2.arrowedLine(annotated, base2d, tip2d, (255, 255, 0), 2)

            if filtered_base3d is not None and filtered_tip3d is not None:
                if picked_obj:
                    rack_cam = (M02 @ np.append(rack_place_position_robot, 1))[:3]
                    if is_pointing_to(filtered_base3d, filtered_tip3d, rack_cam, 0.15):
                        pointing_obj = {"name": "RACK", "pos": rack_place_position_robot}
                        if selected_id == -1:
                            if pointing_start and time.time() - pointing_start > 2:
                                print("[INFO] wait for verbal command place")
                                if verbal_mode == "place":
                                    print("[INFO] Đặt ống vào giá.")
                                    robot_queue.put(('place', tuple(rack_place_position_robot)))

                                    place_count += 1
                                    if place_count >= max_slots:
                                        # rack_place_position_robot = initial_rack_place_position
                                        # place_count = 0
                                        print("[INFO] Đã đặt đủ slot, reset về vị trí rack ban đầu.")
                                    else:
                                        rack_place_position_robot[1] += 0.1

                                    picked_obj = None
                                    pointing_start = None
                                    selected_id = None
                                    verbal_mode = None
                        else:
                            pointing_start = time.time()
                            selected_id = -1
                    elif is_ok_sign(hand_lm):
                        if selected_id == -1:
                            gesture_detected = True
                            if gesture_start and time.time() - gesture_start > 3:
                                print("[INFO] wait for verbal command handover")
                                if verbal_mode == "handover":
                                    print("[INFO] handover")

                                    #hand detect pisition
                                    cx_w, cy_w = int( hand_lm.landmark[0].x * w), int( hand_lm.landmark[0].y * h)
                                    d_wrist = depth[cy_w, cx_w] * depth_scale
                                    wrist3d_cam = deproject(cx_w, cy_w, d_wrist)
                                    wrist3d = cam_to_robot(wrist3d_cam)

                                    if is_in_workspace(wrist3d[0], wrist3d[1], wrist3d[2], workspace_limits):
                                        print("[INFO] Tay người nằm trong workspace.")
                                        target_x, target_y, target_z = wrist3d[0], wrist3d[1], wrist3d[2]
                                    else:
                                        print("[INFO] Tay người nằm ngoài workspace. Tìm điểm gần nhất...")
                                        target_x, target_y, target_z = closest_point_in_workspace(wrist3d[0], wrist3d[1], wrist3d[2], workspace_limits)
                                        print(f"[INFO] Điểm gần nhất trong workspace: ({target_x:.3f}, {target_y:.3f}, {0.2:.3f})")
                                    #input("[ENTER] để xác nhận gừp...")
                                    robot_queue.put(('handover', (target_x, target_y, 0.2)))

                                    gesture_start = None
                                    picked_obj = None
                                    selected_id = None
                                    verbal_mode = None
                            else:
                                # 👇 BẮT ĐẦU ĐẾM THỜI GIAN KHI PHÁT HIỆN LẦN ĐẦU
                                if gesture_start is None:
                                    gesture_start = time.time()
                        else:
                            selected_id = -1    

                    else:
                        gesture_detected = None
                        pointing_start = None
                        selected_id = None

                else:
                    # --- PICK TỪ RACK ---
                    rack_cam = (M02 @ np.append(rack_pick_position_robot, 1))[:3]
                    if is_pointing_to(filtered_base3d, filtered_tip3d, rack_cam, 0.15):
                        pointing_obj = {"name": "RACK", "pos": rack_pick_position_robot}
                        if selected_id == -1:
                            if pointing_start and time.time() - pointing_start > 2:
                                print("[INFO] wait for verbal command pick")
                                if verbal_mode == "pick":
                                    print("[INFO] Nhặt vật từ rack.")
                                    robot_queue.put(('pick_rack', tuple(rack_pick_position_robot)))

                                    pick_count += 1
                                    place_count -=1

                                    if pick_count >= max_slots:
                                        # rack_pick_position_robot = initial_rack_pick_position
                                        # pick_count = 0
                                        print("[INFO] Reset pick rack về vị trí ban đầu.")
                                    else:

                                        rack_pick_position_robot[1] += 0.1                            

                                    picked_obj = {'id': -99, 'source': 'rack'}  # dummy object
                                    pointing_start = None
                                    selected_id = None
                                    verbal_mode = None
                            else:
                                if pointing_start is None:
                                    pointing_start = time.time()
                        else:
                            pointing_start = time.time()
                            selected_id = -1

                    else:
                        # --- PICK TỪ MẶT BÀN (các ống nghiệm) ---
                        closest = None
                        min_dist = 1.0
                        for obj in objects:
                            dist = np.linalg.norm(np.cross(filtered_tip3d - filtered_base3d, obj['center3d'] - filtered_base3d)) / np.linalg.norm(filtered_tip3d - filtered_base3d)
                            if dist < 0.08 and dist < min_dist:
                                closest = obj
                                min_dist = dist

                        if closest:
                            if selected_id == closest['id']:
                                if pointing_start and time.time() - pointing_start > 1.5:
                                    print("[INFO] wait for verbal command pick")
                                    if verbal_mode == "pick":
                                        pos_robot = cam_to_robot(closest['center3d'])
                                        z_safe = max(0.02, min(pos_robot[2], 0.4))
                                        print(f"[INFO] Pick location: x={pos_robot[0]:.3f}, y={pos_robot[1]:.3f}, z={z_safe:.3f}, roll={closest['roll']:.3f}")
                                        pointing_obj = {"name": f"TUBE {closest['id']}", "pos": pos_robot}
                                        robot_queue.put(('pick', (pos_robot[0], pos_robot[1], z_safe, closest['roll'])))
                                        picked_obj = closest
                                        selected_id = pointing_start = None
                                        verbal_mode = None
                            else:
                                pointing_start = time.time()
                                selected_id = closest['id']
                            closest['highlight'] = True
                        else:
                            pointing_start = None
                            selected_id = None
        else:
            pointing_start = selected_id = None

        if gesture_detected:
            cv2.putText(annotated, f"OK SIGN DETECTED", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        for obj in objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,0) if obj['highlight'] else (0,0,255), 2)

        if pointing_obj:
            txt = f"POINTING TO: {pointing_obj['name']} @ [{pointing_obj['pos'][0]:.2f}, {pointing_obj['pos'][1]:.2f}, {pointing_obj['pos'][2]:.2f}]"
            cv2.rectangle(annotated, (10, 10), (630, 40), (0, 0, 0), -1)
            cv2.putText(annotated, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Falcon Tube Interaction", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    robot_queue.put(None)
    bot.arm.go_to_sleep_pose()
    robot_shutdown()
