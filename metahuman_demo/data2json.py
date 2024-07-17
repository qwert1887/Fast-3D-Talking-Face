import json

channel_names = [
        "jawForward",
        "jawLeft",
        "jawRight",
        "jawOpen",
        "mouthClose",
        "mouthFunnel",
        "mouthPucker",
        "mouthLeft",
        "mouthRight",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
        "cheekPuff",
        "cheekSquintLeft",
        "cheekSquintRight",
        "noseSneerLeft",
        "noseSneerRight"
]

def datas2jsonframes(topic, datas):
    frame = {topic : {"Parameter" : []}}
    for name in channel_names:
        frame[topic]["Parameter"].append({"Name" : name, "Value" : 0.0})
    json_frames = []
    for frame_data in datas:
        try:
            frame_data = frame_data.tolist()
        except Exception:
            pass
        for i in range(len(channel_names)):
            frame[topic]["Parameter"][i]["Value"] = max(0,frame_data[i])
        json_frames.append(json.dumps(frame))
    return json_frames
