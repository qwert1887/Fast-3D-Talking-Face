import numpy as np
import os, argparse
from SelfTalk import SelfTalk
from transformers import Wav2Vec2Processor
import torch
import time
from fastapi import FastAPI
import uvicorn
app = FastAPI()

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # egl


parser = argparse.ArgumentParser(
    description='SelfTalk: A Self-Supervised Commutative Training Diagram to Comprehend 3D Talking Faces')
parser.add_argument("--model_name", type=str, default="BlendVOCA")
parser.add_argument("--dataset", type=str, default="BlendVOCA")
parser.add_argument("--feature_dim", type=int, default=512, help='512 for vocaset')
parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset')
parser.add_argument("--bs_dim", type=int, default=32 ,
                    help='number of blendshape')
parser.add_argument("--device", type=str, default="cuda", help='cuda or cpu')
args = parser.parse_args()

# build model
model = SelfTalk(args)
model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name)),
                                 map_location=torch.device(args.device)))
model = model.to(torch.device(args.device))
model.eval()
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

from pydantic import BaseModel

class Item(BaseModel):
    file:list

@app.post('/')
@torch.no_grad()
def test_model(item:Item):
    start = time.time()
    speech_array = np.array(item.file)
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    start = time.time()
    prediction = model.predict(audio_feature)
    prediction = prediction.squeeze().detach().cpu().numpy()
    np.save("fast_3d.npy",prediction)
    end = time.time()
    print("Model predict time: ", end - start)
    result = {"bs":prediction.tolist()}
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3392)
