import os
import numpy as np
import cv2
import torch
from PIL import Image
import sys
sys.path.append('/work')

# apply gamma correction on output images
def gammaCorrect(img, dim=-1):

    if dim == -1:
        dim = len(img.shape) - 1 
    assert(img.shape[dim] == 3)
    gamma, black, color_scale = 2.0,  3.0 / 255.0, [1.4, 1.1, 1.6]

    if torch.is_tensor(img):
        scale = torch.FloatTensor(color_scale).view([3 if i == dim else 1 for i in range(img.dim())])
        img = img * scale.to(img) / 1.1
        correct_img = torch.clamp((((1.0 / (1 - black)) * 0.95 * torch.clamp(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0, 0, 2,)
    else:
        scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
        img = img * scale / 1.1
        correct_img = np.clip((((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0, 0, 2, )
    
    return correct_img

image_path_list_one_sample_per_training_data = ["/work/dataset/m--20180227--0000--6795937--GHS/images/E024_Kiss_Lips_Look_Down/400023/008443.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E051_Tongue_Out_Flat/400023/019284.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E061_Lips_Puffed/400023/023389.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E050_Bite_Tongue/400023/018826.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E022_Raise_Inner_Eyebrows/400023/007802.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E032_Jaw_Open_Pull_Lips_In/400023/011855.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E039_Lips_Open_Right/400023/014281.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E074_Blink/400023/025057.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E063_Nostrils_Sucked_In/400023/024060.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E064_Raise_Right_Eyebrow/400023/024422.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E019_Frown/400023/006778.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E049_Mouth_Open_Tongue_Out/400023/018463.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E036_Stick_Lower_Lip_Out/400023/013244.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E002_Swallow/400023/000181.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E048_Tongue_Out_Lips_Closed/400023/018119.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E065_Raise_Left_Eyebrow/400023/024722.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E010_Smile_Stretched/400023/003235.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E013_Open_Lips_Mouth_Stretch_Nose_Wrinkled/400023/004479.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031632.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E033_Jaw_Clench/400023/012196.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E025_Shh/400023/008800.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E060_Blow_Cheeks_Full_Of_Air/400023/023002.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E021_Pressed_Lips_Brows_Down/400023/007407.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E008_Smile_Mouth_Closed/400023/002407.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E001_Neutral_Eyes_Open/400023/000102.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E042_Mouth_Nose_Left/400023/015447.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E020_Lower_Eyebrows/400023/007066.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E053_Tongue_Out_Rolled/400023/020180.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E062_Nostrils_Dilated/400023/023701.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E038_Bite_Upper_Lip/400023/013948.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E007_Neck_Stretch_Brows_Up/400023/002039.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E017_Jaw_Open_Mouth_Corners_Down_Nose_Wrinkled/400023/006184.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E004_Relaxed_Mouth_Open/400023/001081.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E037_Bite_Lower_Lip/400023/013586.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E006_Jaw_Drop_Brows_Up/400023/001636.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E044_Mouth_Open_Jaw_Left_Show_Teeth/400023/016136.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E043_Mouth_Open_Jaw_Right_Show_Teeth/400023/015779.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E031_Jaw_Open_Lips_Together/400023/011487.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E046_Jaw_Forward/400023/017310.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E011_Jaw_Open_Sharp_Corner_Lip_Stretch/400023/003675.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E040_Lips_Open_Left/400023/014600.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E026_Oooo/400023/009119.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E041_Mouth_Nose_Right/400023/015160.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E009_Smile_Mouth_Open/400023/002799.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E058_Right_Cheek_Puffed/400023/022241.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E003_Neutral_Eyes_Closed/400023/000777.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E055_Tongue_Out_Left_Teeth_Showing/400023/021080.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E018_Raise_Cheeks/400023/006481.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E005_Eyes_Wide_Open/400023/001349.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E015_Jaw_Open_Upper_Lip_Raised/400023/005454.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E028_Scream_Eyebrows_Up/400023/009813.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E012_Jaw_Open_Huge_Smile/400023/004060.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E016_Raise_Upper_Lip_Scrunch_Nose/400023/005784.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E030_Open_Mouth_Wide_Tongue_Up_And_Back/400023/011147.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E035_Lips_Together_Pushed_Forward/400023/012955.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E045_Jaw_Back/400023/016914.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E023_Hide_Lips_Look_Up/400023/008121.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E014_Open_Mouth_Stretch_Nose_Wrinkled/400023/005036.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E056_Suck_Cheeks_In/400023/021582.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E052_Tongue_Out_Thick/400023/019704.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E054_Tongue_Out_Right_Teeth_Showing/400023/020630.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E029_Show_All_Teeth/400023/010225.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E034_Jaw_Open_Lips_Pushed_Out/400023/012581.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E027_Scrunch_Face_Squeeze_Eyes/400023/009461.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E047_Tongue_Over_Upper_Lip/400023/017723.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E057_Cheeks_Puffed/400023/021897.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/E059_Left_Cheek_Puffed/400023/022603.png"]

image_path_list = [
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031632.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031677.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031722.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031767.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031812.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031857.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031902.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031947.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/031992.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032037.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032082.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032127.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032172.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032217.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032262.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032307.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032352.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032397.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032442.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032487.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032532.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032577.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032622.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032667.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032712.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032757.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032802.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032847.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032892.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032937.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/032982.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033027.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033072.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033117.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033162.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033207.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033252.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033297.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033342.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033387.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033432.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033477.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033522.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033567.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033612.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033657.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033702.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033747.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033792.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033837.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033882.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033927.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/033972.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034017.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034062.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034107.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034152.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034197.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034242.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034287.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034332.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034377.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034422.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034467.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034512.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034557.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034602.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034647.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034692.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034737.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034782.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034827.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034872.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034917.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/034962.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/035007.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/035052.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/035097.png",
"/work/dataset/m--20180227--0000--6795937--GHS/images/EXP_ROM07_Facial_Expressions/400023/035142.png"]

output_folder = "/work/render_results/ground_truth_input_testing_data" 
# Load the specific image manually
for image_path in image_path_list:
    if os.path.exists(image_path):
        print(f"Loading image from: {image_path}")
        img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img_writeout = cv2.resize(img, (1334, 2048))
        # Create result directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        # Save the original loaded image
        
        pil_img = (255 * gammaCorrect(img_writeout / 255.0)).astype(np.uint8)
        pil_img = Image.fromarray(pil_img)
        # Apply the same rotation as in save_img_single
        # pil_img = pil_img.rotate(-90, expand=True)
        pil_img.save(os.path.join(output_folder, f"{image_path.split('/')[-1]}"))
        print(f"Saved input image as: {image_path.split('/')[-1]}")