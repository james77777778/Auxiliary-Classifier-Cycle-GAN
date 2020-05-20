import os
import os.path as osp
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from absl import flags, app

from src.utils.dataset import UnalignedDataset
from src.utils.utils import dict2table
from src.models.cyclegan import CycleGAN, ACCycleGAN


FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', 'output', "")
flags.DEFINE_enum('model', 'cyclegan', ['cyclegan', 'accyclegan'],
                  "model type")
flags.DEFINE_string('dataset', 'dataset/summer2winter_yosemite/test', "")
flags.DEFINE_string('model_weight', '', "")


def main(agrv):
    print(dict2table(FLAGS.flag_values_dict()))
    # parameters
    OUTPUT_DIR = Path(FLAGS.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")

    # load dataset
    train_dataset = UnalignedDataset(
        FLAGS.dataset, image_size=128, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    # load model
    if FLAGS.model == "cyclegan":
        model = CycleGAN().to(device)
    elif FLAGS.model == "accyclegan":
        model = ACCycleGAN().to(device)
    model_weights = torch.load(FLAGS.model_weight)
    model.load_state_dict(model_weights['model'])
    model.train()

    # get data and do the inference
    for batch in train_loader:
        A, B = batch["A"].to(device), batch["B"].to(device)
        A_path, B_path = batch["A_path"], batch["B_path"]
        with torch.no_grad():
            res = model(A, B)
            imgs_A = torch.cat((A, res["fake_B"]))
            imgs_B = torch.cat((B, res["fake_A"]))
            # denormalize
            imgs_A = train_dataset.denormalize(imgs_A)
            imgs_B = train_dataset.denormalize(imgs_B)
            # to pillow
            imgs_A = [TF.to_pil_image(o.cpu()) for o in imgs_A]
            imgs_B = [TF.to_pil_image(o.cpu()) for o in imgs_B]
            for i, (a, b) in enumerate(zip(imgs_A, imgs_B)):
                savename_A = osp.basename(A_path[i]) + ".jpg"
                savename_B = osp.basename(B_path[i]) + ".jpg"
                a.save(OUTPUT_DIR/"A"/savename_A, quality=95, subsampling=0)
                b.save(OUTPUT_DIR/"B"/savename_B, quality=95, subsampling=0)
                print(savename_A, savename_B)


if __name__ == "__main__":
    app.run(main)
