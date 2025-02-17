{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "machine_shape": "hm",
      "gpuType": "L4",
      "authorship_tag": "ABX9TyOEILcyekEziFqtLwdrop1M",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/NANInithin/Vehicle-speed-estimation-through-aerial-videos/blob/main/YOLO11_Speed_esimation_Training.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zjBQvvk5SYUF",
        "outputId": "96179c5d-ac14-444e-9244-833b07dce843"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.75 🚀 Python-3.11.11 torch-2.5.1+cu124 CUDA:0 (NVIDIA L4, 22693MiB)\n",
            "Setup complete ✅ (12 CPUs, 53.0 GB RAM, 33.1/235.7 GB disk)\n",
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "%pip install ultralytics\n",
        "import ultralytics\n",
        "ultralytics.checks()\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "gpu_info = !nvidia-smi\n",
        "gpu_info = '\\n'.join(gpu_info)\n",
        "if gpu_info.find('failed') >= 0:\n",
        "  print('Not connected to a GPU')\n",
        "else:\n",
        "  print(gpu_info)"
      ],
      "metadata": {
        "id": "A_SSMqCyTdfk",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "32cd7d69-beee-4580-fbdb-acbe00aa7cea"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mon Feb 17 19:49:16 2025       \n",
            "+-----------------------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |\n",
            "|-----------------------------------------+------------------------+----------------------+\n",
            "| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n",
            "|                                         |                        |               MIG M. |\n",
            "|=========================================+========================+======================|\n",
            "|   0  NVIDIA L4                      Off |   00000000:00:03.0 Off |                    0 |\n",
            "| N/A   43C    P8             17W /   72W |       3MiB /  23034MiB |      0%      Default |\n",
            "|                                         |                        |                  N/A |\n",
            "+-----------------------------------------+------------------------+----------------------+\n",
            "                                                                                         \n",
            "+-----------------------------------------------------------------------------------------+\n",
            "| Processes:                                                                              |\n",
            "|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |\n",
            "|        ID   ID                                                               Usage      |\n",
            "|=========================================================================================|\n",
            "|  No running processes found                                                             |\n",
            "+-----------------------------------------------------------------------------------------+\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from ultralytics import YOLO\n",
        "model = YOLO(\"yolo11n.pt\")\n",
        "results = model.train(data=\"VisDrone.yaml\", epochs=100, imgsz=640, lr0=0.01, batch=16)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "07Nsl3C7EQ7W",
        "outputId": "487eb1a6-ece0-4114-b2e0-1f7ab036e3e1"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt'...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 5.35M/5.35M [00:00<00:00, 349MB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.3.75 🚀 Python-3.11.11 torch-2.5.1+cu124 CUDA:0 (NVIDIA L4, 22693MiB)\n",
            "\u001b[34m\u001b[1mengine/trainer: \u001b[0mtask=detect, mode=train, model=yolo11n.pt, data=VisDrone.yaml, epochs=100, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=train, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train\n",
            "\n",
            "Dataset 'VisDrone.yaml' images not found ⚠️, missing path '/content/datasets/VisDrone/VisDrone2019-DET-val/images'\n",
            "Downloading https://ultralytics.com/assets/VisDrone2019-DET-train.zip to '/content/datasets/VisDrone/VisDrone2019-DET-train.zip'...\n",
            "Downloading https://ultralytics.com/assets/VisDrone2019-DET-val.zip to '/content/datasets/VisDrone/VisDrone2019-DET-val.zip'...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading https://ultralytics.com/assets/VisDrone2019-DET-test-dev.zip to '/content/datasets/VisDrone/VisDrone2019-DET-test-dev.zip'...\n",
            "Downloading https://ultralytics.com/assets/VisDrone2019-DET-test-challenge.zip to '/content/datasets/VisDrone/VisDrone2019-DET-test-challenge.zip'...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Converting /content/datasets/VisDrone/VisDrone2019-DET-train: 6471it [00:25, 252.66it/s]\n",
            "Converting /content/datasets/VisDrone/VisDrone2019-DET-val: 548it [00:02, 201.61it/s]\n",
            "Converting /content/datasets/VisDrone/VisDrone2019-DET-test-dev: 1610it [00:05, 301.69it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset download success ✅ (97.7s), saved to \u001b[1m/content/datasets\u001b[0m\n",
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf'...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 755k/755k [00:00<00:00, 120MB/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overriding model.yaml nc=80 with nc=10\n",
            "\n",
            "                   from  n    params  module                                       arguments                     \n",
            "  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 \n",
            "  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                \n",
            "  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      \n",
            "  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            "  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     \n",
            "  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            "  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           \n",
            "  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              \n",
            "  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           \n",
            "  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 \n",
            " 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 \n",
            " 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          \n",
            " 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           \n",
            " 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            " 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          \n",
            " 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            " 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           \n",
            " 23        [16, 19, 22]  1    432622  ultralytics.nn.modules.head.Detect           [10, [64, 128, 256]]          \n",
            "YOLO11n summary: 319 layers, 2,591,790 parameters, 2,591,774 gradients, 6.5 GFLOPs\n",
            "\n",
            "Transferred 448/499 items from pretrained weights\n",
            "\u001b[34m\u001b[1mTensorBoard: \u001b[0mStart with 'tensorboard --logdir runs/detect/train', view at http://localhost:6006/\n",
            "Freezing layer 'model.23.dfl.conv.weight'\n",
            "\u001b[34m\u001b[1mAMP: \u001b[0mrunning Automatic Mixed Precision (AMP) checks...\n",
            "\u001b[34m\u001b[1mAMP: \u001b[0mchecks passed ✅\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0mScanning /content/datasets/VisDrone/VisDrone2019-DET-train/labels... 6471 images, 0 backgrounds, 0 corrupt: 100%|██████████| 6471/6471 [00:05<00:00, 1270.63it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0mWARNING ⚠️ /content/datasets/VisDrone/VisDrone2019-DET-train/images/0000137_02220_d_0000163.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0mWARNING ⚠️ /content/datasets/VisDrone/VisDrone2019-DET-train/images/0000140_00118_d_0000002.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0mWARNING ⚠️ /content/datasets/VisDrone/VisDrone2019-DET-train/images/9999945_00000_d_0000114.jpg: 1 duplicate labels removed\n",
            "\u001b[34m\u001b[1mtrain: \u001b[0mWARNING ⚠️ /content/datasets/VisDrone/VisDrone2019-DET-train/images/9999987_00000_d_0000049.jpg: 1 duplicate labels removed\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0mNew cache created: /content/datasets/VisDrone/VisDrone2019-DET-train/labels.cache\n",
            "\u001b[34m\u001b[1malbumentations: \u001b[0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, num_output_channels=3, method='weighted_average'), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/datasets/VisDrone/VisDrone2019-DET-val/labels... 548 images, 0 backgrounds, 0 corrupt: 100%|██████████| 548/548 [00:00<00:00, 827.79it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mval: \u001b[0mNew cache created: /content/datasets/VisDrone/VisDrone2019-DET-val/labels.cache\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Plotting labels to runs/detect/train/labels.jpg... \n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... \n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m SGD(lr=0.01, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)\n",
            "\u001b[34m\u001b[1mTensorBoard: \u001b[0mmodel graph visualization added ✅\n",
            "Image sizes 640 train, 640 val\n",
            "Using 8 dataloader workers\n",
            "Logging results to \u001b[1mruns/detect/train\u001b[0m\n",
            "Starting training for 100 epochs...\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      1/100      9.68G       1.83      2.459      1.087        255        640: 100%|██████████| 405/405 [01:08<00:00,  5.95it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:05<00:00,  3.15it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.26      0.167      0.121     0.0631\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      2/100      7.24G      1.748      1.624      1.033        761        640: 100%|██████████| 405/405 [01:04<00:00,  6.24it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:04<00:00,  3.92it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.226      0.196      0.154     0.0838\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      3/100      10.5G       1.76      1.554      1.025        578        640: 100%|██████████| 405/405 [01:04<00:00,  6.33it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:05<00:00,  3.31it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.269      0.215      0.182     0.0969\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      4/100      7.37G      1.733      1.507      1.016        413        640: 100%|██████████| 405/405 [01:02<00:00,  6.45it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:04<00:00,  4.38it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.279      0.221       0.19      0.105\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      5/100      9.42G      1.689      1.442      1.003        477        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:04<00:00,  4.34it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.276      0.218       0.19      0.107\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      6/100      8.92G      1.664      1.393     0.9931        410        640: 100%|██████████| 405/405 [01:02<00:00,  6.45it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.68it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.306       0.25      0.219       0.12\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      7/100      6.91G      1.638      1.354      0.987        568        640: 100%|██████████| 405/405 [01:03<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.80it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.315      0.251      0.222      0.123\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      8/100      5.95G      1.613      1.323     0.9789        366        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.84it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.34      0.252      0.231       0.13\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      9/100      7.87G      1.598      1.301     0.9746        659        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.89it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.345      0.258      0.239      0.135\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     10/100      8.15G      1.587      1.275     0.9681        437        640: 100%|██████████| 405/405 [01:03<00:00,  6.38it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.91it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.353      0.264      0.241      0.134\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     11/100      9.48G      1.563      1.242     0.9658        564        640: 100%|██████████| 405/405 [01:02<00:00,  6.45it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.83it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.358       0.27       0.25       0.14\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     12/100      9.46G      1.557      1.238     0.9632        293        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.01it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.359      0.265      0.248       0.14\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     13/100      8.21G      1.549      1.216     0.9624        455        640: 100%|██████████| 405/405 [01:02<00:00,  6.44it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.97it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.343      0.265      0.247      0.139\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     14/100      8.86G       1.53      1.209     0.9564        589        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.98it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.363       0.28      0.264      0.148\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     15/100      7.96G      1.546      1.199     0.9573        661        640: 100%|██████████| 405/405 [01:03<00:00,  6.36it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.90it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.369      0.284      0.268      0.151\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     16/100      7.23G      1.527      1.184     0.9551        814        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.92it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.368      0.275      0.262      0.147\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     17/100      11.2G      1.519      1.179     0.9522        606        640: 100%|██████████| 405/405 [01:03<00:00,  6.37it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.95it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.37      0.278      0.264      0.149\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     18/100      8.45G      1.519      1.168     0.9521        479        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.11it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.368      0.285       0.27      0.153\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     19/100      7.78G      1.498      1.153     0.9491        449        640: 100%|██████████| 405/405 [01:02<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.88it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.359      0.279      0.266      0.151\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     20/100      8.21G      1.494      1.148     0.9467        669        640: 100%|██████████| 405/405 [01:02<00:00,  6.46it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.99it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.376       0.29      0.274      0.156\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     21/100      6.98G      1.493      1.143     0.9453        748        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.96it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.383      0.296      0.281      0.158\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     22/100      8.31G      1.482      1.132      0.944        678        640: 100%|██████████| 405/405 [01:03<00:00,  6.38it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.03it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.382      0.284      0.275      0.158\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     23/100      10.6G       1.48      1.126     0.9442        493        640: 100%|██████████| 405/405 [01:02<00:00,  6.48it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.95it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.365      0.305      0.283       0.16\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     24/100      8.59G       1.47      1.116     0.9414        666        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.389      0.298      0.284       0.16\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     25/100      7.14G      1.467      1.109       0.94        522        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.90it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.384       0.29      0.279      0.159\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     26/100      11.4G      1.476      1.115     0.9415        639        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.88it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.391       0.29      0.281      0.161\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     27/100      7.28G      1.462      1.103     0.9355        573        640: 100%|██████████| 405/405 [01:03<00:00,  6.36it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.92it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759        0.4      0.303      0.293      0.168\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     28/100      7.39G      1.463        1.1     0.9359        690        640: 100%|██████████| 405/405 [01:03<00:00,  6.38it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.95it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759        0.4        0.3      0.288      0.164\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     29/100      9.93G      1.459        1.1     0.9358        663        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.01it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.412      0.302      0.296       0.17\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     30/100      7.89G      1.456      1.086      0.935        622        640: 100%|██████████| 405/405 [01:03<00:00,  6.38it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.88it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.411      0.303        0.3      0.171\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     31/100      8.02G       1.45      1.086     0.9355        387        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.95it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.413      0.299      0.298       0.17\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     32/100      10.4G      1.451       1.08     0.9332        446        640: 100%|██████████| 405/405 [01:03<00:00,  6.35it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.95it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.418      0.307      0.301      0.172\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     33/100       7.9G      1.445      1.076      0.933        594        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.02it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.404      0.316      0.305      0.176\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     34/100      7.12G      1.443      1.075     0.9313        619        640: 100%|██████████| 405/405 [01:03<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.409      0.305      0.303      0.173\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     35/100      6.64G      1.438      1.069     0.9301        493        640: 100%|██████████| 405/405 [01:02<00:00,  6.44it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.01it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.402      0.314      0.303      0.174\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     36/100      7.66G      1.432      1.062     0.9304        696        640: 100%|██████████| 405/405 [01:02<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.411      0.311      0.303      0.175\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     37/100      9.42G      1.425      1.056     0.9285        439        640: 100%|██████████| 405/405 [01:02<00:00,  6.44it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.99it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759        0.4      0.316      0.305      0.176\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     38/100      8.38G      1.435      1.063     0.9292        464        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.42      0.312      0.309      0.177\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     39/100      7.14G      1.429      1.053     0.9268        811        640: 100%|██████████| 405/405 [01:02<00:00,  6.45it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.94it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.404      0.319      0.308      0.177\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     40/100      10.5G      1.423      1.052     0.9277        596        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.16it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.42      0.306      0.308      0.177\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     41/100      7.09G      1.414      1.039     0.9255        584        640: 100%|██████████| 405/405 [01:02<00:00,  6.46it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.414      0.319      0.312      0.181\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     42/100      7.78G      1.416      1.045     0.9273        438        640: 100%|██████████| 405/405 [01:02<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.419      0.317      0.311       0.18\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     43/100      9.16G       1.42      1.045      0.925        646        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.413      0.323      0.313       0.18\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     44/100      8.03G      1.412      1.033     0.9233        989        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.99it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.418      0.321      0.316      0.183\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     45/100        10G      1.404      1.028     0.9233        466        640: 100%|██████████| 405/405 [01:02<00:00,  6.44it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.07it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.421      0.322      0.318      0.184\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     46/100        11G      1.414      1.029     0.9248        409        640: 100%|██████████| 405/405 [01:03<00:00,  6.37it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.92it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.423      0.317      0.316      0.183\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     47/100      8.57G      1.402      1.023     0.9224        897        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.10it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.416      0.322      0.314      0.182\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     48/100      6.24G      1.406      1.026     0.9232        715        640: 100%|██████████| 405/405 [01:02<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.06it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.42      0.325      0.318      0.183\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     49/100      8.82G      1.398      1.018     0.9206        526        640: 100%|██████████| 405/405 [01:03<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.10it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.424       0.32       0.32      0.187\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     50/100       7.9G      1.393      1.017     0.9214        672        640: 100%|██████████| 405/405 [01:02<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.11it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.43      0.325      0.324      0.189\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     51/100      8.57G      1.393      1.017     0.9202        350        640: 100%|██████████| 405/405 [01:02<00:00,  6.44it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.01it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.408      0.328       0.32      0.186\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     52/100      8.47G      1.392      1.009     0.9203        608        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.13it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.423      0.324      0.322      0.188\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     53/100      11.7G      1.392      1.011     0.9209        382        640: 100%|██████████| 405/405 [01:03<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.429      0.324      0.322      0.185\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     54/100      9.33G      1.388      1.006     0.9176        522        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.13it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.423      0.327      0.325      0.189\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     55/100       8.4G      1.382     0.9992     0.9184        411        640: 100%|██████████| 405/405 [01:02<00:00,  6.46it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.12it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.44      0.322      0.326      0.189\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     56/100      7.41G      1.385     0.9961     0.9161        374        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.99it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.431      0.328      0.324      0.188\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     57/100      9.59G      1.377     0.9944     0.9153        496        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.06it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.427      0.326      0.327       0.19\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     58/100       8.9G       1.37     0.9857     0.9153        423        640: 100%|██████████| 405/405 [01:02<00:00,  6.45it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.434      0.326      0.326       0.19\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     59/100      8.68G      1.377     0.9897     0.9164        445        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.14it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.434      0.327      0.328       0.19\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     60/100      8.28G      1.378      0.991     0.9157        791        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.429       0.33      0.327       0.19\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     61/100      8.94G      1.371     0.9851      0.916        407        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.428      0.331      0.327      0.191\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     62/100      9.08G      1.371     0.9875     0.9147        379        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.10it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.427      0.328      0.327      0.191\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     63/100      9.72G      1.366     0.9811     0.9148        593        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.426       0.33      0.327      0.189\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     64/100      5.84G      1.366     0.9785      0.914        505        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.43      0.335      0.329      0.191\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     65/100      9.07G      1.367     0.9764     0.9128        846        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.11it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.424      0.331      0.326      0.189\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     66/100      11.2G      1.365      0.976     0.9136        583        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.437      0.332      0.331      0.192\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     67/100      9.46G      1.366     0.9718     0.9133        472        640: 100%|██████████| 405/405 [01:03<00:00,  6.36it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.43      0.331      0.327       0.19\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     68/100      8.07G      1.362     0.9674     0.9112       1302        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.07it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.435      0.328      0.328      0.191\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     69/100      10.3G      1.347     0.9609      0.911        562        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.01it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.424      0.335      0.329      0.192\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     70/100      7.76G      1.349     0.9582     0.9111        366        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.02it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.434      0.329      0.328      0.191\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     71/100      7.39G      1.353     0.9598     0.9101        504        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.03it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.426      0.337      0.331      0.193\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     72/100      10.2G      1.359     0.9629     0.9109        714        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.07it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.428      0.337       0.33      0.193\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     73/100         8G      1.352      0.955     0.9103        463        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.427      0.338      0.333      0.194\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     74/100      9.34G      1.361     0.9628     0.9092        518        640: 100%|██████████| 405/405 [01:03<00:00,  6.36it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.434      0.334      0.331      0.193\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     75/100      8.84G      1.336     0.9427     0.9079        352        640: 100%|██████████| 405/405 [01:02<00:00,  6.48it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.12it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.442      0.332      0.331      0.192\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     76/100      7.29G       1.34     0.9452     0.9099        376        640: 100%|██████████| 405/405 [01:02<00:00,  6.46it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.449      0.331      0.332      0.194\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     77/100      6.95G      1.343     0.9485     0.9071        575        640: 100%|██████████| 405/405 [01:02<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.447      0.332      0.334      0.195\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     78/100      7.11G      1.343     0.9474     0.9072        546        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.14it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.435      0.336      0.333      0.194\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     79/100      10.3G      1.331     0.9369     0.9066        599        640: 100%|██████████| 405/405 [01:02<00:00,  6.44it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.429      0.338      0.334      0.195\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     80/100        10G      1.337     0.9413      0.907        688        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.432      0.339      0.334      0.194\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     81/100      8.38G      1.336     0.9376     0.9073        879        640: 100%|██████████| 405/405 [01:03<00:00,  6.37it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.439      0.337      0.335      0.195\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     82/100      8.63G      1.334     0.9327     0.9051        549        640: 100%|██████████| 405/405 [01:03<00:00,  6.42it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.12it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.441      0.337      0.337      0.196\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     83/100      9.08G      1.329     0.9342     0.9054        503        640: 100%|██████████| 405/405 [01:02<00:00,  6.44it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.451      0.335      0.336      0.196\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     84/100      7.17G      1.327     0.9276     0.9047        466        640: 100%|██████████| 405/405 [01:02<00:00,  6.44it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.20it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.449      0.337      0.337      0.197\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     85/100      9.37G      1.327     0.9284     0.9043        567        640: 100%|██████████| 405/405 [01:03<00:00,  6.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.13it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759       0.44      0.335      0.336      0.196\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     86/100      10.8G      1.325     0.9245     0.9041        585        640: 100%|██████████| 405/405 [01:03<00:00,  6.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.17it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.442      0.336      0.337      0.197\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     87/100      10.1G      1.321     0.9207      0.903        684        640: 100%|██████████| 405/405 [01:02<00:00,  6.46it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.438      0.337      0.337      0.197\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     88/100      7.12G      1.312     0.9132     0.9011        742        640: 100%|██████████| 405/405 [01:02<00:00,  6.44it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.434       0.34      0.338      0.197\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     89/100      9.79G      1.319     0.9176     0.9023        226        640: 100%|██████████| 405/405 [01:03<00:00,  6.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.14it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.438      0.338      0.336      0.197\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     90/100      8.71G       1.32     0.9228     0.9036        592        640: 100%|██████████| 405/405 [01:03<00:00,  6.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.437       0.34      0.337      0.197\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Closing dataloader mosaic\n",
            "\u001b[34m\u001b[1malbumentations: \u001b[0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, num_output_channels=3, method='weighted_average'), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     91/100      4.89G      1.294     0.8967     0.9047        429        640: 100%|██████████| 405/405 [00:58<00:00,  6.94it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.01it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.446      0.338      0.335      0.196\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     92/100      6.25G      1.282     0.8809        0.9        287        640: 100%|██████████| 405/405 [00:57<00:00,  7.08it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.07it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.437      0.338      0.334      0.195\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     93/100      10.2G      1.279     0.8765     0.8995         86        640: 100%|██████████| 405/405 [00:57<00:00,  7.04it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.10it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.435      0.337      0.333      0.195\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     94/100      9.62G      1.274     0.8719        0.9        323        640: 100%|██████████| 405/405 [00:56<00:00,  7.13it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.11it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.437      0.335      0.332      0.194\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     95/100      8.72G      1.269     0.8673     0.8984        313        640: 100%|██████████| 405/405 [00:57<00:00,  7.09it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.07it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.442      0.333      0.333      0.195\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     96/100      6.21G      1.265     0.8626     0.8971        190        640: 100%|██████████| 405/405 [00:57<00:00,  7.07it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.441      0.334      0.333      0.194\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     97/100      8.91G      1.263     0.8588     0.8971        303        640: 100%|██████████| 405/405 [00:57<00:00,  7.07it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.06it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.439      0.336      0.333      0.194\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     98/100      7.68G       1.26     0.8572     0.8952        373        640: 100%|██████████| 405/405 [00:57<00:00,  7.09it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.441      0.335      0.333      0.194\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     99/100      7.31G      1.263     0.8586     0.8965        221        640: 100%|██████████| 405/405 [00:56<00:00,  7.11it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.13it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.439      0.334      0.333      0.195\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "    100/100      9.08G       1.26     0.8541      0.896        320        640: 100%|██████████| 405/405 [00:56<00:00,  7.14it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  5.11it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.438      0.336      0.333      0.194\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "100 epochs completed in 1.863 hours.\n",
            "Optimizer stripped from runs/detect/train/weights/last.pt, 5.5MB\n",
            "Optimizer stripped from runs/detect/train/weights/best.pt, 5.5MB\n",
            "\n",
            "Validating runs/detect/train/weights/best.pt...\n",
            "Ultralytics 8.3.75 🚀 Python-3.11.11 torch-2.5.1+cu124 CUDA:0 (NVIDIA L4, 22693MiB)\n",
            "YOLO11n summary (fused): 238 layers, 2,584,102 parameters, 0 gradients, 6.3 GFLOPs\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:09<00:00,  1.95it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        548      38759      0.436      0.341      0.338      0.197\n",
            "            pedestrian        520       8844      0.437      0.358      0.355      0.153\n",
            "                people        482       5125      0.513      0.241      0.287      0.104\n",
            "               bicycle        364       1287      0.239       0.12     0.0995     0.0385\n",
            "                   car        515      14064      0.634      0.753      0.757      0.522\n",
            "                   van        421       1975      0.446      0.384      0.375      0.259\n",
            "                 truck        266        750      0.434      0.304      0.308      0.202\n",
            "              tricycle        337       1045      0.352      0.241      0.217      0.123\n",
            "       awning-tricycle        220        532      0.279       0.16       0.12     0.0754\n",
            "                   bus        131        251      0.579      0.462      0.496      0.342\n",
            "                 motor        485       4886      0.449      0.383      0.368      0.154\n",
            "Speed: 0.1ms preprocess, 0.6ms inference, 0.0ms loss, 1.3ms postprocess per image\n",
            "Results saved to \u001b[1mruns/detect/train\u001b[0m\n"
          ]
        }
      ]
    }
  ]
}