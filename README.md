<h1 align="center">● Medical SAM Adapter</h1>

<p align="center">
    <a href="https://discord.gg/DN4rvk95CC">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <img src="https://img.shields.io/static/v1?label=license&message=GPL&color=white&style=flat" alt="License"/>
</p>

Medical SAM Adapter, or say MSA, is a project to fineturn [SAM](https://github.com/facebookresearch/segment-anything) using [Adaption](https://lightning.ai/pages/community/tutorial/lora-llm/) for the Medical Imaging.
This method is elaborated on the paper [Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation](https://arxiv.org/abs/2304.12620).

## A Quick Overview 
 <img width="880" height="380" src="https://github.com/WuJunde/Medical-SAM-Adapter/blob/main/figs/medsamadpt.jpeg">

 ## News
 - [TOP] Join in our [Discord](https://discord.gg/EqbgSPEX) to ask questions and discuss with others.
 - [TOP] 24-03-02 We have released our pre-trained Adapters in [Medical-Adapter-Zoo](https://huggingface.co/KidsWithTokens/Medical-Adapter-Zoo/tree/main). Try it without painful training 😉 Credit: @shinning0821
 - 23-05-10. This project is still quickly updating 🌝. Check TODO list to see what will be released next.
 - 23-05-11. GitHub Dicussion opened. You guys can now talk, code and make friends on the playground 👨‍❤️‍👨. 
 - 23-12-22. Released data loader and example case on [REFUGE](https://refuge.grand-challenge.org/) dataset. Credit: @jiayuanz3
 - 24-01-04. Released the Efficient Med-SAM-Adapter❗️ A new, faster, and more lightweight version incorporates Meta [EfficientSAM](https://yformer.github.io/efficient-sam/)🏇. Full credit goes to @shinning0821. 
 - 24-01-07. The image resolution now can be resized by ``-image_size``. Credit: @shinning0821
 - 24-01-11. Added a detailed guide on utilizing the Efficient Med-SAM-Adapter, complete with a comparison of performance and speed. You can find this resource in  [guidance/efficient_sam.ipynb](./guidance/efficient_sam.ipynb). Credit: @shinning0821
 - 24-01-14. We've just launched our first official version, v0.1.0-alpha 🥳. This release includes support for [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), which can be activated by setting ``-net mobile_sam``. Additionally, you now have the flexibility to use ViT, Tiny ViT, and Efficient ViT as encoders. Check the details [here](https://github.com/KidsWithTokens/Medical-SAM-Adapter/releases/tag/v0.1.0-alpha). Credit: @shinning0821
 - 24-01-20. Added a guide on utilizing the mobile sam in Med-SAM-Adapter, with a comparison of performance and speed. You can find it in [guidance/mobile_sam.ipynb](https://github.com/KidsWithTokens/Medical-SAM-Adapter/blob/main/guidance/mobile_sam.ipynb) Credit: @shinning0821
 - 24-01-21. We've added [LoRA](https://huggingface.co/docs/diffusers/training/lora) to our framework🤖. Use it by setting ``-mod`` as ``sam_lora``.
A guidance can be found in [here](https://github.com/KidsWithTokens/Medical-SAM-Adapter/blob/main/guidance/lora.ipynb). Credit: @shinning0821
 - 24-01-22. We've added dataloader for [LIDC dataset](https://paperswithcode.com/dataset/lidc-idri), a multi-rater(4 raters 👨‍⚕️🧑🏽‍⚕️👩‍⚕️🧑🏽‍⚕️) lesions segmentation from low-dose lung CTs 🩻. You can download the preprocessed LIDC dataset at [here](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch). Also updated environment, and random_click function. Credit: @jiayuanz3
 - 24-03-06. We've supported multi-class segmentation. Use it by setting ``-multimask_output`` to the number of classes favored. Also updated REFUGE example to two classes (optic disc & cup). Credit: @LJQCN101
 - 24-03-06. We've supported many other datasets and rebuild the code of datasets and dataloaders. Seen in `guidance/Dataset.md` Credit: @shinning0821

## Medical Adapter Zoo 🐘🐊🦍🦒🦨🦜🦥
We've released a bunch of pre-trained Adapters for various organs/lesions in [Medical-Adapter-Zoo](https://huggingface.co/KidsWithTokens/Medical-Adapter-Zoo/tree/main). Just pick the adapter that matches your disease and easily adjust SAM to suit your specific needs 😉. 

If you can't find what you're looking for. Please suggest it through any contact method available to us (GitHub issue, HuggingFace community, or [Discord](https://discord.gg/EqbgSPEX)). We'll do our very best to include it.
 
 ## Requirement

 Install the environment:

 ``conda env create -f environment.yml``

 ``conda activate sam_adapt``

 Then download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), and put it at ./checkpoint/sam/

 You can run:

 ``wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth``

 ``mv sam_vit_b_01ec64.pth ./checkpoint/sam``
 creat the folder if it does not exist

 ## Example Cases

 ### Melanoma Segmentation from Skin Images (2D)

 1. Download ISIC dataset part 1 from https://challenge.isic-archive.com/data/. Then put the csv files in "./data/isic" under your data path. Your dataset folder under "your_data_path" should be like:
ISIC/
     ISBI2016_ISIC_Part1_Test_Data/...
     
     ISBI2016_ISIC_Part1_Training_Data/...
     
     ISBI2016_ISIC_Part1_Test_GroundTruth.csv
     
      ISBI2016_ISIC_Part1_Training_GroundTruth.csv
    
    You can fine the csv files [here](https://github.com/KidsWithTokens/MedSegDiff/tree/master/data/isic_csv)

 3. Begin Adapting! run: ``python train.py -net sam -mod sam_adpt -exp_name *msa_test_isic* -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 32 -dataset isic -data_path *../data*``
 change "data_path" and "exp_name" for your own useage. you can change "exp_name" to anything you want.

 Select ISIC prompts with ``-prompt``: ``click`` (default), ``click3``, ``bbox``, ``bbox0.5``, or ``bbox0.75``. For example, add ``-prompt click3`` for three foreground clicks or ``-prompt bbox0.5`` for a box that encloses approximately 50% of foreground pixels. ``bbox0.75`` targets 75% foreground overlap, while ``bbox`` uses the tight ground-truth box. Box prompts are supported with SAM and MobileSAM.

 You can descrease the ``image size`` or batch size ``b`` if out of memory.

 3. Evaluation: The code can automatically evaluate the model on the test set during traing, set "--val_freq" to control how many epoches you want to evaluate once. You can also run val.py for the independent evaluation.

 4. Result Visualization: You can set "--vis" parameter to control how many epoches you want to see the results in the training or evaluation process.

 In default, everything will be saved at `` ./logs/`` 

 ### REFUGE: Optic-disc Segmentation from Fundus Images (2D) 
 [REFUGE](https://refuge.grand-challenge.org/) dataset contains 1200 fundus images with optic disc/cup segmentations and clinical glaucoma labels. 

 1. Dowaload the dataset manually from [here](https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/tree/main), or using command lines:

 ``git lfs install``

 ``git clone git@hf.co:datasets/realslimman/REFUGE-MultiRater``

 unzip and put the dataset to the target folder

 ``unzip ./REFUGE-MultiRater.zip``

 ``mv REFUGE-MultiRater ./data``

 2. For training the adapter, run: ``python train.py -net sam -mod sam_adpt -exp_name REFUGE-MSAdapt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 32 -dataset REFUGE -data_path ./data/REFUGE-MultiRater``
 you can change "exp_name" to anything you want.

 You can descrease the ``image size`` or batch size ``b`` if out of memory.

 ### Abdominal Multiple Organs Segmentation (3D)

 This tutorial demonstrates how MSA can adapt SAM to 3D multi-organ segmentation task using the BTCV challenge dataset.
For BTCV dataset, under Institutional Review Board (IRB) supervision, 50 abdomen CT scans of were randomly selected from a combination of an ongoing colorectal cancer chemotherapy trial, and a retrospective ventral hernia study. The 50 scans were captured during portal venous contrast phase with variable volume sizes (512 x 512 x 85 - 512 x 512 x 198) and field of views (approx. 280 x 280 x 280 mm3 - 500 x 500 x 650 mm3). The in-plane resolution varies from 0.54 x 0.54 mm2 to 0.98 x 0.98 mm2, while the slice thickness ranges from 2.5 mm to 5.0 mm.
Target: 13 abdominal organs including
Spleen
Right Kidney
Left Kidney
Gallbladder
Esophagus
Liver
Stomach
Aorta
IVC
Portal and Splenic Veins
Pancreas
Right adrenal gland
Left adrenal gland.
Modality: CT
Size: 30 3D volumes (24 Training + 6 Testing)
Challenge: BTCV MICCAI Challenge
The following figure shows image patches with the organ sub-regions that are annotated in the CT (top left) and the final labels for the whole dataset (right).
1. Prepare BTCV dataset following [MONAI](https://docs.monai.io/en/stable/index.html) instruction:
Download BTCV dataset from: https://www.synapse.org/#!Synapse:syn3193805/wiki/217752. After you open the link, navigate to the "Files" tab, then download Abdomen/RawData.zip.
After downloading the zip file, unzip. Then put images from RawData/Training/img in ../data/imagesTr, and put labels from RawData/Training/label in ../data/labelsTr.
Download the json file for data splits from this [link](https://drive.google.com/file/d/1qcGh41p-rI3H_sQ0JwOAhNiQSXriQqGi/view). Place the JSON file at ../data/dataset_0.json.
2. For the Adaptation, run: ``python train.py -net sam -mod sam_adpt -exp_name msa-3d-sam-btcv -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 8 -dataset decathlon -thd True -chunk 96 -data_path ../data -num_sample 4``  
You can modify following parameters to save the memory usage: '-b' the batch size, '-chunk' the 3D depth (channel) for each sample, '-num_sample' number of samples for [Monai.RandCropByPosNegLabeld](https://docs.monai.io/en/stable/transforms.html#randcropbyposneglabeld), 'evl_chunk' the 3D channel split step in the evaluation, decrease it if out of memory in the evaluation. 
## Run on  your own dataset
It is simple to run MSA on the other datasets. Just write another dataset class following which in `` ./dataset.py``. You only need to make sure you return a dict with 
     {
                 'image': A tensor saving images with size [C,H,W] for 2D image, size [C, H, W, D] for 3D data.
                 D is the depth of 3D volume, C is the channel of a scan/frame, which is commonly 1 for CT, MRI, US data. 
                 If processing, say like a colorful surgical video, D could the number of time frames, and C will be 3 for a RGB frame.
                 'label': The target masks. Same size with the images except the resolutions (H and W).
                 'p_label': The prompt label to decide positive/negative prompt. To simplify, you can always set 1 if don't need the negative prompt function.
                 'pt': The prompt. Should be the same as that in SAM, e.g., a click prompt should be [x of click, y of click], one click for each scan/frame if using 3d data.
                 'image_meta_dict': Optional. if you want save/visulize the result, you should put the name of the image in it with the key ['filename_or_obj'].
                 ...(others as you want)
     }
Welcome to open issues if you meet any problem. It would be appreciated if you could contribute your dataset extensions. Unlike natural images, medical images vary a lot depending on different tasks. Expanding the generalization of a method requires everyone's efforts.

 ### TODO LIST

- [ ] Jupyter tutorials.
- [x] Fix bugs in BTCV. Add BTCV example.
- [ ] Release REFUGE2, BraTs dataloaders and examples
- [x] Changable Image Resolution 
- [ ] Fix bugs in Multi-GPU parallel
- [x] Sample and Vis in training
- [ ] Release general data pre-processing and post-processing
- [x] Release evaluation
- [ ] Deploy on HuggingFace
- [x] configuration
- [ ] Release SSL code
- [ ] Release Medical Adapter Zoo

 ## Cite
 ~~~
@misc{wu2023medical,
      title={Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation}, 
      author={Junde Wu and Wei Ji and Yuanpei Liu and Huazhu Fu and Min Xu and Yanwu Xu and Yueming Jin},
      year={2023},
      eprint={2304.12620},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
 ~~~






