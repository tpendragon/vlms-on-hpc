# VLMs on HPC

This is a set of scripts that can be used to run text recognition (OCR/HTR) on Princeton's high-performance computing clusters.  

The main goal is to
- Download images from a IIIF endpoint or folder of PDF files
- Download an open-source model from HuggingFace Hub
- Recognize text in the images and save them as markdown

For Princeton faculty, staff, and students, you can request an account on Adroit [here](https://forms.rc.princeton.edu/registration/).

Regularly updated documentation on Adroit can be found [here](https://researchcomputing.princeton.edu/systems/adroit) 

## Connect to the Server
To connect to the Adroit cluster 

```bash
ssh <username>@adroit.princeton.edu
``` 

the password is the same one you'd use for other CAS logins. You'll need to accept a Duo Push or other authentication.  If you're off campus, keep in mind that you must connect through the campus VPN.  

Alternatively, go to Adroit Cluster Shell Access from [myadroit.princeton.edu](https://myadroit.princeton.edu)

Once logged in, you'll be in your home directory on the login node. For me, it's `/home/aj7878` You have very limited space on the login node (server).


## Clone the Code
You will want to navigate to your folder in the shared network drive. For example, 
```bash
cd /scratch/network/<username>
```

Choose a good name for your project, make a new directory (`mkdir`), and change directory (`cd`)
```bash
mkdir my_awesome_project && cd my_awesome_project
```

Once in your directory, you can clone this repository into your folder
```
git clone https://github.com/PULdischo/vlms-on-hpc.git .
```

now activate Anaconda (`conda`) to manage Python dependencies
```bash 
module load anaconda3/2024.6
``` 
> note that a newer version may be available. You can enter `module avail` to list the available conda modules.

create a virtual enviornment 
```bash
conda env create -f conda_env.yml
```

The HPC node does not have access to the Internet, so you need to download all model and image files in advance on the login node. You can find Hugging Face Hub models here: https://huggingface.co/models


To do this: 
```bash
python fetch.py model <huggingface/repo-name> # default is "nanonets/Nanonets-OCR-s"
```

I find it helpful to do a test run on the login node to check for errors.  With your virtual enviornment activated, you can run `python main.py`. If everything is set up properly, you'll get an error from vLLM that it can't find the GPU (the login node doesn't have one).

## A moment for housekeeping.
When you're first setting things up, you'll need to tell the server to store your models and other data on the network drive rather than the login node. If you don't do this, you'll run out of disk space and nothing will work. Fortunately, there's an easy fix.


On Adroit, run the following:
```bash 
$ rsync -avu $HOME/.conda /scratch/network/$USER/
$ rm -Rf $HOME/.conda
$ cd $HOME && ln -s /scratch/network/$USER/.conda .conda
```

NVIDIA has a cache directory that can quickly take up all the space in your home directory. To move that to the network drive, run:
```bash 
$ rsync -avu $HOME/.triton /scratch/network/$USER/
$ rm -Rf $HOME/.triton
$ cd $HOME && ln -s /scratch/network/$USER/.triton .triton
```

Finally, HuggingFace needs to know about the network drive. We'll create an enviornment variable that tells 🤗 to save model data in our network cache directory. [Reserch computing recommends](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face) that you save this so that it's set automatically on boot.    
```bash 
$ echo "export HF_HOME=/scratch/network/$USER/.cache/huggingface/" >> $HOME/.bashrc
```
You'll then need to log out of adroit and log back in. 

## Image files or PDFs?
This project currently supports the processing of image collections from a IIIF endpoint like Princeton's [DPUL](https://dpul.princeton.edu/) or a folder of PDF files. 

### Image collections 

Many major libraries and museums support the International Image Interoperability Framework (IIIF). There's useful list of members of the IIIF community [here](https://iiif.io/guides/finding_resources/). In addition to providing a viewer for researchers, IIIF serves data about the collection to the web. You can access this data in a IIIF manifest. This is a package of metadata in JSON format that includes information about the materials, metadata and links for all of the images in the collection. 

For example, [this manuscript](https://www.loc.gov/resource/music.musapschmidt-10006011/?sp=1) from the Library of Congress includes a link to 

>  IIIF Presentation Manifest  
>  [Manifest (JSON/LD)](https://www.loc.gov/item/musapschmidt06012/manifest.json)  

If you click on the link, you'll go to: https://www.loc.gov/item/musapschmidt06012/manifest.json
That is the manifest's URI (Uniform Resource Identifier) a link that returns the data you need.

You may also find the IIIF logo.  !["IIIF logo. Alternating blue and red letters that spell IIIF"](https://iiif.io/assets/images/logos/logo-sm.png)  

[This item](https://eap.bl.uk/archive-file/EAP699-9-1) from the British Library's Endangered Archives Programme has the logo in the lower right corner. If you click on it, you get the JSON data for the IIIF manifest. 
For this example, the URI looks like this: https://eap.bl.uk/archive-file/EAP699-9-1/manifest?manifest=https%3A//eap.bl.uk/archive-file/EAP699-9-1/manifest

It's up to you to find relevant collections for your work and to find the IIIF manifest's URI. But, once you have that, you can download all the images and their metadata.  

This project includes a script to download a manifest and associated images. To run it, type:

```bash
python fetch.py images <IIIF manifest URI>
```

All images will be saved in the img folder.  This is the same folder that will hold the markdown files. For example `0001.jpg` will have a `0001.md` file in the same folder. 


Before running your job, you'll want to look at and update the job.slurm file. 
To open an editor in the terminal, type `nano job.slurm`
The main things to note and ajust are:
- your job name
- the number of GPUs. Start with one. If you get a CUDA memory error, you can add them as needed. For example, `gpu:2` calls for two GPUs. 
- update the email for notifications (mail-user)

The file will look something like this: 

```bash
#!/bin/bash
#SBATCH --job-name=quiche        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=2G                 # total memory (RAM) per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=apjanco@princeton.edu

module purge
module load anaconda3/2024.6
conda activate vlm
HF_HUB_OFFLINE=1
python main.py
```

When you have the model downloaded, the images ready and everything is set to go
```bash
sbatch job.slurm
```

You can view the status of your running jobs here: https://myadroit.princeton.edu/pun/sys/dashboard/activejobs

As the markdown files are generated, you will see them appear in the img folder. `ls img/`

Once a job is completed, you will get an email report on the resources used and efficentcy of your job. Based on your utilization,  

For example: 

```bash
================================================================================
                              Slurm Job Statistics
================================================================================
         Job ID: 2556969
  NetID/Account: aj7878/pustaff
       Job Name: quiche
          State: TIMEOUT
          Nodes: 1
      CPU Cores: 1
     CPU Memory: 2GB
           GPUs: 4
  QOS/Partition: gpu-short/gpu
        Cluster: adroit
     Start Time: Wed Jul 2, 2025 at 11:08 AM
       Run Time: 02:00:01
     Time Limit: 02:00:00

                              Overall Utilization
================================================================================
  CPU utilization  [|||||||||||||||||||||||||||||||||||||||||||||||99%]
  CPU memory usage [||||||||||||||||||||||||||||||||||||||         77%]
  GPU utilization  [|||||||||                                      18%]
  GPU memory usage [|||||                                          10%]
```
You can get the same information by typing `jobstats 2556969`

### PDFs 

Processing PDFs is very similar to processing images.  The `main_pdf.py` script is very similar to `main.py`. It loads files in a `pdfs` directory and checks that they are valid PDF files. Note that they don't have to have the `.pdf` suffix. We're checking the contents of the file, not its name. We then convert each page of the PDF into an image in memory.  The intermediate images aren't saved to disk. One important parameter to know about is image dpi. On line 85 of `main_pdf.py`, you'll find 

```python
pix = page.get_pixmap(dpi=100)  
```

The dpi setting controls the size and detail of the generated image. For most typed documents, 50-100 dots per square inch is more than sufficent. Relatively low resolution uses less memory and is significantly faster.  The image is converted into image tokens, so a smaller image has fewer tokens.  If the resolution is too high, the model may not be able to fit all of the image and text tokens into its context length.

For example, 
```bash 
Error opening pdfs/156002856211550797464202220250174197763: The decoder prompt (length 10896) is longer than the maximum model length of 8192. 
```

The main_pdf script will process any files that you put in `pdfs` and save the result in a `markdown` folder. The files will have the same name, but a different suffix.  

Before running your job, you'll want to look at and update the job.pdf.slurm file. 
To open an editor in the terminal, type `nano job.pdf.slurm`
The main things to note and ajust are:
- your job name
- the number of GPUs. Start with one. If you get a CUDA memory error, you can add them as needed. For example, `gpu:2` calls for two GPUs. 
- update the email for notifications (mail-user)

The file will look something like this: 

```bash
#!/bin/bash
#SBATCH --job-name=quiche        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G                 # total memory (RAM) per node
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=<username>@princeton.edu

module purge
module load anaconda3/2024.6
conda activate vlm
HF_HUB_OFFLINE=1
python main_pdf.py
```

When you have the model downloaded, the images ready and everything is set to go
```bash
sbatch job.pdf.slurm
```

## Moving your files from Adroit

To move you images and text off the Adroit servers, you can do the following 

### Create a HuggingFace Dataset and push to the hub
Log in to Huggingface with your token
```bash 
hugginface-cli login
```
Then enter your token 
Now you can push all your files to HuggingFace Hub with 
```bash
python fetch.py to-hub <your HF username>/<new repo name> 
```
By default, your dataset is private. You can publish as public by adding `--public`

### Use SCP
While connected to Adroit, find the files that you'd like to transfer.
For example, my markdown output files might be in: 
`/scratch/network/myusername/quice/markdown`

Open the terminal on your computer. 
You can copy the folder from Adroit to your computer with the `scp` command. 
`scp <origin> <destination>`
for example, 
`scp myusername@adroit.princeton.edu:/scratch/network/myusername/quice/markdown/ ~/Downloads/`

Further reading: 
- https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face

- https://github.com/davidt0x/hf_tutorial
