Shift from commandline to myadroit as main interface 
process on adroit compute node 

configure adroit to run jobs 
- create symlink to project folder on adroit scratch
- create conda env 
- create slurm job file template

process a single folder of files (pdf, images)
images:
- load from IIIF
- load from Drive 
model:
- load from Huggingface, update config with path to snapshot
- load from local path
export to Drive (json, csv, md ) 

