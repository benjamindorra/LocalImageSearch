# LocalImageSearch

A simple tool for local reverse image search
![](figures/illustration_image_search.png)

Now including VLM capabilities with the MobileCLIP2-S0 model ! ([mobileclip repo](https://github.com/apple/ml-mobileclip))

## Installation

1) clone the repository.

2) in the command line, navigate to the repository.

### Recommended: with uv

3) install uv ([installation guide](https://docs.astral.sh/uv/#installation))

4) install the dependencies: $ uv build

### Alternative installation with venv

3) create a virtual environment (python3 -m venv LocalImageSearchEnv).

4) start the environment.
   
    Linux/Mac: $ source LocalImageSearchEnv/bin/activate
  
    Windows: $ LocalImageSearchEnv\bin\activate.bat

5) install the requirements ($ python3 -m pip install -r requirements.txt).

## Usage

### Recommended: launching using uv

1) launch the software: $ uv run dbSearchGui.py

### Alternative launch with venv

1) start the environment.
   
    Linux/Mac: $ source LocalImageSearchEnv/bin/activate
   
    Windows: $ LocalImageSearchEnv\bin\activate.bat

2) launch the software ($ python3 dbSearchGui.py).

### Using the software

3) Select a directory to search in with the "Search in" button.

4) Select the number of images displayed on each page.

5) Drag-and drop an image in the grey square to start a search.

https://github.com/benjamindorra/LocalImageSearch/blob/main/figures/drag_and_drop.webm

Results will open in a new window. 

6) Alternatively, enter keywords in the search bar and press enter.
If the directory has not been indexed yet, step 5 or 6 could take some time as all images have to pass through the image analysis model. 

7) If results seem outdated, try reindexing the directory using the button in the top right.

## Troubleshooting and known errors

### Installation

Under linux, this error may appear while installing required pachages: "/usr/bin/python3: No module named pip". In this case, just install pip in your system. In debian-based systems, use $ sudo apt install python3-pip

### Known limitations

Note that the method used is an exact nearest neighbour search. It does not scale well to very large image sets. This is good enough for fast search through a few thousand images, so I don't plan to change this unless there is demand.
