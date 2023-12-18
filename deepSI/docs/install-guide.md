# Detailed Python installation instructions with deepSI

Python is a free and open-source alternative to MATLAB which is also used for many other applications such as web server applications. Moreover, Python is the most popular language for deep learning and is what you will be using the coming exercises. 

For these exercises, you will be using python 3.7 with some extensions for streamlined array data handling (e.g `numpy`), visualization (e.g. `matplotlib`), and of course, deep learning (`PyTorch`). 

## Cloud installation using google colab

 1. Open Google Colab via https://colab.research.google.com/
 2. Login in to your Google account. 
 3. Upload the desired notebook.
 4. use "`!pip install git+git://github.com/GerbenBeintema/deepSI@master`" to install deepSI directly from the repository
 4. To save your work either save it to your Google drive or download it as a `.ipynb`.

## Local installation using Anaconda

 1. Install Anaconda. 
    * Download (~450 MB) the anaconda 3.8 python installer via [Download page](https://www.anaconda.com/products/individual) (64 bit or 32 bit dependent on your system)
    * Install by following the instructions after opening the installer. (default setting are preferred)
    * see: [problem solving with python](https://problemsolvingwithpython.com/01-Orientation/01.03-Installing-Anaconda-on-Windows/) for a detailed instructions
 2. Installing PyTorch and deepSI
    * Open the Anaconda navigator
    * launch the `Powershell Prompt` (or `CMD.exe Prompt`)
    * Create a new environment using
      * "`conda create -n ML pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`" (PyTorch with CUDA)
      * or "`conda create -n ML pytorch torchvision torchaudio cpuonly -c pytorch`"  (PyTorch without CUDA)
      * depending on if you have a CUDA compatible graphics card. See https://pytorch.org/get-started/locally/ for further details.
      * ("ML" is the name which can be changed to preference)
      * Follow install instruction on screen
      * type "`conda activate ML`" to enter the created environment
    * type "`conda install -c anaconda git ipywidgets`"
    * type "`pip install git+git://github.com/GerbenBeintema/deepSI@master`" to install deepSI in the ML environment
 4. Opening a notebook.
    * Open the anaconda navigator
    * change the active environment to ML
    * Launch the Jupyter Notebook 
    * Navigate to the desired Notebook and open. 
 5. Extra: Update deepSI
    * launch the `Powershell Prompt` (or `CMD.exe Prompt`)
    * type `pip install git+git://github.com/GerbenBeintema/deepSI@master`

After the installation one should open the `Quickstart-Tutorial.ipynb` notebook and follow the instructions. 