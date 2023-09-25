[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No. 897102
# NU-BCI

Requirements:
  - In order to read real-time data stream from AmpServer (through AmpServerClient.py), [AmpServer Pro SDK](https://www.egi.com/images/stories/placards/ASProSDK_v21_ugid_8409500-57_20181029.pdf) is required.
 
What this project is:
  - The primary goal of this project is to create a bci-system fit for our lab's needs:
    * Real time EEG data from EGI NetAmps amplifiers (specifically NA300).
    * Two-way communication with E-Prime.
    * Allow for trial-wise classification.
  - Secondary goals include:
    * Provide inspiration, possibly even a working framework, to other labs seeking to create a bci-system using the same or similar technologies.
    * Hide as much low-level stuff as possible in order to improve accessibility.
  - Technologies used:
    * Standard python libraries are used almost exclusively, the only exception being numpy.
 
What this project is not:
  - A general BCI framework.
    * This project was started since other BCI frameworks for some reason did not offer the functionality our lab needed. If a general framework is what you need, there are other, probably more suitable projects, such as [BCI2000](https://www.bci2000.org/mediawiki/index.php/Main_Page) or [Timeflux](https://timeflux.io/)
 
 Usage:
  - Installation:
    ```
    git clone https://github.com/vegardkb/NU-BCI.git
    cd NU-BCI/
    pip install -r requirements.txt
    ```
  - How to run first time:
    ```
    python config/generate_config.py
    python CommandLineInterface.py
    ```
    If you'd also like to see the debug log open a terminal or command prompt, cd into NU-BCI and try:
    ```
    tail -f log/debug.log
    ```
    or if tail doesn't work(windows):
    ```
    powershell
    Get-Content log/debug.log -Wait
    ```
Software versions:
  - Python installation required.
  - Required python packages outlined in requirements.txt
  - Tested with E-Prime 2.0, Net Station v5.4 and v4.5.
  
System layout:
[![](https://mermaid.ink/img/pako:eNqNkstqwzAQRX9FaB0vuujLhULimLSLloKzs7NQrXEiooeRR4UQ598rW86DhIC1GuaembmIu6el4UBjuras3pDlvNDEv0WeRj9WKFiRKHpvMxTKSdGSWf7hFNOrQM3zNF2QqaqlqATYwHYtzpC1JM29lIH981IYSHuiNLoSa2cZCqOblgw3k148jZBScb9j8NNrSyuYJAqQhQNJ_u2iWfK5ulxwvj_NS8maJni7NHBGktCeXbUHR9NgV1bEOKwdnvjkRlgUOkhS6G2GOwnkgTRozRZiC3xSGmlsV71dY4_jsKdx2PM47OWI_UoHA9eVN-DrfZBOqAKrmOA-PvtusKC4AQUFjX3JoWJOYkELffCoq_3XQsoFGkvjiskGJpQ5NNlOlzRG69cP0Fwwn0Z1bEI_8xVi2qf18A8JNeRm?type=png)](https://mermaid.live/edit#pako:eNqNkstqwzAQRX9FaB0vuujLhULimLSLloKzs7NQrXEiooeRR4UQ598rW86DhIC1GuaembmIu6el4UBjuras3pDlvNDEv0WeRj9WKFiRKHpvMxTKSdGSWf7hFNOrQM3zNF2QqaqlqATYwHYtzpC1JM29lIH981IYSHuiNLoSa2cZCqOblgw3k148jZBScb9j8NNrSyuYJAqQhQNJ_u2iWfK5ulxwvj_NS8maJni7NHBGktCeXbUHR9NgV1bEOKwdnvjkRlgUOkhS6G2GOwnkgTRozRZiC3xSGmlsV71dY4_jsKdx2PM47OWI_UoHA9eVN-DrfZBOqAKrmOA-PvtusKC4AQUFjX3JoWJOYkELffCoq_3XQsoFGkvjiskGJpQ5NNlOlzRG69cP0Fwwn0Z1bEI_8xVi2qf18A8JNeRm)

Code overview:
  - Top layer directory contains the code for UI and master object (CommandLineInterface.py and Operator.py), as well as git-related code.
  - /modules/ contains all slave objects in the NU-BCI framework.
  - /config/ contains the config file and a python script for setting the configuration.
  - /data/ is the location where EEG data and metainformation (predictions) are saved after an experiment.
  - /eprime/ contains E-Prime 2.1 scripts.
  - /offline/ contains all code related to ML and data analysis.

This project is worked on by NTNU's NuLab:
  - [In-house website](https://nulab-ntnu.github.io/)
  - [NTNU's website](https://www.ntnu.edu/psychology/nulab)
  
Notes:
  - The AmpServerClient module is heavily inspired by [labstreaminglayer/App-EGIAmpServer](https://github.com/labstreaminglayer/App-EGIAmpServer)
  - This project is the successor to https://github.com/dermanu/NU-BCI
